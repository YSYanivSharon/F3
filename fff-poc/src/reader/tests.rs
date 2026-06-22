use arrow_array::{Int32Array, RecordBatch, StringArray};
use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions};
use arrow_schema::{DataType, Field, Schema};
use fff_format::{File::fff::flatbuf::CompressionType, MAJOR_VERSION, MINOR_VERSION};
use xxhash_rust::xxh64;

use crate::reader::legacy::FileReader;
use crate::writer::FileWriter;
use crate::{common::checksum::ChecksumType, options::FileWriterOptions};

use super::*;
use std::io::Seek;
use std::path::PathBuf;
use std::{io::Cursor, sync::Arc};

#[test]
#[rustfmt::skip]
fn test_read_postscript() {
    use std::io::Write;
    let mut file = tempfile::tempfile().unwrap();
    let cursor = Cursor::new(vec![
        42,0,0,0, /* metadata size */
        23,0,0,0, /* footer size */
        CompressionType::Uncompressed.into(), /* compression type */
        ChecksumType::XxHash as u8,
        17,0,0,0,0,0,0,0, /* data checksum */
        19,0,0,0,0,0,0,0, /* schema checksum */
        MAJOR_VERSION as u8, 0, /* major version */
        MINOR_VERSION as u8, 0, /* minor version */
        b'F',  b'3', /* magic */
    ]);
    file.write_all(cursor.get_ref()).unwrap();
    let mut reader = FileReader::new(file);
    let postscript = reader.read_postscript().unwrap();
    assert_eq!(postscript.footer_size, 23);
    assert_eq!(postscript.compression, CompressionType::Uncompressed);
    assert_eq!(postscript.major_version, MAJOR_VERSION);
    assert_eq!(postscript.minor_version, MINOR_VERSION);
}

#[test]
fn test_footer_roundtrip() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, true),
    ]);
    let mut hasher = xxh64::Xxh64Builder::default().build();
    let write_options = IpcWriteOptions::default();
    let mut dictionary_tracker =
        DictionaryTracker::new_with_preserve_dict_id(true, write_options.preserve_dict_id());

    hasher.update(
        &IpcDataGenerator {}
            .schema_to_bytes_with_dictionary_tracker(
                &schema,
                &mut dictionary_tracker,
                &write_options,
            )
            .ipc_message,
    );
    let schema_checksum = hasher.digest();
    let mut file = tempfile::tempfile().unwrap();
    {
        // create some data
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![5, 4, 3, 2, 1]);

        // build a record batch
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)]).unwrap();
        let mut writer =
            FileWriter::try_new(batch.schema(), &file, FileWriterOptions::default()).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.finish().unwrap();
    }
    file.rewind().unwrap();
    let mut reader = FileReader::new(file);
    let postscript = reader.read_postscript().unwrap();
    assert_eq!(postscript.checksum_type, ChecksumType::XxHash);
    assert_eq!(postscript.schema_checksum, schema_checksum);
    let footer = reader.read_footer(&postscript).unwrap_or_else(|e| {
        panic!("Error reading footer: {:?}", e);
    });
    assert_eq!(footer.schema().fields().len(), 2);
    assert_eq!(footer.schema().fields()[0].name(), "a");
    // let row_groups = footer.row_groups();
    // assert_eq!(row_groups.row_counts().unwrap().len(), 1);
    // assert_eq!(row_groups.offsets().unwrap().len(), 1);
    // assert_eq!(row_groups.row_counts().unwrap().get(0), 5);
    // assert_eq!(row_groups.offsets().unwrap().get(0), 0);
    // let column_metadata_sections = row_groups
    //     .row_group_metadatas()
    //     .unwrap()
    //     .get(0)
    //     .col_metadatas()
    //     .unwrap();
    // assert_eq!(column_metadata_sections.len(), 2);
    let column_metadata = footer.row_group_metadatas()[0].column_metadatas[0];
    let column_chunk = column_metadata.column_chunks().unwrap().get(0);
    println!("{:?}", column_chunk);
    assert_eq!(column_chunk.offset(), 0);
    assert_eq!(column_chunk.num_rows(), 5);
}

/// This test requires the file to be created manually.
/// Then modify the version map in footer.rs (both lower and higher than before) to test version incompatibility.
#[test]
#[ignore]
fn test_version_incompatibility() {
    let schema = Schema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int32, true),
    ]);
    let path = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .parent()
        .unwrap()
        .join("data")
        .join("compatibility.fff");
    let file = std::fs::File::create(path.clone()).unwrap();
    {
        // create some data
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![5, 4, 3, 2, 1]);

        // build a record batch
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(a), Arc::new(b)]).unwrap();
        let mut writer =
            FileWriter::try_new(batch.schema(), &file, FileWriterOptions::default()).unwrap();
        writer.write_batch(&batch).unwrap();
        writer.finish().unwrap();
    }
    let file = std::fs::File::open(path).unwrap();
    let mut reader = FileReaderV2Builder::new(Arc::new(file)).build().unwrap();
    let _output_batches = reader.read_file().unwrap();
}

/// Streaming a scan one ROW GROUP at a time must yield exactly the same rows,
/// in the same order, as reading the whole file. Uses a finite `row_group_size`
/// so the file has multiple row groups, exercising the per-row-group streaming
/// path that bounds scan memory.
#[test]
fn test_for_each_row_group_matches_read_file() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int32, false),
        Field::new("s", DataType::Utf8, true),
    ]));
    // Finite row groups of 8 rows; 5 write_batch calls of 4 rows = 20 rows total,
    // which the writer splits into 3 row groups (8 + 8 + 4).
    let options = FileWriterOptions::builder().set_row_group_size(8).build();
    let mut file = tempfile::tempfile().unwrap();
    {
        let mut writer = FileWriter::try_new(schema.clone(), &file, options).unwrap();
        for g in 0..5i32 {
            let base = g * 4;
            let n = Int32Array::from((base..base + 4).collect::<Vec<i32>>());
            let s =
                StringArray::from((base..base + 4).map(|i| format!("row-{i}")).collect::<Vec<_>>());
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(n), Arc::new(s)]).unwrap();
            writer.write_batch(&batch).unwrap();
        }
        writer.finish().unwrap();
    }
    file.rewind().unwrap();
    let file = Arc::new(file);

    // row_count() (used by `info`) reads row-group metadata without decoding.
    let total_rows = FileReaderV2Builder::new(file.clone()).build().unwrap().row_count();
    assert_eq!(total_rows, 20, "expected 20 rows total");

    // Reference: whole-file read.
    let mut full = FileReaderV2Builder::new(file.clone()).build().unwrap();
    let reference = full.read_file().unwrap();
    assert_eq!(reference.iter().map(|b| b.num_rows()).sum::<usize>(), 20);
    assert!(
        reference.len() > 1,
        "test needs multiple row groups, got {}",
        reference.len()
    );

    // Streaming: one row group at a time (bounded memory).
    let mut reader = FileReaderV2Builder::new(file.clone()).build().unwrap();
    let mut streamed = vec![];
    reader
        .for_each_row_group(|b| {
            streamed.push(b);
            Ok(())
        })
        .unwrap();

    assert_eq!(streamed, reference, "streamed read must equal whole-file read");
}
