import argparse
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import srsly
from codetiming import Timer
from config import Settings
from dotenv import load_dotenv
from rich import progress
from schemas.wine import LanceModelWine, Wine

import lancedb
from lancedb.pydantic import pydantic_to_schema
from lancedb.table import Table

load_dotenv()
# Custom types
JsonBlob = dict[str, Any]


class FileNotFoundError(Exception):
    pass


@lru_cache()
def get_settings():
    # Use lru_cache to avoid loading .env file for every request
    return Settings()


def chunk_iterable(item_list: list[JsonBlob], chunksize: int) -> Iterator[list[JsonBlob]]:
    """
    Break a large iterable into an iterable of smaller iterables of size `chunksize`
    """
    for i in range(0, len(item_list), chunksize):
        yield item_list[i : i + chunksize]


def get_json_data(data_dir: Path, filename: str) -> list[JsonBlob]:
    """Get all line-delimited json files (.jsonl) from a directory with a given prefix"""
    file_path = data_dir / filename
    if not file_path.is_file():
        # File may not have been uncompressed yet so try to do that first
        data = srsly.read_gzip_jsonl(file_path)
        # This time if it isn't there it really doesn't exist
        if not file_path.is_file():
            raise FileNotFoundError(f"No valid .jsonl file found in `{data_dir}`")
    else:
        data = srsly.read_gzip_jsonl(file_path)
    return data


def validate(
    data: list[JsonBlob],
    exclude_none: bool = False,
) -> list[JsonBlob]:
    validated_data = [Wine(**item).model_dump(exclude_none=exclude_none) for item in data]
    return validated_data


def insert_batches(tbl: Table, validated_data: list[JsonBlob]) -> None:
    """Ingest data in batches for FTS index"""
    chunked_data = chunk_iterable(validated_data, CHUNKSIZE)
    print(f"Adding data to table for FTS index...")
    # Add rich progress bar
    with progress.Progress(
        "[progress.description]{task.description}",
        progress.BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        progress.TimeElapsedColumn(),
    ) as prog:
        overall_progress_task = prog.add_task(
            "Inserting data...", total=len(validated_data) // CHUNKSIZE
        )
        for chunk in chunked_data:
            prog.update(overall_progress_task, advance=1)
            tbl.add(chunk, mode="append")


def main(tbl: Table, data: list[JsonBlob]) -> None:
    """Create FTS index only (no vector index)"""
    with Timer(
        name="Data validation in pydantic",
        text="Validated data using Pydantic in {:.4f} sec",
    ):
        validated_data = validate(data, exclude_none=False)

    with Timer(
        name="Insert data in batches",
        text="Inserted data in {:.4f} sec",
    ):
        insert_batches(tbl, validated_data)
        print(f"Finished inserting {len(tbl)} records into LanceDB table")

    with Timer(name="Create FTS index", text="Created FTS index in {:.4f} sec"):
        # Create a full-text search index via Tantivy (which implements Lucene + BM25 in Rust)
        print("Creating FTS index...")
        tbl.create_fts_index(["to_vectorize"])


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser("Bulk index database from the wine reviews JSONL data")
    parser.add_argument("--limit", "-l", type=int, default=0, help="Limit the size of the dataset to load for testing purposes")
    parser.add_argument("--chunksize", type=int, default=1000, help="Size of each chunk to break the dataset into before processing")
    parser.add_argument("--filename", type=str, default="winemag-data-130k-v2.jsonl.gz", help="Name of the JSONL zip file to use")
    args = vars(parser.parse_args())
    # fmt: on

    LIMIT = args["limit"]
    DATA_DIR = Path(__file__).parents[1] / "data"
    FILENAME = args["filename"]
    CHUNKSIZE = args["chunksize"]

    data = list(get_json_data(DATA_DIR, FILENAME))
    assert data, "No data found in the specified file"
    data = data[:LIMIT] if LIMIT > 0 else data

    DB_NAME = "./winemag"
    TABLE = "wines"
    if os.path.exists(DB_NAME):
        shutil.rmtree(DB_NAME)

    db = lancedb.connect(DB_NAME)
    try:
        tbl = db.create_table(TABLE, schema=pydantic_to_schema(LanceModelWine), mode="create")
    except OSError:
        tbl = db.open_table(TABLE)

    main(tbl, data)
    print("Finished execution!")