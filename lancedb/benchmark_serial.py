"""
Run this script to benchmark the serial search performance of FTS search (no vector search)
"""
import argparse
import random
from pathlib import Path
from typing import Any

from codetiming import Timer
from rich import progress
from schemas.wine import SearchResult

import lancedb
from lancedb.table import Table

# Custom types
JsonBlob = dict[str, Any]


def get_query_terms(filename: str) -> list[str]:
    assert filename.endswith(".txt")
    query_terms_file = Path("./benchmark_queries") / filename
    with open(query_terms_file, "r") as f:
        queries = f.readlines()
    assert queries
    result = [query.strip() for query in queries]
    return result


def fts_search(table: Table, query: str) -> list[SearchResult] | None:
    search_result = (
        table.search(query, vector_column_name="description")
        .select(["id", "title", "description", "country", "variety", "price", "points"])
        .limit(10)
    ).to_pydantic(SearchResult)
    if not search_result:
        return None
    return search_result


def main():
    queries = get_query_terms("keyword_terms.txt")
    random_choice_queries = [random.choice(queries) for _ in range(LIMIT)]

    # Run the search directly on the lancedb table
    with Timer(name="Serial search", text="Finished search in {:.4f} sec"):
        # Add rich progress bar
        with progress.Progress(
            "[progress.description]{task.description}",
            progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            progress.TimeElapsedColumn(),
        ) as prog:
            overall_progress_task = prog.add_task(
                "Performing FTS search", total=len(random_choice_queries)
            )
            for query in random_choice_queries:
                _ = fts_search(tbl, query)
                prog.update(overall_progress_task, advance=1)


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=37, help="Seed for random number generator")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Number of search terms to randomly generate")
    args = parser.parse_args()
    # fmt: on

    LIMIT = args.limit
    SEED = args.seed

    # Assumes that the table in the DB has already been created
    DB_NAME = "./winemag"
    TABLE = "wines"
    db = lancedb.connect(DB_NAME)
    tbl = db.open_table(TABLE)

    main()
