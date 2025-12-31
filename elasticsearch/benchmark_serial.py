"""
Run this script to benchmark the serial search performance of FTS search (no vector search)
"""
import argparse
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

from codetiming import Timer
from config import Settings
from dotenv import load_dotenv
from rich import progress
from schemas.wine import SearchResult

from elasticsearch import Elasticsearch

load_dotenv()
# Custom types
JsonBlob = dict[str, Any]


@lru_cache()
def get_settings():
    # Use lru_cache to avoid loading .env file for every request
    return Settings()


def get_query_terms(filename: str) -> list[str]:
    assert filename.endswith(".txt")
    query_terms_file = Path("./benchmark_queries") / filename
    with open(query_terms_file, "r") as f:
        queries = f.readlines()
    assert queries
    result = [query.strip() for query in queries]
    return result


def get_elastic_client(settings) -> Elasticsearch:
    # Get environment variables
    USERNAME = settings.elastic_user
    PASSWORD = settings.elastic_password
    PORT = settings.elastic_port
    ELASTIC_URL = settings.elastic_url
    # Connect to ElasticSearch
    elastic_client = Elasticsearch(
        f"http://{ELASTIC_URL}:{PORT}",
        basic_auth=(USERNAME, PASSWORD),
        request_timeout=300,
        max_retries=3,
        retry_on_timeout=True,
        verify_certs=False,
    )
    return elastic_client


def fts_search(client: Elasticsearch, query: str) -> list[SearchResult] | None:
    response = client.search(
        index="wines",
        size=10,
        query={
            "match": {
                "description": {
                    "query": query,
                }
            }
        },
        source=["id", "title", "description", "country", "variety", "price", "points"],
    )
    result = response["hits"].get("hits")
    if result:
        return [item["_source"] for item in result]
    else:
        return None


def main():
    queries = get_query_terms("keyword_terms.txt")
    random_choice_queries = [random.choice(queries) for _ in range(LIMIT)]

    # Run the search directly on the Elasticsearch DB
    elastic_client = get_elastic_client(get_settings())
    assert elastic_client.ping()

    # Run the search directly on the Elasticsearch DB
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
                _ = fts_search(elastic_client, query)
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

    main()
