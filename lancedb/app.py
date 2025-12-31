"""
FastAPI app to serve FTS search endpoints (no vector search)
"""
import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache

from config import Settings
from fastapi import FastAPI, HTTPException, Query, Request
from schemas.wine import SearchResult

import lancedb

executor = ThreadPoolExecutor(max_workers=4)


@lru_cache()
def get_settings():
    # Use lru_cache to avoid loading .env file for every request
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for lancedb connection."""
    # Define LanceDB client
    db = lancedb.connect("./winemag")
    app.table = db.open_table("wines")
    print("Successfully connected to LanceDB")
    yield
    print("Successfully closed LanceDB connection and released resources")


app = FastAPI(
    title="REST API for wine reviews on LanceDB (FTS only)",
    description=(
        "Query from a LanceDB database of 130k wine reviews from the Wine Enthusiast magazine using Full-Text Search"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# --- app ---


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "REST API for querying LanceDB database of 130k wine reviews using Full-Text Search"
    }


# --- Search functions ---


def _fts_search(request: Request, terms: str) -> list[SearchResult] | None:
    # In FTS, we limit to a max of 10K points to be more in line with Elasticsearch
    search_result = (
        request.app.table.search(terms, vector_column_name="description")
        .select(["id", "title", "description", "country", "variety", "price", "points"])
        .limit(10)
    ).to_pydantic(SearchResult)
    if not search_result:
        return None
    return search_result


# --- Endpoints ---


@app.get(
    "/fts_search",
    response_model=list[SearchResult],
    response_description="Search for wines via full-text keywords",
)
async def fts_search(
    request: Request,
    query: str = Query(
        description="Specify terms to search for in the variety, title and description"
    ),
) -> list[SearchResult] | None:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, _fts_search, request, query)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No wine with the provided terms '{query}' found in database - please try again",
        )
    return result