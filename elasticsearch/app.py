"""
FastAPI app to serve FTS search endpoints (no vector search)
"""
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from functools import lru_cache

from config import Settings
from fastapi import FastAPI, HTTPException, Query, Request
from schemas.wine import SearchResult

from elasticsearch import AsyncElasticsearch


@lru_cache()
def get_settings():
    # Use lru_cache to avoid loading .env file for every request
    return Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Async context manager for Elasticsearch connection."""
    settings = get_settings()

    username = settings.elastic_user
    password = settings.elastic_password
    port = settings.elastic_port
    service = settings.elastic_url
    elastic_client = AsyncElasticsearch(
        f"http://{service}:{port}",
        basic_auth=(username, password),
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
        verify_certs=False,
    )
    app.client = elastic_client
    print("Successfully connected to Elasticsearch")
    yield
    await elastic_client.close()
    print("Successfully closed Elasticsearch connection")


app = FastAPI(
    title="REST API for wine reviews on Elasticsearch (FTS only)",
    description=(
        "Query from an Elasticsearch database of 130k wine reviews from the Wine Enthusiast magazine using Full-Text Search"
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# --- app ---


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "REST API for querying Elasticsearch database of 130k wine reviews using Full-Text Search"
    }


# --- Search functions ---


async def _fts_search(request: Request, query: str) -> list[SearchResult] | None:
    response = await request.app.client.search(
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
    result = await _fts_search(request, query)

    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No wine with the provided terms '{query}' found in database - please try again",
        )
    return result