"""
FastAPI app to serve FTS search endpoints (no vector search)

支持两种 FTS 索引类型：
- Tantivy FTS 索引（外部索引）
- Lance 原生 FTS 索引（内置索引）
"""
import asyncio
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import lru_cache

from config import Settings
from fastapi import FastAPI, HTTPException, Query, Request

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
    
    # 尝试加载两种 FTS 索引类型的表
    app.tables = {}
    for fts_type in ["tantivy", "lance"]:
        table_name = f"wines_{fts_type}"
        try:
            app.tables[fts_type] = db.open_table(table_name)
            print(f"Successfully loaded table '{table_name}' ({fts_type.upper()} FTS)")
        except Exception as e:
            print(f"Warning: Could not load table '{table_name}': {e}")
    
    if not app.tables:
        raise RuntimeError("No FTS tables found. Please run index.py first.")
    
    print(f"Successfully connected to LanceDB with {len(app.tables)} FTS table(s)")
    yield
    print("Successfully closed LanceDB connection and released resources")


app = FastAPI(
    title="REST API for wine reviews on LanceDB (FTS only)",
    description=(
        "Query from a LanceDB database of 130k wine reviews from the Wine Enthusiast magazine using Full-Text Search. "
        "Supports both Tantivy FTS and Lance native FTS indexes."
    ),
    version="0.3.0",
    lifespan=lifespan,
)

# --- app ---


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "REST API for querying LanceDB database of 130k wine reviews using Full-Text Search",
        "available_fts_types": list(app.tables.keys()) if hasattr(app, 'tables') else []
    }


# --- Search functions ---


def _fts_search(request: Request, terms: str, fts_type: str) -> list[dict] | None:
    """执行 FTS 搜索
    
    Args:
        request: FastAPI request object
        terms: 搜索关键词
        fts_type: FTS 索引类型 - 'tantivy' 或 'lance'
    """
    if fts_type not in request.app.tables:
        return None
    
    table = request.app.tables[fts_type]
    search_result = (
        table.search(terms, query_type="fts")
        .select(["id", "title", "description", "country", "variety", "price", "points"])
        .limit(10)
    ).to_list()
    if not search_result:
        return None
    return search_result


# --- Endpoints ---


@app.get(
    "/fts_search",
    response_model=list[dict],
    response_description="Search for wines via full-text keywords",
)
async def fts_search(
    request: Request,
    query: str = Query(
        description="Specify terms to search for in the variety, title and description"
    ),
    fts_type: str = Query(
        default="tantivy",
        description="FTS index type: 'tantivy' (external Tantivy index) or 'lance' (native Lance index)"
    ),
) -> list[dict] | None:
    # 验证 fts_type
    if fts_type not in request.app.tables:
        available_types = list(request.app.tables.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Invalid fts_type '{fts_type}'. Available types: {available_types}"
        )
    
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, _fts_search, request, query, fts_type)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No wine with the provided terms '{query}' found in database - please try again",
        )
    return result


@app.get("/available_fts_types", response_description="List available FTS index types")
async def available_fts_types(request: Request) -> dict:
    """返回可用的 FTS 索引类型"""
    return {
        "available_types": list(request.app.tables.keys()),
        "description": {
            "tantivy": "External Tantivy FTS index (Lucene + BM25 in Rust)",
            "lance": "Native Lance FTS index (built-in)"
        }
    }


# --- Search functions ---


def _fts_search(request: Request, terms: str) -> list[dict] | None:
    # In FTS, we limit to a max of 10K points to be more in line with Elasticsearch
    search_result = (
        request.app.table.search(terms, query_type="fts")
        .select(["id", "title", "description", "country", "variety", "price", "points"])
        .limit(10)
    ).to_list()
    if not search_result:
        return None
    return search_result


# --- Endpoints ---


@app.get(
    "/fts_search",
    response_model=list[dict],
    response_description="Search for wines via full-text keywords",
)
async def fts_search(
    request: Request,
    query: str = Query(
        description="Specify terms to search for in the variety, title and description"
    ),
) -> list[dict] | None:
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, _fts_search, request, query)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"No wine with the provided terms '{query}' found in database - please try again",
        )
    return result