"""
Run this script to benchmark the serial search performance of FTS search (no vector search)

Usage:
    # 通过 HTTP API 调用 Tantivy FTS（需先启动 app.py，与 ES 测试方式一致）
    python benchmark_serial.py --limit 1000 --fts-type tantivy

    # 通过 HTTP API 调用 Lance FTS
    python benchmark_serial.py --limit 1000 --fts-type lance

    # 直接嵌入式调用（无网络开销，测试纯 LanceDB 性能）
    python benchmark_serial.py --limit 1000 --direct --fts-type tantivy

    # 对比两种 FTS 索引（直接模式）
    python benchmark_serial.py --limit 1000 --direct --fts-type both
"""
import argparse
import random
from pathlib import Path
from typing import Any

import httpx
from codetiming import Timer
from rich import progress
from rich.console import Console
from rich.table import Table as RichTable

import lancedb
from lancedb.table import Table

# Custom types
JsonBlob = dict[str, Any]
console = Console()


def get_query_terms(filename: str) -> list[str]:
    assert filename.endswith(".txt")
    query_terms_file = Path("./benchmark_queries") / filename
    with open(query_terms_file, "r") as f:
        queries = f.readlines()
    assert queries
    result = [query.strip() for query in queries]
    return result


def fts_search_direct(table: Table, query: str) -> list[JsonBlob] | None:
    """直接嵌入式调用 LanceDB（无网络开销）"""
    search_result = (
        table.search(query, query_type="fts")
        .select(["id", "title", "description", "country", "variety", "price", "points"])
        .limit(10)
    ).to_list()
    if not search_result:
        return None
    return search_result


def fts_search_http(client: httpx.Client, query: str, base_url: str, fts_type: str) -> list[JsonBlob] | None:
    """通过 HTTP API 调用（与 ES 测试方式一致）"""
    response = client.get(f"{base_url}/fts_search", params={"query": query, "fts_type": fts_type})
    if response.status_code == 200:
        return response.json()
    elif response.status_code == 404:
        return None
    else:
        response.raise_for_status()


def run_benchmark_direct(db_name: str, fts_type: str, queries: list[str]) -> float:
    """运行直接嵌入式模式的 benchmark，返回耗时（秒）"""
    table_name = f"wines_{fts_type}"
    db = lancedb.connect(db_name)
    
    try:
        tbl = db.open_table(table_name)
    except Exception as e:
        console.print(f"[red]Error: Cannot open table '{table_name}': {e}[/red]")
        console.print(f"[yellow]Please run 'python index.py --fts-type {fts_type}' first[/yellow]")
        return -1
    
    with Timer(name=f"Serial search ({fts_type})", text=f"Finished {fts_type.upper()} FTS search in {{:.4f}} sec") as timer:
        with progress.Progress(
            "[progress.description]{task.description}",
            progress.BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            progress.TimeElapsedColumn(),
        ) as prog:
            overall_progress_task = prog.add_task(
                f"Performing {fts_type.upper()} FTS search (direct)", total=len(queries)
            )
            for query in queries:
                _ = fts_search_direct(tbl, query)
                prog.update(overall_progress_task, advance=1)
    
    return timer.last


def run_benchmark_http(base_url: str, fts_type: str, queries: list[str]) -> float:
    """运行 HTTP API 模式的 benchmark，返回耗时（秒）"""
    with httpx.Client(timeout=30.0) as client:
        # 测试连接
        try:
            response = client.get(f"{base_url}/")
            response.raise_for_status()
        except httpx.RequestError as e:
            console.print(f"[red]Error: Cannot connect to {base_url}[/red]")
            console.print(f"[yellow]Please start the server first with: python app.py[/yellow]")
            return -1

        # 验证 fts_type 是否可用
        try:
            response = client.get(f"{base_url}/available_fts_types")
            available_types = response.json().get("available_types", [])
            if fts_type not in available_types:
                console.print(f"[red]Error: FTS type '{fts_type}' not available. Available: {available_types}[/red]")
                return -1
        except Exception:
            pass  # 忽略检查错误，继续执行

        with Timer(name=f"Serial search ({fts_type})", text=f"Finished {fts_type.upper()} FTS search in {{:.4f}} sec") as timer:
            with progress.Progress(
                "[progress.description]{task.description}",
                progress.BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                progress.TimeElapsedColumn(),
            ) as prog:
                overall_progress_task = prog.add_task(
                    f"Performing {fts_type.upper()} FTS search (HTTP)", total=len(queries)
                )
                for query in queries:
                    _ = fts_search_http(client, query, base_url, fts_type)
                    prog.update(overall_progress_task, advance=1)
        
        return timer.last


def main():
    queries = get_query_terms("keyword_terms.txt")
    random.seed(SEED)
    random_choice_queries = [random.choice(queries) for _ in range(LIMIT)]

    # 确定要测试的 FTS 类型
    fts_types_to_test = ["tantivy", "lance"] if FTS_TYPE == "both" else [FTS_TYPE]
    
    results = {}
    
    for fts_type in fts_types_to_test:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]Testing {fts_type.upper()} FTS Index[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
        
        if DIRECT_MODE:
            console.print(f"Mode: Direct (embedded LanceDB, no network overhead)")
            elapsed = run_benchmark_direct(DB_NAME, fts_type, random_choice_queries)
        else:
            console.print(f"Mode: HTTP API (connecting to {BASE_URL})")
            elapsed = run_benchmark_http(BASE_URL, fts_type, random_choice_queries)
        
        if elapsed > 0:
            results[fts_type] = elapsed
    
    # 如果测试了多种类型，显示对比结果
    if len(results) > 1:
        console.print(f"\n[bold green]{'='*60}[/bold green]")
        console.print("[bold]Benchmark Results Comparison[/bold]")
        console.print(f"[bold green]{'='*60}[/bold green]")
        
        table = RichTable(title=f"FTS Performance Comparison ({LIMIT} queries)")
        table.add_column("FTS Type", style="cyan")
        table.add_column("Total Time (sec)", style="magenta")
        table.add_column("Avg Time per Query (ms)", style="green")
        table.add_column("Queries per Second", style="yellow")
        
        for fts_type, elapsed in results.items():
            avg_ms = (elapsed / LIMIT) * 1000
            qps = LIMIT / elapsed
            table.add_row(
                fts_type.upper(),
                f"{elapsed:.4f}",
                f"{avg_ms:.2f}",
                f"{qps:.2f}"
            )
        
        console.print(table)
        
        # 显示性能差异
        if "tantivy" in results and "lance" in results:
            diff = results["lance"] / results["tantivy"]
            if diff > 1:
                console.print(f"\n[yellow]Lance FTS is {diff:.2f}x slower than Tantivy FTS[/yellow]")
            else:
                console.print(f"\n[green]Lance FTS is {1/diff:.2f}x faster than Tantivy FTS[/green]")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description="Benchmark FTS search performance")
    parser.add_argument("--seed", type=int, default=37, help="Seed for random number generator")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Number of search terms to randomly generate")
    parser.add_argument("--direct", "-d", action="store_true", help="Use direct embedded LanceDB (no network overhead)")
    parser.add_argument("--host", type=str, default="localhost", help="API server host")
    parser.add_argument("--port", type=int, default=8000, help="API server port")
    parser.add_argument("--fts-type", type=str, choices=["tantivy", "lance", "both"], default="tantivy",
                        help="FTS index type to test: 'tantivy', 'lance', or 'both' for comparison")
    args = parser.parse_args()
    # fmt: on

    LIMIT = args.limit
    SEED = args.seed
    DIRECT_MODE = args.direct
    BASE_URL = f"http://{args.host}:{args.port}"
    FTS_TYPE = args.fts_type

    # LanceDB 配置（仅 direct 模式使用）
    DB_NAME = "./winemag"

    main()
