#!/usr/bin/env python3
"""
scripts/run_api.py
===================
Start the Metal Defect Detection FastAPI server.

Run:
    python scripts/run_api.py
    python scripts/run_api.py --port 8080
    python scripts/run_api.py --host 0.0.0.0 --port 8000

Then open your browser at:
    http://localhost:8000          ← frontend home page
    http://localhost:8000/api/health   ← API health check
    http://localhost:8000/docs         ← auto-generated API docs (Swagger UI)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from dotenv import load_dotenv
load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start the Metal Defect Detection API server.",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to. Use 0.0.0.0 to allow access from other devices on the network.",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Port to listen on (default: 8000).",
    )
    parser.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload on code changes (development only).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of worker processes (default: 1). Keep at 1 for GPU inference.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import uvicorn
    except ImportError:
        print("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    print("=" * 60)
    print("  Metal Defect Detection API")
    print("=" * 60)
    print(f"  URL        : http://{args.host}:{args.port}")
    print(f"  Frontend   : http://{args.host}:{args.port}/home.html")
    print(f"  API Docs   : http://{args.host}:{args.port}/docs")
    print(f"  Health     : http://{args.host}:{args.port}/api/health")
    print("=" * 60)
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        log_level="warning",    # Suppress uvicorn's own verbose logs
    )


if __name__ == "__main__":
    main()