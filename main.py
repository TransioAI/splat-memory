"""FastAPI server and CLI entry point for splat-memory."""

import argparse

import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="Splat Memory",
    description="Single-image spatial reasoning: 3D scene understanding from a single photo",
    version="0.1.0",
)


# --- API Endpoints ---


@app.post("/analyze")
async def analyze_image():
    """Analyze an image and return a scene graph."""
    pass


@app.post("/ask")
async def ask_question():
    """Ask a spatial reasoning question about a scene."""
    pass


# --- CLI ---


def main():
    parser = argparse.ArgumentParser(description="Splat Memory: Single-image spatial reasoning")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive Q&A loop")
    parser.add_argument("--serve", action="store_true", help="Start FastAPI server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        print("Use --serve to start the API server, or --image <path> to analyze an image.")


if __name__ == "__main__":
    main()
