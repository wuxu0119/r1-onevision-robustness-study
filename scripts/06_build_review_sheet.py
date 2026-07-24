from __future__ import annotations

import argparse
import ast
import html
import shutil
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="results/analysis/manual_error_annotations.csv",
    )
    parser.add_argument("--output-dir", default="results/review")
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def first_image_path(value: str) -> Path | None:
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    if not isinstance(parsed, list) or not parsed:
        return None
    return Path(parsed[0])


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = root / args.input
    output_dir = root / args.output_dir
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path).head(args.limit)
    cards = []
    for row_number, row in frame.iterrows():
        source = first_image_path(str(row.get("image_paths", "")))
        relative_image = ""
        if source and source.exists():
            destination = image_dir / f"{row_number:03d}_{source.name}"
            shutil.copy2(source, destination)
            relative_image = destination.relative_to(output_dir).as_posix()
        cards.append(
            f"""
            <article class="card">
              <h2>{html.escape(str(row.get("sample_id", "")))}
                  <small>{html.escape(str(row.get("condition", "")))}</small></h2>
              {f'<img src="{relative_image}" alt="input image">' if relative_image else ''}
              <h3>Question</h3>
              <pre>{html.escape(str(row.get("question", "")))}</pre>
              <p><b>Ground truth:</b> {html.escape(str(row.get("answer", "")))}</p>
              <p><b>Extracted:</b> {html.escape(str(row.get("extracted_answer", "")))}</p>
              <h3>Model output</h3>
              <pre>{html.escape(str(row.get("output_text", "")))}</pre>
              <p><b>Error class:</b> P / F / R / A / U</p>
            </article>
            """
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>R1-Onevision error review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f4f5f7; }}
.card {{ max-width: 1100px; margin: 0 auto 28px; padding: 24px;
         background: white; border: 1px solid #d9dde3; border-radius: 12px; }}
img {{ max-width: 100%; max-height: 640px; display: block; margin: 16px auto; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #f7f7f8;
       padding: 12px; border-radius: 8px; }}
small {{ color: #667085; font-weight: normal; }}
</style>
</head>
<body>
<h1>R1-Onevision manual error review</h1>
{''.join(cards)}
</body>
</html>
"""
    output_path = output_dir / "index.html"
    output_path.write_text(document, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()

