#!/usr/bin/env python3
import argparse
import json
import os
import zipfile
from pathlib import Path
from typing import Iterable, Optional

import requests


GOOGLE_DRIVE_FILE_ID = "1XVXYLiHOaeujKo9DYXyC2krJef9YOJRA"


def _get_confirm_token(response: requests.Response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _save_response_content(response: requests.Response, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)


def download_from_google_drive(file_id: str, destination: Path) -> None:
    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(url, params={"id": file_id}, stream=True, timeout=60)
    token = _get_confirm_token(response)
    if token:
        response = session.get(
            url,
            params={"id": file_id, "confirm": token},
            stream=True,
            timeout=60,
        )
    response.raise_for_status()
    _save_response_content(response, destination)


def _find_json_candidates(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.json"):
        if path.is_file():
            yield path


def _select_source_json(root: Path, dataset_name: str) -> Path:
    candidates = list(_find_json_candidates(root))
    if not candidates:
        raise FileNotFoundError(f"No JSON files found under {root}")
    dataset_token = dataset_name.lower()
    preferred = [path for path in candidates if dataset_token in path.name.lower()]
    return max(preferred or candidates, key=lambda path: path.stat().st_size)


def _load_records(source: Path) -> list:
    with source.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return payload["data"]
    raise ValueError(f"Unsupported JSON payload in {source}")


def _split_records(records: list, chunk_size: int) -> list[list]:
    return [records[i : i + chunk_size] for i in range(0, len(records), chunk_size)]


def _validate_webqsp_schema(records: list) -> None:
    if not records:
        return
    sample = records[0]
    expected_keys = {"QuestionId", "RawQuestion", "Parses", "topic_entity"}
    missing = expected_keys.difference(sample.keys())
    if missing:
        print(
            "Warning: sample record is missing expected WebQSP keys: "
            + ", ".join(sorted(missing))
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/rog_webqsp"))
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--dataset-name", type=str, default="webqsp")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    raw_zip = output_dir / "raw" / "rog_webqsp.zip"
    raw_dir = output_dir / "raw" / "extracted"
    chunk_dir = output_dir / "chunks"

    if not raw_zip.exists():
        print(f"Downloading Google Drive file to {raw_zip}...")
        download_from_google_drive(GOOGLE_DRIVE_FILE_ID, raw_zip)
    else:
        print(f"Found existing archive: {raw_zip}")

    if not raw_dir.exists():
        print(f"Extracting archive to {raw_dir}...")
        raw_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(raw_zip, "r") as zf:
            zf.extractall(raw_dir)
    else:
        print(f"Found existing extraction directory: {raw_dir}")

    source_json = _select_source_json(raw_dir, args.dataset_name)
    print(f"Using source JSON: {source_json}")
    records = _load_records(source_json)
    _validate_webqsp_schema(records)

    chunks = _split_records(records, args.chunk_size)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        chunk_path = chunk_dir / f"{args.dataset_name}_chunk_{idx}.json"
        with chunk_path.open("w", encoding="utf-8") as f:
            json.dump(chunk, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(chunk)} records to {chunk_path}")

    metadata = {
        "source_json": str(source_json),
        "total_records": len(records),
        "chunk_size": args.chunk_size,
        "chunk_count": len(chunks),
        "chunk_dir": str(chunk_dir),
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Wrote metadata to {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
