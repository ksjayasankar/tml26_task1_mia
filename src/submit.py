"""Leaderboard submission helper.

Server: http://34.63.153.158/submit/01-mia
Auth:   X-API-Key header. Read from $TML_API_KEY (or .env).
Input:  CSV with columns [id, score], score in [0,1].

Uses pandas for the CSV write so the on-disk bytes match what the
original task_template.py produced (the server's parser was first
tested against that). dotenv and requests are lazy-imported so this
module imports cleanly inside the bare pytorch docker image; both are
only needed when actually submitting.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Submission server. Updated 2026-04-27 from the prior 35.192.205.84:80 to
# 34.63.153.158 per the new task_template.py shipped by the instructors.
BASE_URL = "http://34.63.153.158"
TASK_ID = "01-mia"


def _maybe_load_dotenv():
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
        load_dotenv()
    except ImportError:
        pass


def _require_api_key() -> str:
    _maybe_load_dotenv()
    key = os.environ.get("TML_API_KEY", "").strip()
    if not key:
        sys.exit("TML_API_KEY not set. Add it to .env or export it.")
    return key


def write_submission_csv(ids, scores, out_path: Path | str) -> Path:
    """Write a [id, score] CSV. ids may be ints or strings; scores must be in [0,1]."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"id": [str(i) for i in ids], "score": list(scores)})
    if not df["score"].between(0.0, 1.0).all():
        bad = df.loc[~df["score"].between(0.0, 1.0), "score"].iloc[0]
        raise ValueError(f"All scores must lie in [0, 1]; got {bad}")
    if df["id"].duplicated().any():
        raise ValueError("Duplicate ids found in submission.")
    df.to_csv(out_path, index=False)
    return out_path


def submit_csv(csv_path: Path | str, base_url: str = BASE_URL, task_id: str = TASK_ID) -> dict:
    """POST a CSV to the leaderboard server. Returns the parsed JSON response."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        sys.exit(f"File not found: {csv_path}")
    api_key = _require_api_key()
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("requests not installed; pip install requests (or run from a host that has it)")
    with open(csv_path, "rb") as f:
        resp = requests.post(
            f"{base_url}/submit/{task_id}",
            headers={"X-API-Key": api_key},
            files={"file": (csv_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}
    if resp.status_code == 413:
        sys.exit("Upload rejected: file too large (HTTP 413).")
    resp.raise_for_status()
    return body


def main():
    import argparse
    p = argparse.ArgumentParser(description="Submit a CSV to the TML MIA leaderboard.")
    p.add_argument("csv", type=Path, help="Path to submission CSV")
    args = p.parse_args()
    body = submit_csv(args.csv)
    print("Successfully submitted.")
    print("Server response:", body)
    sid = body.get("submission_id") if isinstance(body, dict) else None
    if sid:
        print(f"Submission ID: {sid}")


if __name__ == "__main__":
    main()
