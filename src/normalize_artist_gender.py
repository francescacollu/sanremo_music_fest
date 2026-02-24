from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def normalize_artist_gender(
    artist_names_path: Optional[Path] = None,
    artist_gender_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Normalize artist_gender.csv to a per-artist table keyed by artist_id.

    The output has one row per (artist_id, artist_name) with associated gender/notes,
    including members of bracketed group/duet entries.
    """
    project_root = _project_root()
    if artist_names_path is None:
        artist_names_path = project_root / "data" / "processed" / "artist_names.csv"
    if artist_gender_path is None:
        artist_gender_path = project_root / "data" / "processed" / "artist_gender.csv"
    if output_path is None:
        output_path = project_root / "data" / "processed" / "artist_gender_by_artist.csv"

    artist_names = pd.read_csv(artist_names_path)
    if "artist_id" not in artist_names.columns or "artist_name" not in artist_names.columns:
        raise ValueError("artist_names.csv must contain 'artist_id' and 'artist_name' columns.")

    artist_names["key"] = artist_names["artist_name"].astype(str).str.strip()

    gender = pd.read_csv(artist_gender_path)
    if "artist_name" not in gender.columns:
        raise ValueError("artist_gender.csv must contain 'artist_name' column.")

    gender["artist_name"] = gender["artist_name"].astype(str).str.strip()

    rows: list[dict] = []
    for idx, row in gender.iterrows():
        raw_name = row.get("artist_name")
        if pd.isna(raw_name):
            continue
        s = str(raw_name).strip()
        if not s:
            continue

        gender_value = row.get("gender")
        notes_value = row.get("notes")
        # 1-based data-row index; header is considered row 1 in the CSV file.
        source_row = int(idx) + 2

        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            members = [p.strip() for p in inner.split(",") if p.strip()]
            for member in members:
                rows.append(
                    {
                        "artist_name": member,
                        "gender": gender_value,
                        "notes": notes_value,
                        "source_row": source_row,
                    }
                )
        else:
            rows.append(
                {
                    "artist_name": s,
                    "gender": gender_value,
                    "notes": notes_value,
                    "source_row": source_row,
                }
            )

    if not rows:
        normalized = pd.DataFrame(columns=["artist_id", "artist_name", "gender", "notes", "source_row"])
        normalized.to_csv(output_path, index=False)
        return normalized

    expanded = pd.DataFrame(rows)
    expanded["key"] = expanded["artist_name"].astype(str).str.strip()

    merged = expanded.merge(
        artist_names[["artist_id", "key"]],
        on="key",
        how="left",
    )
    merged = merged.drop(columns=["key"])

    cols = ["artist_id", "artist_name", "gender", "notes", "source_row"]
    merged = merged[cols]
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == "__main__":
    normalize_artist_gender()

