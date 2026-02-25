from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_artist_aliases(processed_dir: Optional[Path] = None) -> Optional[dict[str, str]]:
    """Load variant -> canonical_name from artist_aliases.csv. Return None if missing or empty."""
    if processed_dir is None:
        processed_dir = _project_root() / "data" / "processed"
    path = processed_dir / "artist_aliases.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "name_variant" not in df.columns or "canonical_name" not in df.columns:
        return None
    df["name_variant"] = df["name_variant"].astype(str).str.strip()
    df["canonical_name"] = df["canonical_name"].astype(str).str.strip()
    df = df[df["name_variant"].ne("")]
    if df.empty:
        return None
    return df.set_index("name_variant")["canonical_name"].to_dict()


def normalize_artist_gender(
    artist_names_path: Optional[Path] = None,
    artist_gender_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Normalize artist_gender.csv to a per-artist table keyed by artist_id.

    Resolves artist_name through artist_aliases.csv to canonical form before joining,
    so gender entered for variants (e.g. "B. Antonacci") is applied to the canonical artist.
    Deduplicates by canonical name (keeps first non-empty gender).
    """
    project_root = _project_root()
    if processed_dir is None:
        processed_dir = project_root / "data" / "processed"
    if artist_names_path is None:
        artist_names_path = processed_dir / "artist_names.csv"
    if artist_gender_path is None:
        artist_gender_path = processed_dir / "artist_gender.csv"
    if output_path is None:
        output_path = processed_dir / "artist_gender_by_artist.csv"

    artist_names = pd.read_csv(artist_names_path)
    if "artist_id" not in artist_names.columns or "artist_name" not in artist_names.columns:
        raise ValueError("artist_names.csv must contain 'artist_id' and 'artist_name' columns.")

    artist_names["key"] = artist_names["artist_name"].astype(str).str.strip()

    alias_map = _load_artist_aliases(processed_dir)

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
        source_row = int(idx) + 2

        # Whole-cell alias first: e.g. "[Elio, le Storie Tese]" -> "Elio e le Storie Tese"
        if alias_map and s in alias_map:
            canonical = alias_map[s]
            rows.append(
                {
                    "artist_name": canonical,
                    "gender": gender_value,
                    "notes": notes_value,
                    "source_row": source_row,
                }
            )
            continue
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1]
            members = [p.strip() for p in inner.split(",") if p.strip()]
            for member in members:
                canonical = alias_map.get(member, member) if alias_map else member
                rows.append(
                    {
                        "artist_name": canonical,
                        "gender": gender_value,
                        "notes": notes_value,
                        "source_row": source_row,
                    }
                )
        else:
            canonical = alias_map.get(s, s) if alias_map else s
            rows.append(
                {
                    "artist_name": canonical,
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
    expanded["artist_name"] = expanded["artist_name"].astype(str).str.strip()

    # Deduplicate by canonical name: keep first non-empty gender per artist
    expanded = expanded.sort_values(
        "gender",
        key=lambda s: s.isna() | (s.astype(str).str.strip() == ""),
        na_position="last",
    )
    expanded = expanded.drop_duplicates(subset=["artist_name"], keep="first")
    expanded["key"] = expanded["artist_name"]

    merged = expanded.merge(
        artist_names[["artist_id", "key"]],
        on="key",
        how="left",
    )
    merged = merged.drop(columns=["key"])

    cols = ["artist_id", "artist_name", "gender", "notes", "source_row"]
    merged = merged[cols]
    # Write artist_id as integer (no decimal)
    merged["artist_id"] = pd.to_numeric(merged["artist_id"], errors="coerce").astype("Int64")
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == "__main__":
    normalize_artist_gender()

