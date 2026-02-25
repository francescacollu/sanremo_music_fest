"""Clean data: strip footnotes, normalize guest, parse song_year, PK checks, build expanded list."""
import re
from pathlib import Path

import pandas as pd


def _artist_aliases_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "processed" / "artist_aliases.csv"


def _load_artist_aliases() -> dict[str, str] | None:
    """Load variant -> canonical_name map from artist_aliases.csv. Return None if file missing or empty."""
    path = _artist_aliases_path()
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


def _split_artist_cell(cell: str) -> list[str]:
    """Split a cell into individual names (single name or [A, B, C] -> list)."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s or s == "-":
        return []
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        return [p.strip() for p in inner.split(",") if p.strip()]
    return [s]


def resolve_artist_aliases(cell, alias_map: dict[str, str] | None) -> str:
    """Replace artist name(s) with canonical form. Whole-cell match first, then name-by-name."""
    if alias_map is None:
        return cell
    if pd.isna(cell):
        return cell
    s = str(cell).strip()
    if not s or s == "-":
        return s
    if s in alias_map:
        return alias_map[s]
    names = _split_artist_cell(s)
    if not names:
        return s
    resolved = [alias_map.get(n, n) for n in names]
    if len(resolved) == 1:
        return resolved[0]
    return "[" + ", ".join(resolved) + "]"


def strip_footnotes(s):
    if pd.isna(s) or not isinstance(s, str):
        return s
    return re.sub(r"\s*\[\d[\d,\s]*\]\s*$", "", s).strip()


def normalize_guest(series: pd.Series) -> pd.Series:
    out = series.apply(strip_footnotes)
    out = out.replace({"": None}).fillna("-")
    out = out.astype(str).str.strip().replace("", "-")
    return out


def parse_song_year(val):
    if pd.isna(val) or val == "-" or (isinstance(val, str) and val.strip() == ""):
        return pd.NA
    s = str(val).strip()
    if "-" in s and re.match(r"^\d{4}-\d{4}", s):
        return int(s.split("-")[0])
    if re.match(r"^\d{4}$", s):
        return int(s)
    return pd.NA


def check_pk_covers(covers: pd.DataFrame) -> None:
    pk = ["year", "contestant", "song"]
    dup = covers.duplicated(subset=pk)
    if dup.any():
        raise ValueError(f"Duplicate rows in covers on {pk}: {dup.sum()} rows")


def check_pk_winners(winners: pd.DataFrame) -> None:
    if not winners["year"].is_unique:
        n = winners.duplicated(subset=["year"]).sum()
        raise ValueError(f"Duplicate years in winners: {n} rows")


def check_pk_medley(medley: pd.DataFrame) -> None:
    pk = ["year", "contestant", "song"]
    dup = medley.duplicated(subset=pk)
    if dup.any():
        raise ValueError(f"Duplicate rows in medley on {pk}: {dup.sum()} rows")


def format_artist_list_cell(val):
    if pd.isna(val):
        return val
    s = str(val).strip()
    if s == "" or s == "-":
        return s
    if not re.search(r"/|\se(?:d)?\s", s, flags=re.IGNORECASE):
        return s
    names = []
    for chunk in s.split("/"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.split(r"\s+e(?:d)?\s+", chunk, flags=re.IGNORECASE)
        for name in parts:
            name = name.strip(" ,")
            if name:
                names.append(name)
    if len(names) <= 1:
        return s
    return "[" + ", ".join(names) + "]"


def clean_covers(covers: pd.DataFrame) -> pd.DataFrame:
    covers = covers.copy()
    covers["guest_artist"] = normalize_guest(covers["guest_artist"])
    covers["contestant"] = covers["contestant"].apply(format_artist_list_cell)
    covers["original_artist"] = covers["original_artist"].apply(format_artist_list_cell)
    covers["guest_artist"] = covers["guest_artist"].apply(format_artist_list_cell)
    alias_map = _load_artist_aliases()
    if alias_map:
        for col in ("contestant", "original_artist", "guest_artist"):
            covers[col] = covers[col].apply(lambda x: resolve_artist_aliases(x, alias_map))
    covers["song_year_parsed"] = covers["song_year"].apply(parse_song_year)
    check_pk_covers(covers)
    return covers


def clean_winners(winners: pd.DataFrame) -> pd.DataFrame:
    winners_clean = winners.copy()
    winners_clean["winner"] = winners_clean["winner"].apply(format_artist_list_cell)
    if "guest_artists" in winners_clean.columns:
        winners_clean["guest_artists"] = winners_clean["guest_artists"].apply(format_artist_list_cell)
    alias_map = _load_artist_aliases()
    if alias_map:
        winners_clean["winner"] = winners_clean["winner"].apply(lambda x: resolve_artist_aliases(x, alias_map))
        if "guest_artists" in winners_clean.columns:
            winners_clean["guest_artists"] = winners_clean["guest_artists"].apply(lambda x: resolve_artist_aliases(x, alias_map))
    check_pk_winners(winners_clean)
    return winners_clean


def clean_medley(medley: pd.DataFrame) -> pd.DataFrame:
    medley_clean = medley.copy()
    medley_clean["contestant"] = medley_clean["contestant"].apply(format_artist_list_cell)
    medley_clean["original_artist"] = medley_clean["original_artist"].apply(format_artist_list_cell)
    alias_map = _load_artist_aliases()
    if alias_map:
        medley_clean["contestant"] = medley_clean["contestant"].apply(lambda x: resolve_artist_aliases(x, alias_map))
        medley_clean["original_artist"] = medley_clean["original_artist"].apply(lambda x: resolve_artist_aliases(x, alias_map))
    check_pk_medley(medley_clean)
    return medley_clean


def build_expanded_list(covers: pd.DataFrame, medley: pd.DataFrame) -> pd.DataFrame:
    is_medley = (
        covers["song"].str.contains("Medley", case=False, na=False)
        | (covers["original_artist"] == "Artisti vari")
    )
    non_medley = covers[~is_medley][
        ["year", "edition", "contestant", "song", "original_artist", "song_year", "song_year_parsed", "guest_artist"]
    ].copy()

    medley_rows = medley[["year", "edition", "contestant", "song", "original_artist"]].copy()
    medley_rows["song_year"] = None
    medley_rows["song_year_parsed"] = pd.NA
    medley_rows["guest_artist"] = "-"

    expanded = pd.concat([non_medley, medley_rows], ignore_index=True)
    return expanded


def join_winners_to_covers(covers: pd.DataFrame, winners: pd.DataFrame) -> pd.DataFrame:
    merged = covers.merge(
        winners[["year", "winner", "song"]].rename(columns={"song": "winning_song", "winner": "winner_name"}),
        on="year",
        how="left",
    )
    merged["is_winner"] = merged["contestant"] == merged["winner_name"]
    return merged.drop(columns=["winning_song", "winner_name"])
