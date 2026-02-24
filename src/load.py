"""Load and rename columns for covers, winners, and medley CSVs."""
import pandas as pd
import csv
from pathlib import Path

COVERS_RENAME = {
    "anno": "year",
    "edizione": "edition",
    "cantante in gara": "contestant",
    "canzone": "song",
    "artista_originale": "original_artist",
    "anno_canzone": "song_year",
    "artista_duetto": "guest_artist",
}

WINNERS_RENAME = {
    "anno": "year",
    "edizione": "edition",
    "vincitore": "winner",
    "canzone": "song",
    "ospite/i": "guest_artists",
}

MEDLEY_RENAME = {
    "anno": "year",
    "edizione": "edition",
    "cantante in gara": "contestant",
    "brano": "song",
    "autore (artista originale)": "original_artist",
}


def load_covers(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "storico_serata_cover.csv"
    rows: list[list[str]] = []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        n_cols = len(header)
        for row in reader:
            if not row:
                continue
            if len(row) > n_cols:
                base = row[: n_cols - 1]
                last = ",".join(row[n_cols - 1 :])
                row = base + [last]
            rows.append(row)

    df = pd.DataFrame(rows, columns=[c.strip() for c in header])
    df = df.rename(columns=COVERS_RENAME)
    df["year"] = df["year"].astype(int)
    return df


def load_winners(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "vincitori_serata_cover.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=WINNERS_RENAME)
    df["year"] = df["year"].astype(int)
    return df


def load_medley(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "medley.csv"
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=MEDLEY_RENAME)
    df["year"] = df["year"].astype(int)
    return df
