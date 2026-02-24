import csv
from pathlib import Path


def add_artist_ids(
    input_path: Path = Path("data/processed/artist_names.csv"),
) -> None:
    """
    Add a stable integer primary key column `artist_id` to artist_names.csv.

    IDs are assigned deterministically by sorting artist names
    case-insensitively and then numbering from 1..N.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "artist_name" not in reader.fieldnames:
            raise ValueError("Expected a column named 'artist_name'")
        names = [row["artist_name"] for row in reader if row.get("artist_name")]

    # Ensure deterministic ordering for stable IDs
    # We keep unique names only.
    unique_names = sorted(set(names), key=lambda x: x.casefold())

    output_rows = [
        {"artist_id": idx, "artist_name": name}
        for idx, name in enumerate(unique_names, start=1)
    ]

    with input_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["artist_id", "artist_name"])
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    base_path = Path(__file__).resolve().parents[1]
    add_artist_ids(base_path / "data" / "processed" / "artist_names.csv")

