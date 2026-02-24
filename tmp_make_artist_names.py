import csv
from pathlib import Path

base_path = Path("data/processed")
input_path = base_path / "artist_gender.csv"
output_path = base_path / "artist_names.py.csv"  # temp name to avoid clashes

names = set()

with input_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row.get("artist_name", "")
        if not name:
            continue
        name = name.strip()
        if name.startswith("[") and name.endswith("]"):
            inner = name[1:-1]
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            names.update(parts)
        else:
            names.add(name)

sorted_names = sorted(names)

with output_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["artist_name"])
    for n in sorted_names:
        writer.writerow([n])

print(f"Wrote {len(sorted_names)} unique artist names to {output_path}")
