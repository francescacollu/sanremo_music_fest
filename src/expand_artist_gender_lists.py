import pandas as pd
from pathlib import Path


def expand_and_dedup_artist_gender(path: Path) -> None:
    df = pd.read_csv(path)

    # Normalize artist_name
    df["artist_name"] = df["artist_name"].astype(str).str.strip()

    # Identify list-style rows
    is_list = df["artist_name"].str.startswith("[") & df["artist_name"].str.endswith("]")
    single_rows = df.loc[~is_list].copy()
    list_rows = df.loc[is_list].copy()

    # Expand list rows into one row per member
    expanded_rows = []
    for _, row in list_rows.iterrows():
        s = row["artist_name"]
        inner = s[1:-1]
        members = [p.strip() for p in inner.split(",") if p.strip()]
        for m in members:
            expanded_rows.append(
                {
                    "artist_name": m,
                    "gender": row.get("gender"),
                    "notes": row.get("notes"),
                }
            )

    expanded_df = pd.DataFrame(expanded_rows, columns=["artist_name", "gender", "notes"])
    combined = pd.concat([single_rows, expanded_df], ignore_index=True)

    # Deduplicate per artist_name, preferring non-empty gender values
    combined["artist_name"] = combined["artist_name"].astype(str).str.strip()
    # Treat empty strings as missing
    combined["gender"] = combined["gender"].where(combined["gender"].notna() & (combined["gender"].astype(str) != ""), None)

    groups = []
    conflicts = []

    for artist, grp in combined.groupby("artist_name", dropna=False):
        non_null = grp[grp["gender"].notna()]
        if not non_null.empty:
            unique_genders = sorted(non_null["gender"].unique())
            if len(unique_genders) == 1:
                # All agree: keep the first non-null gender row
                groups.append(non_null.iloc[[0]])
            else:
                # Conflicting genders – record conflict but still keep one
                conflicts.append(grp)
                groups.append(non_null.iloc[[0]])
        else:
            # All genders empty: keep a single representative row
            groups.append(grp.iloc[[0]])

    deduped = pd.concat(groups, ignore_index=True)

    # Optional: write conflicts for manual inspection
    if conflicts:
        conflicts_df = pd.concat(conflicts, ignore_index=True)
        conflicts_df.to_csv(path.parent / "artist_gender_conflicts.csv", index=False)

    # Overwrite main file
    deduped = deduped[["artist_name", "gender", "notes"]]
    deduped.to_csv(path, index=False)


def main() -> None:
    base = Path("data/processed")
    path = base / "artist_gender.csv"
    expand_and_dedup_artist_gender(path)


if __name__ == "__main__":
    main()

