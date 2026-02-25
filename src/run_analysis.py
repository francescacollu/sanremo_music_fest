"""
Run full Sanremo cover night analysis pipeline.
Sections 1-5: load/clean, counts and bar charts, bipartite/co-occurrence networks,
temporal/guests/winners/medley, gender, and self-covers.
Writes figures and summaries to outputs/.
"""
from pathlib import Path
from collections import defaultdict

import pandas as pd
import networkx as nx

from load import load_covers, load_winners, load_medley
from clean import (
    clean_covers,
    clean_winners,
    clean_medley,
    build_expanded_list,
    join_winners_to_covers,
)
from normalize_artist_gender import normalize_artist_gender
from add_artist_ids import add_artist_ids
import viz


def _split_artists_cell(cell: str) -> list[str]:
    """Split a formatted artist cell into individual names."""
    if pd.isna(cell):
        return []
    s = str(cell).strip()
    if not s or s == "-":
        return []
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1]
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return parts
    return [s]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _raw_data_dir() -> Path:
    return _project_root() / "data" / "raw"


def _processed_data_dir() -> Path:
    return _project_root() / "data" / "processed"


def run(
    data_dir: Path | None = None,
    out_dir: Path | None = None,
    processed_dir: Path | None = None,
):
    if data_dir is None:
        data_dir = _raw_data_dir()
    if out_dir is None:
        out_dir = _project_root() / "outputs"
    if processed_dir is None:
        processed_dir = _processed_data_dir()

    out_dir = Path(out_dir)
    processed_dir = Path(processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Section 1: load and clean
    covers_raw = load_covers(data_dir)
    winners_raw = load_winners(data_dir)
    medley_raw = load_medley(data_dir)

    covers = clean_covers(covers_raw)
    winners = clean_winners(winners_raw)
    medley = clean_medley(medley_raw)

    expanded = build_expanded_list(covers, medley)
    covers_with_winner = join_winners_to_covers(covers, winners)

    # Persist cleaned tables and expanded list for inspection
    covers.to_csv(processed_dir / "covers_clean.csv", index=False)
    winners.to_csv(processed_dir / "winners_clean.csv", index=False)
    medley.to_csv(processed_dir / "medley_clean.csv", index=False)
    expanded.to_csv(processed_dir / "expanded_covers.csv", index=False)

    # Update artist_names from normalized data: append new canonical names or full regenerate
    unique_names = set()
    for series in [
        expanded["original_artist"],
        expanded["contestant"],
        covers["guest_artist"],
        medley["contestant"],
        medley["original_artist"],
    ]:
        for cell in series.dropna():
            for name in _split_artists_cell(cell):
                if name and str(name).strip() != "-":
                    unique_names.add(str(name).strip())
    unique_canonical = sorted(unique_names, key=str.casefold)
    artist_names_path = processed_dir / "artist_names.csv"
    if artist_names_path.exists():
        existing = pd.read_csv(artist_names_path)
        if len(existing) > 0 and "artist_id" in existing.columns and "artist_name" in existing.columns:
            existing_names = set(existing["artist_name"].astype(str).str.strip())
            new_names = [n for n in unique_canonical if n not in existing_names]
            if new_names:
                max_id = int(existing["artist_id"].max())
                append_df = pd.DataFrame(
                    {"artist_id": range(max_id + 1, max_id + 1 + len(new_names)), "artist_name": new_names}
                )
                combined = pd.concat([existing, append_df], ignore_index=True)
                combined.to_csv(artist_names_path, index=False)
        else:
            pd.DataFrame({"artist_name": unique_canonical}).to_csv(artist_names_path, index=False)
            add_artist_ids(artist_names_path)
    else:
        pd.DataFrame({"artist_name": unique_canonical}).to_csv(artist_names_path, index=False)
        add_artist_ids(artist_names_path)

    # Section 2: most sung songs and authors, bar charts
    song_counts = expanded["song"].value_counts().reset_index()
    song_counts.columns = ["song", "count"]
    author_counts = expanded["original_artist"].value_counts().reset_index()
    author_counts.columns = ["original_artist", "count"]

    fig_songs = viz.bar_top_songs(song_counts)
    viz.write_figure(fig_songs, out_dir / "top_songs", html=True)
    fig_authors = viz.bar_top_authors(author_counts)
    viz.write_figure(fig_authors, out_dir / "top_authors", html=True)

    # Section 3: bipartite and co-occurrence networks
    B_song = nx.Graph()
    for _, row in expanded.iterrows():
        B_song.add_edge(f"contestant:{row['contestant']}", f"song:{row['song']}", year=row["year"])
    degree_song = nx.degree_centrality(B_song)
    song_centrality = [(n.replace("song:", ""), d) for n, d in degree_song.items() if n.startswith("song:")]
    song_centrality.sort(key=lambda x: -x[1])
    top_songs_centrality = song_centrality[:10]

    B_artist = nx.Graph()
    for _, row in expanded.iterrows():
        B_artist.add_edge(f"contestant:{row['contestant']}", f"artist:{row['original_artist']}", year=row["year"])
    degree_artist = nx.degree_centrality(B_artist)
    artist_centrality = [(n.replace("artist:", ""), d) for n, d in degree_artist.items() if n.startswith("artist:")]
    artist_centrality.sort(key=lambda x: -x[1])
    top_artists_centrality = artist_centrality[:10]

    # Persist full centrality tables for further analysis
    song_centrality_df = pd.DataFrame(song_centrality, columns=["song", "degree_centrality"])
    artist_centrality_df = pd.DataFrame(artist_centrality, columns=["original_artist", "degree_centrality"])
    song_centrality_df.to_csv(out_dir / "song_degree_centrality.csv", index=False)
    artist_centrality_df.to_csv(out_dir / "artist_degree_centrality.csv", index=False)

    G_guest = nx.Graph()
    for _, row in covers.iterrows():
        g = row["guest_artist"]
        if pd.notna(g) and str(g).strip() != "-":
            G_guest.add_edge(row["contestant"], g, year=row["year"])
    degree_guest = nx.degree_centrality(G_guest)
    guest_centrality = sorted(degree_guest.items(), key=lambda x: -x[1])[:15]

    edition_artists = expanded.groupby("year")["original_artist"].apply(list).to_dict()
    cooccur = defaultdict(int)
    for year, artists in edition_artists.items():
        unique = list(set(artists))
        for i, a in enumerate(unique):
            for b in unique[i + 1 :]:
                if a != b:
                    pair = tuple(sorted([a, b]))
                    cooccur[pair] += 1
    G_cooccur = nx.Graph()
    for (a, b), w in cooccur.items():
        G_cooccur.add_edge(a, b, weight=w)
    top_cooccur = sorted(G_cooccur.edges(data=True), key=lambda x: -x[2].get("weight", 0))[:15]

    # Co-occurrence network figure for original artists
    fig_cooccur = viz.network_cooccurrence(G_cooccur)
    viz.write_figure(fig_cooccur, out_dir / "cooccurrence_network", html=True)

    # Contestant → original-artist relationship table and Sankey diagram
    contestant_artist = (
        expanded.groupby(["contestant", "original_artist"])
        .size()
        .reset_index(name="n_covers_pair")
    )
    contestant_artist.to_csv(out_dir / "contestant_to_original_counts.csv", index=False)

    fig_sankey = viz.sankey_contestant_to_original(contestant_artist)
    viz.write_figure(fig_sankey, out_dir / "contestant_to_original_sankey", html=True)

    # Section 4: temporal, guests, winners vs canon, medley summary
    expanded_decade = expanded.copy()
    expanded_decade["decade"] = (expanded_decade["song_year_parsed"] // 10 * 10).astype("Int64")

    # Overall distribution of original-song decades
    decade_counts = expanded_decade["decade"].dropna().astype(int).value_counts().sort_index()
    decade_df = decade_counts.reset_index()
    decade_df.columns = ["decade", "count"]
    decade_df["decade"] = decade_df["decade"].astype(str) + "s"
    fig_decade = viz.bar_decade(decade_df)
    viz.write_figure(fig_decade, out_dir / "decade_distribution", html=True)

    # Decade mix per edition (for stacked area over time)
    decade_per_year = (
        expanded_decade.dropna(subset=["decade"])
        .groupby(["year", "decade"])
        .size()
        .reset_index(name="n_covers")
    )
    decade_per_year["total_per_year"] = decade_per_year.groupby("year")["n_covers"].transform("sum")
    decade_per_year["share"] = decade_per_year["n_covers"] / decade_per_year["total_per_year"]
    decade_per_year["decade"] = decade_per_year["decade"].astype(int).astype(str) + "s"
    decade_per_year = decade_per_year.sort_values(["year", "decade"])
    decade_per_year.to_csv(out_dir / "covers_decade_mix_per_year.csv", index=False)

    fig_decade_time = viz.area_decade_over_time(decade_per_year)
    viz.write_figure(fig_decade_time, out_dir / "decade_mix_over_time", html=True)

    covers_per_year = covers.groupby("year").size().reset_index(name="n_covers")
    fig_trend = viz.bar_covers_per_year(covers_per_year)
    viz.write_figure(fig_trend, out_dir / "covers_per_edition", html=True)

    # Year-lag distribution: how far back contestants look each edition
    expanded_with_lag = expanded_decade.dropna(subset=["song_year_parsed"]).copy()
    expanded_with_lag["song_year_parsed"] = expanded_with_lag["song_year_parsed"].astype(int)
    expanded_with_lag["year_lag"] = expanded_with_lag["year"] - expanded_with_lag["song_year_parsed"]
    expanded_with_lag.to_csv(out_dir / "covers_with_year_lag.csv", index=False)

    fig_year_lag = viz.box_year_lag(expanded_with_lag)
    viz.write_figure(fig_year_lag, out_dir / "year_lag_by_edition", html=True)

    # Section 5: gender mapping and flows (requires manual annotation file)
    artist_gender_path = processed_dir / "artist_gender.csv"
    if not artist_gender_path.exists():
        all_artists = pd.concat(
            [expanded["original_artist"], expanded["contestant"], covers["guest_artist"]],
            ignore_index=True,
        )
        artist_names = (
            all_artists.dropna()
            .astype(str)
            .str.strip()
        )
        artist_names = artist_names[artist_names.ne("-")].drop_duplicates().sort_values()
        artist_gender_template = pd.DataFrame(
            {
                "artist_name": artist_names,
                "gender": pd.NA,  # e.g. 'F', 'M', 'Group', 'Mixed', 'Unknown'
                "notes": pd.NA,
            }
        )
        artist_gender_template.to_csv(artist_gender_path, index=False)
        expanded_gender = None
        gender_by_year = None
        gender_flow = None
    else:
        # Normalize gender annotations to a per-artist table keyed by artist_id.
        artist_gender_by_artist = normalize_artist_gender(
            artist_names_path=processed_dir / "artist_names.csv",
            artist_gender_path=artist_gender_path,
            output_path=processed_dir / "artist_gender_by_artist.csv",
            processed_dir=processed_dir,
        )
        artist_gender_by_artist["artist_name"] = (
            artist_gender_by_artist["artist_name"].astype(str).str.strip()
        )

        # Use artist_id for joins: build name -> artist_id from artist_names
        artist_names_df = pd.read_csv(processed_dir / "artist_names.csv")
        artist_names_df["artist_name"] = artist_names_df["artist_name"].astype(str).str.strip()
        name_to_id = artist_names_df.set_index("artist_name")["artist_id"].to_dict()
        expanded_gender = expanded.copy()
        expanded_gender["original_artist_id"] = expanded_gender["original_artist"].map(name_to_id)
        expanded_gender["contestant_id"] = expanded_gender["contestant"].map(name_to_id)
        expanded_gender = expanded_gender.merge(
            artist_gender_by_artist[["artist_id", "gender"]].rename(
                columns={"artist_id": "original_artist_id", "gender": "original_gender"}
            ),
            on="original_artist_id",
            how="left",
        )
        expanded_gender = expanded_gender.merge(
            artist_gender_by_artist[["artist_id", "gender"]].rename(
                columns={"artist_id": "contestant_id", "gender": "performer_gender"}
            ),
            on="contestant_id",
            how="left",
        )

        # Share of covers with female original artists per edition
        mask_original_known = expanded_gender["original_gender"].notna()
        gender_by_year = (
            expanded_gender[mask_original_known]
            .groupby("year")
            .agg(
                n_covers=("original_gender", "size"),
                n_female_original=("original_gender", lambda s: (s == "F").sum()),
            )
            .reset_index()
        )
        if not gender_by_year.empty:
            gender_by_year["share_female_original"] = (
                gender_by_year["n_female_original"] / gender_by_year["n_covers"]
            )
            gender_by_year.to_csv(out_dir / "gender_original_share_per_year.csv", index=False)

            fig_gender_line = viz.line_share_female_originals(gender_by_year)
            viz.write_figure(fig_gender_line, out_dir / "gender_original_share_over_time", html=True)

        # Flow between performer gender and original-artist gender
        mask_both_known = expanded_gender["performer_gender"].notna() & expanded_gender["original_gender"].notna()
        gender_flow = (
            expanded_gender[mask_both_known]
            .groupby(["performer_gender", "original_gender"])
            .size()
            .reset_index(name="n_covers")
        )
        if not gender_flow.empty:
            gender_flow.to_csv(out_dir / "gender_performer_to_original_counts.csv", index=False)

            fig_gender_sankey = viz.sankey_gender_flow(gender_flow)
            viz.write_figure(fig_gender_sankey, out_dir / "gender_performer_to_original_sankey", html=True)

    # Section 6: self-covers and self-tributes
    expanded_self = expanded.copy()
    expanded_self["is_self_cover"] = expanded_self.apply(
        lambda row: bool(
            set(_split_artists_cell(row["contestant"]))
            & set(_split_artists_cell(row["original_artist"]))
        ),
        axis=1,
    )
    self_covers = expanded_self[expanded_self["is_self_cover"]]
    self_covers.to_csv(out_dir / "self_covers_detailed.csv", index=False)

    self_covers_per_year = self_covers.groupby("year").size().reset_index(name="n_self_covers")
    self_covers_per_year.to_csv(out_dir / "self_covers_per_year.csv", index=False)

    if not self_covers_per_year.empty:
        fig_self_covers = viz.bar_self_covers_per_year(self_covers_per_year)
        viz.write_figure(fig_self_covers, out_dir / "self_covers_per_year", html=True)

    winning_songs = winners["song"].tolist()
    top_songs_set = set(song_counts.head(30)["song"])
    overlap_songs = [s for s in winning_songs if s in top_songs_set]
    winner_rows = covers_with_winner[covers_with_winner["is_winner"]]
    winning_artists = winner_rows["original_artist"].tolist()
    top_artists_set = set(author_counts.head(30)["original_artist"])
    overlap_artists = [a for a in winning_artists if a in top_artists_set]

    # Overlap between canon artists and medley selections
    medley_songs_set = set(medley["song"].unique())
    medley_artists_set = set(medley["original_artist"].unique())
    overlap_songs_medley = [s for s in top_songs_set if s in medley_songs_set]
    overlap_artists_medley = [a for a in top_artists_set if a in medley_artists_set]

    # Ego-network style summaries for a few iconic originals
    iconic_artists = ["Lucio Battisti", "Lucio Dalla", "Fabrizio De André"]
    ego_summaries: dict[str, list[tuple[str, int]]] = {}
    for artist in iconic_artists:
        if artist in G_cooccur:
            neigh = []
            for neighbor, data in G_cooccur[artist].items():
                w = data.get("weight", 0)
                neigh.append((neighbor, w))
            neigh.sort(key=lambda x: -x[1])
            ego_summaries[artist] = neigh

    medley_perf = (
        medley.groupby(["year", "contestant"])
        .agg(n_songs=("song", "count"), songs=("song", list), artists=("original_artist", list))
        .reset_index()
    )

    # Write text summary
    summary_lines = [
        "Sanremo cover night – analysis summary",
        "=" * 50,
        "",
        "Top 10 songs by degree centrality (bipartite contestant-song):",
        *[f"  {s} ({c:.4f})" for s, c in top_songs_centrality],
        "",
        "Top 10 original artists by degree centrality:",
        *[f"  {a} ({c:.4f})" for a, c in top_artists_centrality],
        "",
        "Top 15 guest/contestant by degree:",
        *[f"  {n} ({c:.4f})" for n, c in guest_centrality],
        "",
        "Top 15 co-occurrence pairs (same edition):",
        *[f"  {a} / {b} (weight={d.get('weight', 0)})" for a, b, d in top_cooccur],
        "",
        "Winners vs canon:",
        f"  Winning songs: {winning_songs}",
        f"  Overlap with top 30 most covered songs: {overlap_songs}",
        f"  Winning original artists: {winning_artists}",
        f"  Overlap with top 30 most covered artists: {overlap_artists}",
        "",
        "Medleys vs canon (top 30):",
        f"  Canon songs appearing in medleys: {overlap_songs_medley}",
        f"  Canon artists appearing in medleys: {overlap_artists_medley}",
        "",
        "Ego networks for selected iconic artists (co-occurring originals):",
    ]
    for artist, neighbors in ego_summaries.items():
        summary_lines.append(f"  {artist}:")
        for neighbor, weight in neighbors[:10]:
            summary_lines.append(f"    - {neighbor} (weight={weight})")
    if not ego_summaries:
        summary_lines.append("  (no ego networks available)")
    summary_lines.extend(
        [
            "",
            "Self-covers and self-tributes:",
            f"  Total self-covers detected: {len(self_covers)}",
            f"  Years with at least one self-cover: {sorted(self_covers_per_year['year'].tolist()) if not self_covers_per_year.empty else []}",
        ]
    )
    summary_lines.extend(
        [
            "",
            f"Medley performances: {len(medley_perf)}",
            "Songs per medley (year, contestant, n_songs):",
            medley_perf[["year", "contestant", "n_songs"]].to_string(index=False),
        ]
    )
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def main():
    run()


if __name__ == "__main__":
    main()
