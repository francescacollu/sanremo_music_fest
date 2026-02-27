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


def _aggregate_gender_list(genders: list) -> object:
    """Aggregate a list of per-artist genders to a row-level category (F, M, Mixed, or pd.NA)."""
    vals = []
    for g in genders:
        if pd.isna(g):
            continue
        s = str(g).strip()
        if not s:
            continue
        vals.append(s)
    if not vals:
        return pd.NA
    unique_vals = set(vals)
    if len(unique_vals) == 1:
        return vals[0]
    return "Mixed"


def _row_gender_from_cell(cell, mapping: dict) -> object:
    """Get a single gender category for a cell (contestant or original_artist) using name_to_gender."""
    names = _split_artists_cell(cell)
    if not names:
        return pd.NA
    genders = [mapping.get(n) for n in names if n and str(n).strip() != "-"]
    return _aggregate_gender_list(genders)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _raw_data_dir() -> Path:
    return _project_root() / "data" / "raw"


def _processed_data_dir() -> Path:
    return _project_root() / "data" / "processed"


def _angle_dir(out_dir: Path, angle: str) -> Path:
    """Return a subdirectory of out_dir for a given analysis angle (e.g. 'gender')."""
    angle_dir = Path(out_dir) / angle
    angle_dir.mkdir(parents=True, exist_ok=True)
    return angle_dir


def _write_canon_creation_analysis_guide(canon_out: Path) -> None:
    """Write a short guide describing the canon_creation outputs."""
    lines = [
        "# Canon creation — Analysis guide",
        "",
        "This folder holds outputs for the **canon creation** angle: what gets covered, how diverse or original performers' choices are, and how that changes over time.",
        "",
        "---",
        "",
        "## 1. Data and outputs in this folder",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `contestant_to_original_counts.csv` | Count of covers per (contestant, original_artist) pair. |",
        "| `contestant_to_original_sankey.html` | Sankey diagram: who covers whom (contestants to original artists). |",
        "| `covers_per_edition.html` | Bar chart: number of cover performances per edition. |",
        "| `covers_with_year_lag.csv` | Cover-level data with year lag (festival year minus original song year). |",
        "| `year_lag_by_edition.html` | Boxplot of year lag by edition (how far back contestants look). |",
        "| `original_artists_per_performer.csv` | Per performer (all editions): number of distinct original artists covered, total cover count. |",
        "| `original_artists_per_performer.html` | Bar chart: top performers by number of distinct originals. |",
        "| `original_artists_per_performer_per_year.csv` | Same metrics by (year, contestant). |",
        "| `performer_originality.csv` | Per performer: mean \"uniqueness\" of their choices (1 / global cover count of that original). Higher = rarer originals. |",
        "| `canon_metrics_over_time.csv` | Per edition: mean distinct originals per performer, mean originality. |",
        "| `canon_metrics_over_time.html` | Line chart of these two metrics over editions. |",
        "| `canon_original_gender_share_per_year.csv` | Share of covers by original-artist gender per year (if gender data exists). |",
        "| `canon_original_gender_share_over_time.html` | Stacked area of that share over editions. |",
        "| `top_artists.csv` / `top_artists.html` | Most covered original artists (canon). |",
        "| `top_songs.csv` / `top_songs.html` | Most covered songs (canon). |",
        "| `decade_distribution.csv` / `decade_distribution.html` | Distribution of original-song decades. |",
        "| `decade_mix_per_year.csv` / `decade_mix_over_time.html` | Share of covers by decade per edition. |",
        "",
        "---",
        "",
        "## 2. Questions this angle answers",
        "",
        "- **How many original artists per performer?** Use `original_artists_per_performer.csv` or the per-year table.",
        "- **How \"original\" are performers' choices?** Use `performer_originality.csv` (mean uniqueness).",
        "- **Changes over time?** Use `canon_metrics_over_time.csv` and the corresponding chart.",
        "- **Female vs male original artists?** Use `canon_original_gender_share_per_year.csv` and the area chart (when gender annotations exist).",
        "- **Top artists and top songs?** Use `top_artists.csv` / `top_songs.csv` and the bar charts.",
        "- **Most covered decades?** Use `decade_distribution.csv` and `decade_mix_per_year.csv`.",
        "- **Who covers whom?** Use `contestant_to_original_counts.csv` and `contestant_to_original_sankey.html`.",
        "- **Volume and time depth?** Use `covers_per_edition.html`, `covers_with_year_lag.csv`, and `year_lag_by_edition.html`.",
        "",
    ]
    (canon_out / "analysis_guide.md").write_text("\n".join(lines), encoding="utf-8")


def _write_taste_communities_analysis_guide(
    G_performer_shared_original: nx.Graph,
    G_cooccur: nx.Graph,
    expanded: pd.DataFrame,
    contestant_to_gender: dict | None,
    performer_edges: list,
    cooccur_edges: list,
    out_path: Path,
    _split_artists_cell,
):
    """Compute network metrics and write analysis_guide.md with results."""
    from collections import Counter

    lines = [
        "# Taste communities (network analysis) — Analysis guide",
        "",
        "This document lists what to check in the networks and reports **pre-computed** metrics to support your data story.",
        "",
        "---",
        "",
        "## 1. Data and outputs in this folder",
        "",
        "- **Performer network**: nodes = contestants (performers); edge between two if they covered at least one same original artist; edge weight = number of shared originals. Files: `performer_shared_original_network.html`, `performer_shared_original_edges.csv`, `performer_shared_original_top_pairs.csv`.",
        "- **Cooccurrence network**: nodes = original artists; edge between two if they co-occurred in at least one edition; edge weight = co-occurrence count. Files: `cooccurrence_network.html`, `cooccurrence_edges.csv`, `cooccurrence_top_pairs.csv`.",
        "- Optional: node color = gender when `artist_gender_by_artist.csv` exists.",
        "",
        "---",
        "",
        "## 2. What to check in the networks (visual)",
        "",
        "- **Performer network**: Do performers cluster by taste? Are there clear communities? When gender is on: do clusters look more male/female/mixed? Who are the \"bridge\" performers (high betweenness)?",
        "- **Cooccurrence network**: Which original artists sit in the centre? Do artists from the same era/genre cluster?",
        "",
        "---",
        "",
        "## 3. Computed metrics — Performer network",
        "",
    ]

    # Weighted degree (top 20)
    degrees = dict(G_performer_shared_original.degree(weight="weight"))
    top_degree = sorted(degrees.items(), key=lambda x: -x[1])[:20]
    lines.append("### Top 20 performers by weighted degree (taste overlap)")
    lines.append("")
    for i, (name, d) in enumerate(top_degree, 1):
        lines.append(f"{i}. **{name}** — {d}")
    lines.extend(["", "---", "", "### Top 15 bridge performers (betweenness centrality)", ""])
    try:
        betweenness = nx.betweenness_centrality(G_performer_shared_original, weight="weight")
        top_bet = sorted(betweenness.items(), key=lambda x: -x[1])[:15]
        for i, (name, b) in enumerate(top_bet, 1):
            lines.append(f"{i}. **{name}** — {b:.4f}")
    except Exception:
        lines.append("(Betweenness could not be computed.)")
    lines.extend(["", "---", "", "### Top 15 performer pairs by shared originals", ""])
    sorted_pairs = sorted(performer_edges, key=lambda x: -x["weight"])[:15]
    for i, e in enumerate(sorted_pairs, 1):
        lines.append(f"{i}. **{e['node_a']}** and **{e['node_b']}** — {e['weight']} shared original artist(s)")
    lines.extend(["", "---", "", "## 4. Community detection (performer network)", ""])

    # Community detection
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G_performer_shared_original, weight="weight"))
    except Exception:
        try:
            from networkx.algorithms.community import label_propagation_communities
            communities = list(label_propagation_communities(G_performer_shared_original))
        except Exception:
            communities = []
    # Sort by size descending
    communities = sorted(communities, key=len, reverse=True)

    if not communities:
        lines.append("(Community detection could not be run.)")
    else:
        lines.append(f"Found **{len(communities)}** communities (greedy modularity).")
        lines.append("")
        # Per-community: size, gender mix, top originals
        for idx, comm in enumerate(communities, 1):
            members = list(comm)
            lines.append(f"### Community {idx} (size: {len(members)})")
            lines.append("")
            lines.append("**Members:** " + ", ".join(sorted(members)[:25]) + ("..." if len(members) > 25 else "") + ".")
            lines.append("")
            if contestant_to_gender:
                gender_counts = Counter()
                for c in members:
                    g = contestant_to_gender.get(c)
                    if g is None or (isinstance(g, float) and str(g) == "nan") or pd.isna(g):
                        gender_counts["Unknown"] += 1
                    else:
                        gender_counts[str(g).strip() or "Unknown"] += 1
                parts = [f"{g}: {n}" for g, n in sorted(gender_counts.items())]
                lines.append("**Gender mix:** " + "; ".join(parts) + ".")
                lines.append("")
            # Top originals covered by this community (from expanded)
            orig_counts = Counter()
            for _, row in expanded.iterrows():
                if row["contestant"] in comm:
                    for o in _split_artists_cell(row["original_artist"]):
                        if o and str(o).strip() != "-":
                            orig_counts[o.strip()] += 1
            top_orig = orig_counts.most_common(10)
            if top_orig:
                lines.append("**Top 10 original artists covered by this community:**")
                for o, c in top_orig:
                    lines.append(f"- {o} ({c})")
            lines.append("")
            lines.append("---")
            lines.append("")

    lines.extend(["", "## 5. Computed metrics — Cooccurrence network", "", "### Top 15 original-artist pairs by co-occurrence weight", ""])
    sorted_cooccur = sorted(cooccur_edges, key=lambda x: -x["weight"])[:15]
    for i, e in enumerate(sorted_cooccur, 1):
        lines.append(f"{i}. **{e['node_a']}** and **{e['node_b']}** — {e['weight']}")
    lines.extend(["", "### Top 15 original artists by weighted degree (co-occurrence)", ""])
    cooccur_degrees = dict(G_cooccur.degree(weight="weight"))
    top_cooccur_deg = sorted(cooccur_degrees.items(), key=lambda x: -x[1])[:15]
    for i, (name, d) in enumerate(top_cooccur_deg, 1):
        lines.append(f"{i}. **{name}** — {d}")
    lines.extend([
        "",
        "---",
        "",
        "## 6. Story angles you can build",
        "",
        "- **Taste communities**: Use the communities above; name them by top covered originals or genre; describe size and gender mix.",
        "- **Bridge performers**: Use the betweenness list; say who connects different taste clusters.",
        "- **Gender and taste**: Compare gender mix across communities; note homophily or mixing.",
        "- **Canon and co-occurrence**: Use central artists and top pairs to describe the \"Sanremo cover canon\" and typical pairings.",
        "- **Strongest ties**: Use the top performer pairs to highlight similar repertoires or key editions.",
        "",
    ])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


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

    # Persist full centrality tables for further analysis (written to taste_communities below)
    song_centrality_df = pd.DataFrame(song_centrality, columns=["song", "degree_centrality"])
    artist_centrality_df = pd.DataFrame(artist_centrality, columns=["original_artist", "degree_centrality"])

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

    taste_communities_out = _angle_dir(out_dir, "taste_communities")
    song_centrality_df.to_csv(taste_communities_out / "song_degree_centrality.csv", index=False)
    artist_centrality_df.to_csv(taste_communities_out / "artist_degree_centrality.csv", index=False)

    # Optional: color nodes by gender when artist_gender_by_artist exists
    name_to_gender_tc = None
    contestant_to_gender = None
    artist_gender_by_artist_path = processed_dir / "artist_gender_by_artist.csv"
    if artist_gender_by_artist_path.exists():
        agb = pd.read_csv(artist_gender_by_artist_path)
        agb["artist_name"] = agb["artist_name"].astype(str).str.strip()
        name_to_gender_tc = agb.set_index("artist_name")["gender"].to_dict()
        contestant_to_gender = {
            c: _row_gender_from_cell(c, name_to_gender_tc) for c in expanded["contestant"].unique()
        }

    # Co-occurrence network figure for original artists
    fig_cooccur = viz.network_cooccurrence(
        G_cooccur,
        #node_color_attr=name_to_gender_tc,
    )
    viz.write_figure(fig_cooccur, taste_communities_out / "cooccurrence_network", html=True)

    # Performer shared-original network: nodes = contestants, edge = covered at least one same original artist
    original_to_contestants = defaultdict(set)
    for _, row in expanded.iterrows():
        for orig in _split_artists_cell(row["original_artist"]):
            if orig and str(orig).strip() != "-":
                original_to_contestants[orig.strip()].add(row["contestant"])
    G_performer_shared_original = nx.Graph()
    for _orig, contestants in original_to_contestants.items():
        contestants = list(contestants)
        for i in range(len(contestants)):
            for j in range(i + 1, len(contestants)):
                a, b = contestants[i], contestants[j]
                if G_performer_shared_original.has_edge(a, b):
                    G_performer_shared_original[a][b]["weight"] += 1
                else:
                    G_performer_shared_original.add_edge(a, b, weight=1)

    performer_total_covers = expanded.groupby("contestant").size().to_dict()
    fig_performer = viz.network_graph(
        G_performer_shared_original,
        top_n=80,
        title="Performers connected by same original artist(s)",
        node_size_attr=performer_total_covers,
        node_hover_suffix=" shared original artists (degree)",
        #node_color_attr=contestant_to_gender,
    )
    viz.write_figure(fig_performer, taste_communities_out / "performer_shared_original_network", html=True)

    # Edge list and top pairs CSVs for taste communities
    performer_edges = [
        {"node_a": u, "node_b": v, "weight": d.get("weight", 1)}
        for u, v, d in G_performer_shared_original.edges(data=True)
    ]
    if performer_edges:
        pd.DataFrame(performer_edges).to_csv(
            taste_communities_out / "performer_shared_original_edges.csv", index=False
        )
        pd.DataFrame(performer_edges).sort_values("weight", ascending=False).to_csv(
            taste_communities_out / "performer_shared_original_top_pairs.csv", index=False
        )
    cooccur_edges = [
        {"node_a": u, "node_b": v, "weight": d.get("weight", 1)}
        for u, v, d in G_cooccur.edges(data=True)
    ]
    if cooccur_edges:
        pd.DataFrame(cooccur_edges).to_csv(
            taste_communities_out / "cooccurrence_edges.csv", index=False
        )
        pd.DataFrame(cooccur_edges).sort_values("weight", ascending=False).to_csv(
            taste_communities_out / "cooccurrence_top_pairs.csv", index=False
        )

    # Taste communities: compute metrics and write analysis guide
    _write_taste_communities_analysis_guide(
        G_performer_shared_original=G_performer_shared_original,
        G_cooccur=G_cooccur,
        expanded=expanded,
        contestant_to_gender=contestant_to_gender,
        performer_edges=performer_edges if performer_edges else [],
        cooccur_edges=cooccur_edges if cooccur_edges else [],
        out_path=taste_communities_out / "analysis_guide.md",
        _split_artists_cell=_split_artists_cell,
    )

    canon_out = _angle_dir(out_dir, "canon_creation")

    # Contestant → original-artist relationship table and Sankey diagram
    contestant_artist = (
        expanded.groupby(["contestant", "original_artist"])
        .size()
        .reset_index(name="n_covers_pair")
    )
    contestant_artist.to_csv(canon_out / "contestant_to_original_counts.csv", index=False)

    fig_sankey = viz.sankey_contestant_to_original(contestant_artist)
    viz.write_figure(fig_sankey, canon_out / "contestant_to_original_sankey", html=True)

    # Section 4: temporal, guests, winners vs canon, medley summary
    expanded_decade = expanded.copy()
    expanded_decade["decade"] = (expanded_decade["song_year_parsed"] // 10 * 10).astype("Int64")

    # Overall distribution of original-song decades (figs written in canon_creation section below)
    decade_counts = expanded_decade["decade"].dropna().astype(int).value_counts().sort_index()
    decade_df = decade_counts.reset_index()
    decade_df.columns = ["decade", "count"]
    decade_df["decade"] = decade_df["decade"].astype(str) + "s"

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

    covers_per_year = covers.groupby("year").size().reset_index(name="n_covers")
    fig_trend = viz.bar_covers_per_year(covers_per_year)
    viz.write_figure(fig_trend, canon_out / "covers_per_edition", html=True)

    # Year-lag distribution: how far back contestants look each edition
    expanded_with_lag = expanded_decade.dropna(subset=["song_year_parsed"]).copy()
    expanded_with_lag["song_year_parsed"] = expanded_with_lag["song_year_parsed"].astype(int)
    expanded_with_lag["year_lag"] = expanded_with_lag["year"] - expanded_with_lag["song_year_parsed"]
    expanded_with_lag.to_csv(canon_out / "covers_with_year_lag.csv", index=False)

    fig_year_lag = viz.box_year_lag(expanded_with_lag)
    viz.write_figure(fig_year_lag, canon_out / "year_lag_by_edition", html=True)

    # Section: canon creation (canon_out already created above)

    # Original artists per performer (overall)
    per_performer = (
        expanded.groupby("contestant")
        .agg(
            n_original_artists=("original_artist", "nunique"),
            n_covers=("original_artist", "count"),
        )
        .reset_index()
    )
    per_performer.to_csv(canon_out / "original_artists_per_performer.csv", index=False)
    fig_canon_perf = viz.bar_original_artists_per_performer(per_performer)
    viz.write_figure(fig_canon_perf, canon_out / "original_artists_per_performer", html=True)

    # Original artists per performer per year
    per_performer_year = (
        expanded.groupby(["year", "contestant"])
        .agg(
            n_original_artists=("original_artist", "nunique"),
            n_covers=("original_artist", "count"),
        )
        .reset_index()
    )
    per_performer_year.to_csv(canon_out / "original_artists_per_performer_per_year.csv", index=False)

    # Uniqueness = 1 / global cover count of that original; per-performer mean originality
    artist_cover_count = expanded["original_artist"].value_counts()
    expanded_with_uniq = expanded.copy()
    expanded_with_uniq["uniqueness"] = expanded_with_uniq["original_artist"].map(
        lambda a: 1.0 / artist_cover_count.get(a, 1)
    )
    performer_orig = (
        expanded_with_uniq.groupby("contestant")
        .agg(
            n_covers=("uniqueness", "count"),
            mean_uniqueness=("uniqueness", "mean"),
        )
        .reset_index()
    )
    performer_orig.rename(columns={"mean_uniqueness": "mean_originality"}, inplace=True)
    performer_orig.to_csv(canon_out / "performer_originality.csv", index=False)

    # Canon metrics over time: mean n_originals and mean originality per edition
    canon_metrics_rows = []
    for y in sorted(expanded["year"].unique()):
        sub = expanded[expanded["year"] == y]
        ac_y = sub["original_artist"].value_counts()
        sub_uniq = sub.copy()
        sub_uniq["uniqueness"] = sub_uniq["original_artist"].map(lambda a: 1.0 / ac_y.get(a, 1))
        mean_orig = sub_uniq.groupby("contestant")["uniqueness"].mean().mean()
        mean_n = per_performer_year.loc[per_performer_year["year"] == y, "n_original_artists"].mean()
        canon_metrics_rows.append({
            "year": y,
            "mean_n_originals_per_performer": mean_n,
            "mean_originality": mean_orig,
        })
    canon_metrics_df = pd.DataFrame(canon_metrics_rows)
    canon_metrics_df.to_csv(canon_out / "canon_metrics_over_time.csv", index=False)
    fig_canon_metrics = viz.line_canon_metrics_over_time(canon_metrics_df)
    viz.write_figure(fig_canon_metrics, canon_out / "canon_metrics_over_time", html=True)

    # Female vs male original artists in canon (if gender data exists)
    artist_gender_by_artist_path = processed_dir / "artist_gender_by_artist.csv"
    if artist_gender_by_artist_path.exists():
        agb = pd.read_csv(artist_gender_by_artist_path)
        agb["artist_name"] = agb["artist_name"].astype(str).str.strip()
        name_to_gender_canon = agb.set_index("artist_name")["gender"].to_dict()
        expanded_canon_gender = expanded.copy()
        expanded_canon_gender["original_gender"] = expanded_canon_gender["original_artist"].apply(
            lambda cell: _row_gender_from_cell(cell, name_to_gender_canon)
        )
        canon_gender_by_year = (
            expanded_canon_gender.dropna(subset=["original_gender"])
            .groupby(["year", "original_gender"])
            .size()
            .reset_index(name="n_covers")
        )
        if not canon_gender_by_year.empty:
            canon_gender_by_year["total_per_year"] = canon_gender_by_year.groupby("year")["n_covers"].transform("sum")
            canon_gender_by_year["share"] = canon_gender_by_year["n_covers"] / canon_gender_by_year["total_per_year"]
            canon_gender_by_year.to_csv(canon_out / "canon_original_gender_share_per_year.csv", index=False)
            fig_canon_gender = viz.area_canon_original_gender_over_time(canon_gender_by_year)
            viz.write_figure(fig_canon_gender, canon_out / "canon_original_gender_share_over_time", html=True)

    # Top artists and top songs (self-contained in canon folder)
    author_counts.to_csv(canon_out / "top_artists.csv", index=False)
    song_counts.to_csv(canon_out / "top_songs.csv", index=False)
    fig_canon_authors = viz.bar_top_authors(author_counts, title="Canon: top most covered original artists")
    viz.write_figure(fig_canon_authors, canon_out / "top_artists", html=True)
    fig_canon_songs = viz.bar_top_songs(song_counts, title="Canon: top most covered songs")
    viz.write_figure(fig_canon_songs, canon_out / "top_songs", html=True)

    # Most covered decades
    decade_df.to_csv(canon_out / "decade_distribution.csv", index=False)
    decade_per_year.to_csv(canon_out / "decade_mix_per_year.csv", index=False)
    fig_canon_decade = viz.bar_decade(decade_df, title="Canon: distribution of original-song decades")
    viz.write_figure(fig_canon_decade, canon_out / "decade_distribution", html=True)
    fig_canon_decade_time = viz.area_decade_over_time(
        decade_per_year, title="Canon: share of covers by decade over editions"
    )
    viz.write_figure(fig_canon_decade_time, canon_out / "decade_mix_over_time", html=True)

    _write_canon_creation_analysis_guide(canon_out)

    # Section 5: gender mapping and flows (requires manual annotation file)
    gender_out_dir = _angle_dir(out_dir, "gender")

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

        # Map canonical artist names to gender and aggregate per performance
        name_to_gender = artist_gender_by_artist.set_index("artist_name")["gender"].to_dict()

        expanded_gender = expanded.copy()
        expanded_gender["original_gender"] = expanded_gender["original_artist"].apply(
            lambda cell: _row_gender_from_cell(cell, name_to_gender)
        )
        expanded_gender["performer_gender"] = expanded_gender["contestant"].apply(
            lambda cell: _row_gender_from_cell(cell, name_to_gender)
        )

        # Share of covers with female original artists per edition
        mask_original_known = expanded_gender["original_gender"].notna() & (
            expanded_gender["original_gender"] != "Unknown"
        )
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
            gender_by_year.to_csv(gender_out_dir / "gender_original_share_per_year.csv", index=False)

        # Flow between performer gender and original-artist gender
        mask_both_known = expanded_gender["performer_gender"].notna() & expanded_gender["original_gender"].notna()
        gender_flow = (
            expanded_gender[mask_both_known]
            .groupby(["performer_gender", "original_gender"])
            .size()
            .reset_index(name="n_covers")
        )
        if not gender_flow.empty:
            gender_flow.to_csv(
                gender_out_dir / "gender_performer_to_original_counts.csv", index=False
            )

            fig_gender_sankey = viz.sankey_gender_flow(gender_flow)
            viz.write_figure(
                fig_gender_sankey,
                gender_out_dir / "gender_performer_to_original_sankey",
                html=True,
            )

        # Performer-gender distribution per year (stacked shares, all originals)
        performer_gender_by_year = (
            expanded_gender.dropna(subset=["performer_gender"])
            .groupby(["year", "performer_gender"])
            .size()
            .reset_index(name="n_covers")
        )
        if not performer_gender_by_year.empty:
            performer_gender_by_year["total_per_year"] = performer_gender_by_year.groupby("year")[
                "n_covers"
            ].transform("sum")
            performer_gender_by_year["share"] = (
                performer_gender_by_year["n_covers"] / performer_gender_by_year["total_per_year"]
            )
            performer_gender_by_year.to_csv(
                gender_out_dir / "gender_performer_share_per_year.csv", index=False
            )

            fig_perf_gender_bar = viz.bar_performer_gender_over_time(
                performer_gender_by_year
            )
            viz.write_figure(
                fig_perf_gender_bar,
                gender_out_dir / "gender_performer_share_over_time",
                html=True,
            )

        # Performer-gender distribution per year, restricted to covers of songs originally by women.
        # Each bar's total height reflects the share of *all* covers that year whose original artist is a woman,
        # and segments within the bar show how those covers are split across performer genders.
        perf_over_female_orig = expanded_gender[
            (expanded_gender["original_gender"] == "F") & expanded_gender["performer_gender"].notna()
        ].copy()
        performer_gender_over_female = (
            perf_over_female_orig.groupby(["year", "performer_gender"])
            .size()
            .reset_index(name="n_covers")
        )
        if not performer_gender_over_female.empty:
            # Per-year totals: number of covers with female originals, and total covers.
            female_per_year = (
                perf_over_female_orig.groupby("year")["original_gender"].size().to_dict()
            )
            total_per_year_all = (
                expanded_gender.groupby("year")["original_gender"].size().to_dict()
            )

            performer_gender_over_female["n_female_original_per_year"] = (
                performer_gender_over_female["year"].map(female_per_year)
            )
            performer_gender_over_female["n_all_covers_per_year"] = (
                performer_gender_over_female["year"].map(total_per_year_all)
            )
            performer_gender_over_female["share"] = (
                performer_gender_over_female["n_covers"]
                / performer_gender_over_female["n_all_covers_per_year"]
            )
            performer_gender_over_female.to_csv(
                gender_out_dir / "gender_performer_mix_over_female_originals_per_year.csv",
                index=False,
            )

            if not gender_by_year.empty:
                fig_combined = viz.combined_gender_original_share_and_performer_stacked(
                    gender_by_year, performer_gender_over_female
                )
                viz.write_figure(
                    fig_combined,
                    gender_out_dir / "gender_original_share_and_performer_over_female_originals_over_time",
                    html=True,
                )

        # Contingency table and conditional probabilities for F/M performer vs original
        fm_mask = expanded_gender["performer_gender"].isin(["F", "M"]) & expanded_gender[
            "original_gender"
        ].isin(["F", "M"])
        gender_contingency_stats = None
        gender_contingency_path = (
            gender_out_dir / "gender_contingency_performance_level.csv"
        )
        if fm_mask.any():
            fm_table = (
                expanded_gender[fm_mask]
                .groupby(["performer_gender", "original_gender"])
                .size()
                .unstack(fill_value=0)
            )
            fm_table = fm_table.reindex(index=["F", "M"], columns=["F", "M"], fill_value=0)
            fm_table.index.name = "performer_gender"
            fm_table.columns.name = "original_gender"

            total = int(fm_table.values.sum())
            if total > 0:
                row_sums = fm_table.sum(axis=1)
                col_sums = fm_table.sum(axis=0)

                # Baseline share of female originals among F/M-only covers
                baseline_share_f = float(col_sums["F"] / total) if col_sums["F"] > 0 else 0.0

                p_f_given_f = float(
                    fm_table.loc["F", "F"] / row_sums["F"]
                ) if row_sums["F"] > 0 else float("nan")
                p_f_given_m = float(
                    fm_table.loc["M", "F"] / row_sums["M"]
                ) if row_sums["M"] > 0 else float("nan")

                # Chi-squared test of independence for 2x2 table (1 df)
                import math

                chi2 = 0.0
                for i in ["F", "M"]:
                    for j in ["F", "M"]:
                        obs = float(fm_table.loc[i, j])
                        exp = float(row_sums[i] * col_sums[j] / total) if total > 0 else 0.0
                        if exp > 0:
                            chi2 += (obs - exp) ** 2 / exp
                # For 1 degree of freedom, p-value = erfc(sqrt(chi2 / 2))
                p_value = float(math.erfc(math.sqrt(chi2 / 2.0))) if chi2 >= 0 else float("nan")

                # Baseline share of original gender (bar chart)
                baseline_original_df = pd.DataFrame(
                    {
                        "original_gender": ["F", "M"],
                        "share": [
                            float(col_sums["F"] / total),
                            float(col_sums["M"] / total),
                        ],
                    }
                )
                fig_baseline = viz.bar_baseline_original_gender(baseline_original_df)
                viz.write_figure(
                    fig_baseline,
                    gender_out_dir / "gender_baseline_original_share",
                    html=True,
                )

                # Heatmap of conditional probabilities P(original | performer) for F/M covers
                fig_gender_heatmap = viz.heatmap_gender_conditional_probs(fm_table)
                viz.write_figure(
                    fig_gender_heatmap,
                    gender_out_dir / "gender_performer_original_prob_heatmap",
                    html=True,
                )

                fm_table_reset = (
                    fm_table.reset_index()
                    .melt(
                        id_vars="performer_gender",
                        value_vars=["F", "M"],
                        var_name="original_gender",
                        value_name="n_covers",
                    )
                )
                fm_table_reset.to_csv(gender_contingency_path, index=False)

                gender_contingency_stats = {
                    "n_total": total,
                    "baseline_share_f": baseline_share_f,
                    "p_f_given_f": p_f_given_f,
                    "p_f_given_m": p_f_given_m,
                    "chi2": chi2,
                    "p_value": p_value,
                }

    # Section 6: self-covers and self-tributes
    self_cover_out = _angle_dir(out_dir, "self_cover")
    expanded_self = expanded.copy()
    expanded_self["is_self_cover"] = expanded_self.apply(
        lambda row: bool(
            set(_split_artists_cell(row["contestant"]))
            & set(_split_artists_cell(row["original_artist"]))
        ),
        axis=1,
    )
    self_covers = expanded_self[expanded_self["is_self_cover"]]
    self_covers.to_csv(self_cover_out / "self_covers_detailed.csv", index=False)

    self_covers_per_year = self_covers.groupby("year").size().reset_index(name="n_self_covers")
    self_covers_per_year.to_csv(self_cover_out / "self_covers_per_year.csv", index=False)

    if not self_covers_per_year.empty:
        fig_self_covers = viz.bar_self_covers_per_year(self_covers_per_year)
        viz.write_figure(fig_self_covers, self_cover_out / "self_covers_per_year", html=True)

    if not self_covers.empty:
        performer_year = (
            self_covers.groupby(["year", "contestant", "edition"])
            .agg(
                n_self_covers=("song", "count"),
                songs_str=("song", lambda s: " | ".join(s.astype(str))),
            )
            .reset_index()
        )
        songs_per_performer = (
            expanded.groupby(["year", "contestant"]).size().reset_index(name="n_songs_performer")
        )
        performer_year = performer_year.merge(
            songs_per_performer, on=["year", "contestant"], how="left"
        )
        performer_year["n_songs_performer"] = performer_year["n_songs_performer"].fillna(0).astype(int)
        performer_year.to_csv(self_cover_out / "self_covers_per_performer_year.csv", index=False)
        fig_scatter = viz.scatter_self_covers_performer(performer_year)
        viz.write_figure(fig_scatter, self_cover_out / "self_covers_scatter_performer", html=True)

        self_covers_decade = self_covers.dropna(subset=["song_year_parsed"]).copy()
        self_covers_decade["song_year_parsed"] = self_covers_decade["song_year_parsed"].astype(int)
        self_covers_decade["decade"] = (
            (self_covers_decade["song_year_parsed"] // 10 * 10).astype(int).astype(str) + "s"
        )
        self_covers_by_decade = (
            self_covers_decade.groupby("decade").size().reset_index(name="count").sort_values("decade")
        )
        self_covers_by_decade.to_csv(self_cover_out / "self_covers_by_decade.csv", index=False)
        fig_decade = viz.bar_self_covers_by_decade(self_covers_by_decade)
        viz.write_figure(fig_decade, self_cover_out / "self_covers_by_decade", html=True)

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
            "Gender-based cover probabilities (performer vs original, F/M only):",
            (
                f"  Total covers with known performer and original gender (F/M only): "
                f"{gender_contingency_stats['n_total']}"
                if gender_contingency_stats is not None
                else "  (Gender contingency statistics unavailable: missing or insufficient gender data.)"
            ),
        ]
    )
    if gender_contingency_stats is not None:
        summary_lines.extend(
            [
                (
                    "  Baseline share of female originals among these covers: "
                    f"{gender_contingency_stats['baseline_share_f']:.3f}"
                ),
                (
                    "  P(original F | performer F): "
                    f"{gender_contingency_stats['p_f_given_f']:.3f}"
                    if pd.notna(gender_contingency_stats["p_f_given_f"])
                    else "  P(original F | performer F): undefined (no female performers in F/M subset)."
                ),
                (
                    "  P(original F | performer M): "
                    f"{gender_contingency_stats['p_f_given_m']:.3f}"
                    if pd.notna(gender_contingency_stats["p_f_given_m"])
                    else "  P(original F | performer M): undefined (no male performers in F/M subset)."
                ),
                (
                    "  Chi-squared test (1 df): "
                    f"chi2={gender_contingency_stats['chi2']:.3f}, "
                    f"p-value={gender_contingency_stats['p_value']:.4f}"
                ),
                "",
                "Multi-artist performances (e.g. duets or multiple original artists) are treated as a single "
                "observation with aggregated performer_gender and original_gender (F, M, Group, Mixed, or Unknown).",
            ]
        )
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
