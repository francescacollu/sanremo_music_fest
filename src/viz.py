"""Plotting helpers: bar charts, simple networks, and figure output."""
from pathlib import Path

import networkx as nx
import plotly.express as px
import plotly.graph_objects as go


def bar_top_songs(song_counts_df, top_n=20, title=None):
    top = song_counts_df.head(top_n)
    fig = px.bar(
        top,
        x="song",
        y="count",
        title=title or "Top most covered songs (including medley constituents)",
        labels={"count": "Number of times covered", "song": "Song"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def bar_top_authors(author_counts_df, top_n=20, title=None):
    top = author_counts_df.head(top_n)
    fig = px.bar(
        top,
        x="original_artist",
        y="count",
        title=title or "Top most covered original artists (including medley constituents)",
        labels={"count": "Number of covers", "original_artist": "Original artist"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def bar_decade(decade_counts_df, title=None):
    fig = px.bar(
        decade_counts_df,
        x="decade",
        y="count",
        title=title or "Distribution of original song decades (cover night)",
    )
    return fig


def bar_covers_per_year(per_year_df, title=None):
    fig = px.bar(
        per_year_df,
        x="year",
        y="n_covers",
        title=title or "Number of cover performances per edition",
    )
    return fig


def bar_self_covers_per_year(self_per_year_df, title=None):
    fig = px.bar(
        self_per_year_df,
        x="year",
        y="n_self_covers",
        title=title or "Number of self-covers/self-tributes per edition",
        labels={"year": "Festival year", "n_self_covers": "Number of self-covers"},
    )
    return fig


def scatter_self_covers_performer(performer_year_df, title=None):
    """Scatter: year vs n_self_covers, one marker per performer, size = n_songs_performer.
    Uses a base + scale so single-song performers stay visible. X-jitter spreads overlapping points."""
    plot_df = performer_year_df.copy()
    plot_df["_size"] = 10 + plot_df["n_songs_performer"] * 5
    g = plot_df.groupby(["year", "n_self_covers"], sort=False)
    rank = g["contestant"].transform(lambda s: range(len(s)))
    n_per_group = g["contestant"].transform("count")
    plot_df["_x_jitter"] = plot_df["year"] + (rank - (n_per_group - 1) / 2) * 0.12
    fig = px.scatter(
        plot_df,
        x="_x_jitter",
        y="n_self_covers",
        size="_size",
        hover_data={
            "year": True,
            "n_self_covers": True,
            "n_songs_performer": True,
            "contestant": True,
            "edition": True,
            "songs_str": True,
            "_x_jitter": False,
            "_size": False,
        },
        title=title or "Self-covers per edition (one marker per performer)",
        labels={
            "year": "Festival year",
            "n_self_covers": "Number of self-covers",
            "n_songs_performer": "Songs performed in edition",
        },
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="gray")))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(tickvals=plot_df["year"].drop_duplicates().sort_values(), title_text="Festival year")
    return fig


def bar_self_covers_by_decade(decade_counts_df, title=None):
    return bar_decade(
        decade_counts_df,
        title=title or "Number of self-covers by decade of the original song",
    )


def write_figure(fig, out_path: Path, html=True, png=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if html:
        fig.write_html(str(out_path.with_suffix(".html")))
    if png:
        fig.write_image(str(out_path.with_suffix(".png")))


# Default colors for categorical node color (e.g. gender). Unknown/other use default.
_NETWORK_NODE_COLOR_MAP = {
    "F": "#e74c3c",
    "M": "#3498db",
    "Group": "#2ecc71",
    "Mixed": "#9b59b6",
    "Unknown": "#95a5a6",
}

# Gender visualizations: single-source palette and typography
GENDER_BG = "#ffffff"
GENDER_MALE = "#14213d"
GENDER_FEMALE = "#fca311"
GENDER_MIXED = "#e5e5e5"
GENDER_FONT = dict(family="Georgia, serif", size=15)
GENDER_COLOR_MAP = {"F": GENDER_FEMALE, "M": GENDER_MALE, "Mixed": GENDER_MIXED, "Group": GENDER_MIXED}


def _apply_gender_layout(fig):
    """Apply shared layout to gender figures: simple_white theme, background, Georgia 20px, centered title."""
    fig.update_layout(
        template="simple_white",
        paper_bgcolor=GENDER_BG,
        plot_bgcolor=GENDER_BG,
        font=GENDER_FONT,
        title_font=dict(family="Georgia, serif", size=20),
        title_x=0.5,
    )


def _network_graph(
    G: nx.Graph,
    top_n: int,
    title: str,
    node_size_attr: dict | None = None,
    node_hover_suffix: str = " (weighted degree)",
    node_color_attr: dict | None = None,
    min_node_size: float = 8,
    max_node_size: float = 45,
    edge_width_min: float = 0.8,
    edge_width_max: float = 4,
):
    """Shared spring-layout network: top_n nodes by weighted degree, normalized node size, edge width by weight, hovers. Optional node color by category with legend."""
    degrees = dict(G.degree(weight="weight"))
    if not degrees:
        raise ValueError("Graph has no nodes; cannot plot network.")

    nodes_sorted = sorted(degrees.items(), key=lambda x: -x[1])[:top_n]
    keep_nodes = {n for n, _ in nodes_sorted}
    subG = G.subgraph(keep_nodes).copy()

    pos = nx.spring_layout(subG, k=0.6, seed=42)

    # Node size: node_size_attr if provided, else weighted degree; normalize to [min_node_size, max_node_size]
    size_values = {}
    for n in keep_nodes:
        size_values[n] = (node_size_attr.get(n, degrees.get(n, 0)) if node_size_attr is not None else degrees.get(n, 0))
    vals = list(size_values.values())
    d_min, d_max = min(vals), max(vals)
    if d_max > d_min:
        node_sizes = [
            min_node_size + (max_node_size - min_node_size) * (size_values[n] - d_min) / (d_max - d_min)
            for n in pos
        ]
    else:
        node_sizes = [min_node_size] * len(pos)

    # One trace per edge: width by weight, hover "A and B share N original artist(s)."
    edge_weights = [subG[u][v].get("weight", 1) for u, v in subG.edges()]
    w_min = min(edge_weights) if edge_weights else 1
    w_max = max(edge_weights) if edge_weights else 1
    width_scale = (edge_width_max - edge_width_min) / (w_max - w_min) if w_max > w_min else 0

    edge_traces = []
    for u, v, data in subG.edges(data=True):
        w = data.get("weight", 1)
        line_w = edge_width_min + width_scale * (w - w_min) if w_max > w_min else edge_width_min
        edge_traces.append(
            go.Scatter(
                x=[pos[u][0], pos[v][0], None],
                y=[pos[u][1], pos[v][1], None],
                line=dict(width=line_w, color="rgba(100,100,100,0.7)"),
                mode="lines",
                hoverinfo="text",
                text=f"{u} and {v} share {int(w)} original artist(s).",
            )
        )

    node_text = list(pos.keys())
    node_hover = [f"{n} — {int(degrees.get(n, 0))}{node_hover_suffix}" for n in pos]
    size_by_node = dict(zip(pos.keys(), node_sizes))

    if node_color_attr:
        # One trace per category for legend; fallback missing to "Unknown"
        default_cat = "Unknown"
        cat_color_map = {**_NETWORK_NODE_COLOR_MAP}
        node_traces = []
        from collections import defaultdict
        by_cat = defaultdict(list)
        for n in pos:
            cat = node_color_attr.get(n)
            if cat is None or (isinstance(cat, float) and str(cat) == "nan"):
                cat = default_cat
            else:
                cat = str(cat).strip() or default_cat
            if cat not in cat_color_map:
                cat_color_map[cat] = "#7f8c8d"
            by_cat[cat].append(n)
        order = list(pos.keys())
        node_to_idx = {n: i for i, n in enumerate(order)}
        for cat in sorted(by_cat.keys()):
            nodes_cat = by_cat[cat]
            node_traces.append(
                go.Scatter(
                    x=[pos[n][0] for n in nodes_cat],
                    y=[pos[n][1] for n in nodes_cat],
                    mode="markers+text",
                    text=[node_text[node_to_idx[n]] for n in nodes_cat],
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=[node_hover[node_to_idx[n]] for n in nodes_cat],
                    marker=dict(
                        size=[size_by_node[n] for n in nodes_cat],
                        color=cat_color_map[cat],
                        line=dict(width=0.5, color="rgba(50,50,50,0.8)"),
                    ),
                    name=cat,
                    legendgroup=cat,
                )
            )
        fig = go.Figure(data=edge_traces + node_traces)
        fig.update_layout(
            title=title,
            showlegend=True,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
    else:
        node_x = [pos[n][0] for n in pos]
        node_y = [pos[n][1] for n in pos]
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            hoverinfo="text",
            hovertext=node_hover,
            marker=dict(
                size=node_sizes,
                color="rgba(31,119,180,0.8)",
                line=dict(width=0.5, color="rgba(50,50,50,0.8)"),
            ),
        )
        fig = go.Figure(data=edge_traces + [node_trace])
        fig.update_layout(
            title=title,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
            margin=dict(l=10, r=10, t=40, b=10),
        )
    return fig


def network_cooccurrence(
    G: nx.Graph,
    top_n: int = 50,
    title: str | None = None,
    node_size_attr: dict | None = None,
    node_hover_suffix: str = " (weighted degree)",
    node_color_attr: dict | None = None,
):
    """Co-occurrence network: artists as nodes, edge width by weight."""
    return _network_graph(
        G,
        top_n=top_n,
        title=title or "Co-occurrence network of original artists (cover night)",
        node_size_attr=node_size_attr,
        node_hover_suffix=node_hover_suffix,
        node_color_attr=node_color_attr,
    )


def network_graph(
    G: nx.Graph,
    top_n: int = 50,
    title: str | None = None,
    node_size_attr: dict | None = None,
    node_hover_suffix: str = " (weighted degree)",
    node_color_attr: dict | None = None,
):
    """Generic network plot (spring layout, normalized node size, edge width by weight)."""
    return _network_graph(
        G,
        top_n,
        title=title or "Network",
        node_size_attr=node_size_attr,
        node_hover_suffix=node_hover_suffix,
        node_color_attr=node_color_attr,
    )


def area_decade_over_time(decade_per_year_df, title: str | None = None):
    """Stacked area chart of original-song decades across editions."""
    fig = px.area(
        decade_per_year_df,
        x="year",
        y="share",
        color="decade",
        title=title or "Share of covers by original-song decade over editions",
        labels={"share": "Share of covers", "year": "Festival year", "decade": "Original decade"},
    )
    return fig


def box_year_lag(expanded_with_lag_df, title: str | None = None):
    """Boxplot of year lag (festival year - original song year) per edition."""
    fig = px.box(
        expanded_with_lag_df,
        x="year",
        y="year_lag",
        points="suspectedoutliers",
        title=title or "How far back contestants look: year lag by edition",
        labels={"year": "Festival year", "year_lag": "Years between original and cover"},
    )
    return fig


def bar_original_artists_per_performer(per_performer_df, top_n=20, title: str | None = None):
    """Bar chart of performers by number of distinct original artists (canon diversity)."""
    top = per_performer_df.nlargest(top_n, "n_original_artists")
    fig = px.bar(
        top,
        x="contestant",
        y="n_original_artists",
        title=title or "Top performers by number of distinct original artists covered",
        labels={"contestant": "Performer", "n_original_artists": "Number of distinct original artists"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def line_canon_metrics_over_time(metrics_df, title: str | None = None):
    """Line chart of mean canon metrics per edition (diversity and originality)."""
    fig = px.line(
        metrics_df,
        x="year",
        y=["mean_n_originals_per_performer", "mean_originality"],
        markers=True,
        title=title or "Canon diversity and originality over editions",
        labels={"year": "Festival year", "value": "Mean value"},
    )
    name_map = {
        "mean_n_originals_per_performer": "Mean distinct originals per performer",
        "mean_originality": "Mean originality",
    }
    fig.for_each_trace(lambda t: t.update(name=name_map.get(t.name, t.name)))
    fig.update_layout(yaxis_title="Mean value", legend_title="Metric")
    return fig


def area_canon_original_gender_over_time(canon_gender_per_year_df, title: str | None = None):
    """Stacked area of share of covers by original-artist gender (F/M/etc.) over editions."""
    fig = px.area(
        canon_gender_per_year_df,
        x="year",
        y="share",
        color="original_gender",
        title=title or "Share of canon by original-artist gender over editions",
        labels={"year": "Festival year", "share": "Share of covers", "original_gender": "Original artist gender"},
    )
    return fig


def sankey_contestant_to_original(contestant_artist_df, top_pairs: int = 60, title: str | None = None):
    """Sankey diagram mapping contestants to original artists."""
    if contestant_artist_df.empty:
        raise ValueError("Empty contestant-artist table; cannot build Sankey diagram.")

    df = contestant_artist_df.sort_values("n_covers_pair", ascending=False).head(top_pairs).copy()

    contestants = sorted(df["contestant"].unique())
    originals = sorted(df["original_artist"].unique())

    contestant_labels = [f"C: {c}" for c in contestants]
    original_labels = [f"A: {a}" for a in originals]
    labels = contestant_labels + original_labels

    index = {name: i for i, name in enumerate(labels)}

    sources = [index[f"C: {c}"] for c in df["contestant"]]
    targets = [index[f"A: {a}"] for a in df["original_artist"]]
    values = df["n_covers_pair"].tolist()

    sankey = go.Sankey(
        node=dict(label=labels, pad=10, thickness=12),
        link=dict(source=sources, target=targets, value=values),
    )

    fig = go.Figure(sankey)
    fig.update_layout(
        title=title or "Who covers whom: contestants → original artists",
        font=dict(size=10),
    )
    return fig


def line_share_female_originals(gender_by_year_df, title: str | None = None):
    """Line chart of share of covers whose original artist is a woman."""
    fig = px.line(
        gender_by_year_df,
        x="year",
        y="share_female_original",
        markers=True,
        title=title or "Share of cover-night songs originally by women",
        labels={"year": "Festival year", "share_female_original": "Share of covers (original by women)"},
    )
    _apply_gender_layout(fig)
    return fig


def sankey_gender_flow(
    gender_flow_df, title: str | None = None, subtitle: str | None = None
):
    """Sankey from performer gender to original-artist gender."""
    if gender_flow_df.empty:
        raise ValueError("Empty gender flow table; cannot build Sankey diagram.")

    df = gender_flow_df.copy()

    performer_cats = sorted(df["performer_gender"].unique())
    original_cats = sorted(df["original_gender"].unique())

    # Map raw gender codes to Italian labels (for hovers and legend), but
    # keep node labels visually empty so only colors and side titles carry meaning.
    gender_label_map = {"M": "Maschile", "F": "Femminile", "Mixed": "Misto", "Group": "Misto"}
    labels = [""] * (len(performer_cats) + len(original_cats))

    # Indices are based on raw gender categories (not label text)
    performer_index = {g: i for i, g in enumerate(performer_cats)}
    n_perf = len(performer_cats)
    original_index = {g: i for i, g in enumerate(original_cats)}

    sources = [performer_index[g] for g in df["performer_gender"]]
    targets = [n_perf + original_index[g] for g in df["original_gender"]]
    values = df["n_covers"].tolist()
    hover_texts = [
        f"Genere del cantante in gara: {gender_label_map.get(p, str(p))} -> "
        f"Genere dell'artista originale: {gender_label_map.get(o, str(o))}: {n} cover"
        for p, o, n in zip(df["performer_gender"], df["original_gender"], values)
    ]

    # Node colors by gender, using the shared palette
    node_colors = (
        [GENDER_COLOR_MAP.get(g, GENDER_MIXED) for g in performer_cats]
        + [GENDER_COLOR_MAP.get(g, GENDER_MIXED) for g in original_cats]
    )
    # Link colors follow the performer (source) gender
    link_colors = [GENDER_COLOR_MAP.get(p, GENDER_MIXED) for p in df["performer_gender"]]

    sankey = go.Sankey(
        node=dict(
            label=labels,
            pad=10,
            thickness=14,
            color=node_colors,
            # Plotly Sankey supports a single border color (not per-node), so we make it transparent
            # to avoid an unwanted black outline around nodes.
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
            customdata=hover_texts,
            hovertemplate="%{customdata}<extra></extra>",
        ),
    )

    fig = go.Figure(sankey)

    # Vertical side labels for the two columns
    fig.add_annotation(
        text="Cantanti in gara",
        x=-0.03,
        xref="paper",
        y=0.5,
        yref="paper",
        textangle=-90,
        showarrow=False,
        font=GENDER_FONT,
        font_size=20,
    )
    fig.add_annotation(
        text="Interpreti originali delle cover",
        x=1.03,
        xref="paper",
        y=0.5,
        yref="paper",
        textangle=90,
        showarrow=False,
        font=GENDER_FONT,
        font_size=20,
    )

    # Legend entries for gender colors (Maschile, Femminile, Misto)
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=GENDER_MALE),
            name="Maschile",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=GENDER_FEMALE),
            name="Femminile",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=10, color=GENDER_MIXED),
            name="Misto",
            showlegend=True,
        )
    )
    main_title = title or "<b>Sanremo, serata cover: chi canta chi, per genere</b>"
    sub = subtitle or "Ogni flusso mostra il numero di cover; il colore del flusso segue il genere del cantante in gara."
    fig.update_layout(
        title=dict(text=f"{main_title}<br><sup>{sub}</sup>", x=0.5),
        showlegend=True,
        legend=dict(
            title_text="Genere",
            orientation="h",
            x=0.5,
            xanchor="center",
            y=-0.01,
            yanchor="top",
        ),
        margin=dict(b=120),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    _apply_gender_layout(fig)
    return fig


def bar_performer_gender_over_time(performer_gender_by_year_df, title: str | None = None):
    """Stacked bar chart of performer gender shares per edition."""
    fig = px.bar(
        performer_gender_by_year_df,
        x="year",
        y="share",
        color="performer_gender",
        color_discrete_map=GENDER_COLOR_MAP,
        title=title or "Share of cover-night performances by performer gender over editions",
        labels={
            "year": "Festival year",
            "share": "Share of performances",
            "performer_gender": "Performer gender",
        },
        barmode="stack",
    )
    _apply_gender_layout(fig)
    return fig


def bar_performer_gender_over_female_originals_over_time(
    performer_gender_over_female_df, title: str | None = None
):
    """Stacked bar of performer gender shares per edition, over songs originally by women."""
    fig = px.bar(
        performer_gender_over_female_df,
        x="year",
        y="share",
        color="performer_gender",
        color_discrete_map=GENDER_COLOR_MAP,
        title=title
        or "Share of performances by performer gender (only songs originally by women)",
        labels={
            "year": "Festival year",
            "share": "Share of performances (original by women)",
            "performer_gender": "Performer gender",
        },
        barmode="stack",
    )
    _apply_gender_layout(fig)
    return fig


def combined_gender_original_share_and_performer_stacked(
    gender_by_year_df, performer_gender_over_female_df, title: str | None = None
):
    """Single figure: line of share of covers originally by women + stacked bar of performer gender over female-originals."""
    fig = go.Figure()
    # Stacked bar traces first (one per performer_gender)
    for cat in sorted(performer_gender_over_female_df["performer_gender"].unique()):
        sub = performer_gender_over_female_df[performer_gender_over_female_df["performer_gender"] == cat]
        fig.add_trace(
            go.Bar(
                x=sub["year"],
                y=sub["share"],
                name=cat,
                marker_color=GENDER_COLOR_MAP.get(cat, GENDER_MIXED),
                legendgroup="performer",
            )
        )
    # Line: share of covers whose original artist is a woman
    fig.add_trace(
        go.Scatter(
            x=gender_by_year_df["year"],
            y=gender_by_year_df["share_female_original"],
            mode="lines+markers",
            name="Share of covers (original by women)",
            line=dict(color=GENDER_MALE, width=2),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        barmode="stack",
        title=title or "Share of covers originally by women and performer gender mix over female-originals",
        xaxis_title="Festival year",
        yaxis_title="Share",
        yaxis_tickformat=".0%",
    )
    _apply_gender_layout(fig)
    return fig


def bar_baseline_original_gender(baseline_df, title: str | None = None):
    """Bar chart of baseline share of original artist gender (F/M) among F/M-only covers.

    Expects a DataFrame with columns: original_gender (F, M), share (proportion in [0,1]).
    """
    fig = px.bar(
        baseline_df,
        x="original_gender",
        y="share",
        color="original_gender",
        color_discrete_map=GENDER_COLOR_MAP,
        title=title or "Baseline share of original artist gender (F/M covers)",
        labels={"original_gender": "Original artist gender", "share": "Share of covers"},
        text_auto=".1%",
    )
    fig.update_layout(yaxis_tickformat=".0%")
    _apply_gender_layout(fig)
    return fig


def heatmap_gender_conditional_probs(fm_table, title: str | None = None):
    """Heatmap of P(original gender | performer gender) for F/M covers.

    Expects a 2x2 table with index = performer_gender, columns = original_gender,
    and values = counts.
    """
    if fm_table.empty:
        raise ValueError("Empty contingency table; cannot build gender heatmap.")

    table = fm_table.astype(float).copy()
    row_sums = table.sum(axis=1)
    # Avoid division by zero: rows with zero total become all-NaN
    probs = table.div(row_sums.replace(0, float("nan")), axis=0)

    plot_title = title or "Probability that the original artist is F/M, conditional on performer gender (F/M covers)"
    fig = px.imshow(
        probs,
        x=list(probs.columns),
        y=list(probs.index),
        text_auto=".2f",
        color_continuous_scale=[GENDER_MALE, GENDER_FEMALE],
        labels={
            "x": "Original artist gender",
            "y": "Performer gender",
            "color": "P(original | performer)",
        },
        title=plot_title,
    )
    fig.update_xaxes(side="top")
    fig.add_annotation(
        text="Each row is a performer gender, each column is original artist gender. Cell (i,j) is the probability that the original artist has gender j given performer gender i.",
        x=0.5,
        xref="paper",
        y=1.02,
        yref="paper",
        showarrow=False,
        font=GENDER_FONT,
    )
    _apply_gender_layout(fig)
    return fig
