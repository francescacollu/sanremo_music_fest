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


def write_figure(fig, out_path: Path, html=True, png=False):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if html:
        fig.write_html(str(out_path.with_suffix(".html")))
    if png:
        fig.write_image(str(out_path.with_suffix(".png")))


def network_cooccurrence(G: nx.Graph, top_n: int = 50, title: str | None = None):
    """Simple co-occurrence network visualization for original artists.

    Nodes are artists; edges connect artists that co-occur in at least one
    edition, with edge width proportional to weight.
    """
    # Select top nodes by weighted degree to keep the plot readable
    degrees = dict(G.degree(weight="weight"))
    if not degrees:
        raise ValueError("Graph has no nodes; cannot plot co-occurrence network.")

    nodes_sorted = sorted(degrees.items(), key=lambda x: -x[1])[:top_n]
    keep_nodes = {n for n, _ in nodes_sorted}
    subG = G.subgraph(keep_nodes).copy()

    # Spring-layout positions for the subgraph
    pos = nx.spring_layout(subG, k=0.6, seed=42)

    # Build edge traces
    edge_x = []
    edge_y = []
    edge_widths = []
    for u, v, data in subG.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        w = data.get("weight", 1)
        edge_widths.append(w)

    # Normalize edge widths for display
    if edge_widths:
        min_w = min(edge_widths)
        max_w = max(edge_widths)
        span = max_w - min_w if max_w != min_w else 1.0
        edge_sizes = [1.0 + 4.0 * (w - min_w) / span for w in edge_widths]
    else:
        edge_sizes = []

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="rgba(150,150,150,0.5)"),
        hoverinfo="none",
        mode="lines",
    )

    # Build node trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(10 + 10 * degrees.get(node, 0))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=node_size,
            color="rgba(31,119,180,0.8)",
            line=dict(width=0.5, color="rgba(50,50,50,0.8)"),
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=title or "Co-occurrence network of original artists (cover night)",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


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
    return fig


def sankey_gender_flow(gender_flow_df, title: str | None = None):
    """Sankey from performer gender to original-artist gender."""
    if gender_flow_df.empty:
        raise ValueError("Empty gender flow table; cannot build Sankey diagram.")

    df = gender_flow_df.copy()

    performer_cats = sorted(df["performer_gender"].unique())
    original_cats = sorted(df["original_gender"].unique())

    performer_labels = [f"P: {g}" for g in performer_cats]
    original_labels = [f"O: {g}" for g in original_cats]
    labels = performer_labels + original_labels

    index = {name: i for i, name in enumerate(labels)}

    sources = [index[f"P: {g}"] for g in df["performer_gender"]]
    targets = [index[f"O: {g}"] for g in df["original_gender"]]
    values = df["n_covers"].tolist()

    sankey = go.Sankey(
        node=dict(label=labels, pad=10, thickness=14),
        link=dict(source=sources, target=targets, value=values),
    )

    fig = go.Figure(sankey)
    fig.update_layout(
        title=title or "Who covers whom, by gender",
        font=dict(size=10),
    )
    return fig
