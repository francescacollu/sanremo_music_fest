# Canon creation — Analysis guide

This folder holds outputs for the **canon creation** angle: what gets covered, how diverse or original performers' choices are, and how that changes over time.

---

## 1. Data and outputs in this folder

| File | Description |
|------|-------------|
| `contestant_to_original_counts.csv` | Count of covers per (contestant, original_artist) pair. |
| `contestant_to_original_sankey.html` | Sankey diagram: who covers whom (contestants to original artists). |
| `covers_per_edition.html` | Bar chart: number of cover performances per edition. |
| `covers_with_year_lag.csv` | Cover-level data with year lag (festival year minus original song year). |
| `year_lag_by_edition.html` | Boxplot of year lag by edition (how far back contestants look). |
| `original_artists_per_performer.csv` | Per performer (all editions): number of distinct original artists covered, total cover count. |
| `original_artists_per_performer.html` | Bar chart: top performers by number of distinct originals. |
| `original_artists_per_performer_per_year.csv` | Same metrics by (year, contestant). |
| `performer_originality.csv` | Per performer: mean "uniqueness" of their choices (1 / global cover count of that original). Higher = rarer originals. |
| `canon_metrics_over_time.csv` | Per edition: mean distinct originals per performer, mean originality. |
| `canon_metrics_over_time.html` | Line chart of these two metrics over editions. |
| `canon_original_gender_share_per_year.csv` | Share of covers by original-artist gender per year (if gender data exists). |
| `canon_original_gender_share_over_time.html` | Stacked area of that share over editions. |
| `top_artists.csv` / `top_artists.html` | Most covered original artists (canon). |
| `top_songs.csv` / `top_songs.html` | Most covered songs (canon). |
| `decade_distribution.csv` / `decade_distribution.html` | Distribution of original-song decades. |
| `decade_mix_per_year.csv` / `decade_mix_over_time.html` | Share of covers by decade per edition. |

---

## 2. Questions this angle answers

- **How many original artists per performer?** Use `original_artists_per_performer.csv` or the per-year table.
- **How "original" are performers' choices?** Use `performer_originality.csv` (mean uniqueness).
- **Changes over time?** Use `canon_metrics_over_time.csv` and the corresponding chart.
- **Female vs male original artists?** Use `canon_original_gender_share_per_year.csv` and the area chart (when gender annotations exist).
- **Top artists and top songs?** Use `top_artists.csv` / `top_songs.csv` and the bar charts.
- **Most covered decades?** Use `decade_distribution.csv` and `decade_mix_per_year.csv`.
- **Who covers whom?** Use `contestant_to_original_counts.csv` and `contestant_to_original_sankey.html`.
- **Volume and time depth?** Use `covers_per_edition.html`, `covers_with_year_lag.csv`, and `year_lag_by_edition.html`.
