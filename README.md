# Sanremo cover night analysis

Data journalism project: network analysis and story angles for the Sanremo Music Festival cover night (serata cover).

## Data

- `storico_serata_cover.csv` – cover performances (year, contestant, song, original artist, guest)
- `vincitori_serata_cover.csv` – winner per edition
- `medley.csv` – song-level detail for each medley performance

## Setup

```bash
pip install -r requirements.txt
```

## Run analysis

From the project root:

```bash
python -m src.run_analysis
```

Outputs (charts and summary) are written to `outputs/`.

## Project structure

- `src/` – load, clean, viz, and `run_analysis.py` (entry point)
- `outputs/` – Plotly HTML figures and `summary.txt`
