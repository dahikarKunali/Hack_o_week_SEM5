# Smart Dataset Query API

## Problem Statement
Most REST API demos are plain to-do list CRUD apps. This project instead builds
an API around **data analysis** — upload any CSV dataset and query it through
REST endpoints without writing a single line of pandas code on the client side.

Use case: a lightweight backend that a frontend/dashboard could call to get
summary statistics, filtered subsets, or grouped aggregates from a dataset —
similar to what you'd want behind a market/health analytics dashboard.

## Tech Stack
- Python 3
- Flask (REST framework)
- Pandas (data handling)

## Setup

```bash
cd dataset-query-api
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Server runs at `http://127.0.0.1:5000`. A sample dataset
(`data/sample_market_data.csv` — Nifty50 vs Bitcoin daily prices) is
auto-loaded on startup so you can test immediately.

## Endpoints

| Method | Endpoint      | Description                                      |
|--------|---------------|---------------------------------------------------|
| GET    | `/health`     | Check API + dataset status                        |
| POST   | `/upload`     | Upload a new CSV (multipart form field `file`)     |
| GET    | `/data`       | Paginated raw rows (`?page=1&limit=10`)            |
| GET    | `/columns`    | List columns and their data types                  |
| GET    | `/stats`      | Mean/median/min/max/std for all or one column       |
| GET    | `/filter`     | Filter rows by exact value or numeric range         |
| GET    | `/aggregate`  | Group by a column and aggregate another             |

## Example Requests

```bash
# Health check
curl http://127.0.0.1:5000/health

# Get first 5 rows
curl "http://127.0.0.1:5000/data?page=1&limit=5"

# Stats for the whole dataset
curl http://127.0.0.1:5000/stats

# Stats for one column
curl "http://127.0.0.1:5000/stats?column=price"

# Filter: only Bitcoin rows
curl "http://127.0.0.1:5000/filter?column=asset&value=Bitcoin"

# Filter: price between 42000 and 45000
curl "http://127.0.0.1:5000/filter?column=price&min=42000&max=45000"

# Aggregate: average price per asset
curl "http://127.0.0.1:5000/aggregate?group_by=asset&target=price&agg=mean"

# Upload your own CSV
curl -F "file=@your_dataset.csv" http://127.0.0.1:5000/upload
```

## Whiteboard Challenge Notes (3 hrs)
1. (30 min) Set up Flask boilerplate + `/health` route — prove server runs.
2. (45 min) Build `/upload` and `/data` — load and view CSV.
3. (45 min) Build `/stats` and `/filter` — core query logic.
4. (30 min) Build `/aggregate` — group-by feature (this is the "wow" endpoint).
5. (30 min) Test everything with curl/Postman, write README, polish.
