"""
Smart Dataset Query API
------------------------
A REST API that lets you upload a CSV dataset and query it on the fly.
Built with Flask + Pandas.

Every GET endpoint renders as a clean, readable HTML page in the browser
(for demos), while still being a real REST API under the hood.
"""

from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

STATE = {"df": None, "filename": None}
DEFAULT_DATA_PATH = os.path.join("data", "sample_market_data.csv")

PURPLE = "#4b2e83"


def load_default_dataset():
    if os.path.exists(DEFAULT_DATA_PATH):
        STATE["df"] = pd.read_csv(DEFAULT_DATA_PATH)
        STATE["filename"] = "sample_market_data.csv"


def get_df():
    return STATE["df"]


PAGE_STYLE = f"""
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; color: #222; }}
  h1 {{ color: {PURPLE}; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-top: 0; margin-bottom: 24px; }}
  a.back {{ color: {PURPLE}; text-decoration: none; font-size: 14px; }}
  a.back:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
  th {{ background: {PURPLE}; color: white; text-align: left; padding: 8px 12px; font-size: 14px; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 14px; }}
  tr:nth-child(even) {{ background: #f7f5fa; }}
  .kv {{ font-size: 15px; margin: 6px 0; }}
  .kv b {{ color: {PURPLE}; }}
  .badge {{ display: inline-block; background: #e8f5e9; color: #2e7d32; padding: 2px 10px;
            border-radius: 12px; font-size: 13px; font-weight: bold; }}
</style>
"""


def page(title, subtitle, body_html):
    return f"""
    <html>
      <head><title>{title}</title>{PAGE_STYLE}</head>
      <body>
        <a class="back" href="/">&larr; Back to all endpoints</a>
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>
        {body_html}
      </body>
    </html>
    """


def render_table(records):
    if not records:
        return "<p>No rows to show.</p>"
    columns = list(records[0].keys())
    head = "".join(f"<th>{c}</th>" for c in columns)
    rows = ""
    for r in records:
        cells = "".join(f"<td>{r[c]}</td>" for c in columns)
        rows += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{rows}</table>"


def render_kv(d):
    return "".join(f'<div class="kv"><b>{k}:</b> {v}</div>' for k, v in d.items())


def render_stats_table(stats_dict):
    if not stats_dict:
        return "<p>No numeric columns.</p>"
    stat_names = list(next(iter(stats_dict.values())).keys())
    head = "<th>Column</th>" + "".join(f"<th>{s}</th>" for s in stat_names)
    rows = ""
    for col, stats in stats_dict.items():
        cells = f"<td><b>{col}</b></td>" + "".join(
            f"<td>{round(stats[s], 2) if isinstance(stats[s], float) else stats[s]}</td>" for s in stat_names
        )
        rows += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{rows}</table>"


@app.route("/", methods=["GET"])
def home():
    links = [
        ("Check Server Status", "/health", "Is the server and dataset ready?"),
        ("View Sample Data", "/data?page=1&limit=5",
         "See the first 5 rows of the dataset"),
        ("View Column Info", "/columns", "List all column names and their types"),
        ("Full Statistics", "/stats", "Mean, median, min, max, std for every column"),
        ("Price Statistics Only", "/stats?column=price",
         "Stats for just the 'price' column"),
        ("Filter: Bitcoin Only", "/filter?column=asset&value=Bitcoin",
         "Show only rows where asset = Bitcoin"),
        ("Average Price by Asset", "/aggregate?group_by=asset&target=price&agg=mean",
         "Grouped average price per asset"),
    ]
    list_items = "".join(
        f'<li><a href="{url}"><b>{name}</b></a><br><span style="color:#666; font-size:14px;">{desc}</span></li>'
        for name, url, desc in links
    )
    return f"""
    <html>
      <head><title>Smart Dataset Query API</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 40px; }}
          h1 {{ color: {PURPLE}; }}
          li {{ margin-bottom: 18px; list-style: none; }}
          a {{ color: {PURPLE}; text-decoration: none; font-size: 17px; }}
          a:hover {{ text-decoration: underline; }}
          ul {{ padding-left: 0; }}
        </style>
      </head>
      <body>
        <h1>Smart Dataset Query API</h1>
        <p>Click any option below to try it:</p>
        <ul>{list_items}</ul>
        <p><i>Note: /upload requires a POST request (use curl/Postman, not a browser click).</i></p>
      </body>
    </html>
    """


@app.route("/health", methods=["GET"])
def health():
    ok = STATE["df"] is not None
    body_html = render_kv({
        "Server": '<span class="badge">Running</span>',
        "Dataset loaded": "Yes" if ok else "No",
        "Filename": STATE["filename"] or "—"
    })
    return page("Server Status", "Quick check that everything is ready.", body_html)


@app.route("/upload", methods=["POST"])
def upload_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request. Use form field 'file'."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Could not parse CSV: {str(e)}"}), 400
    STATE["df"] = df
    STATE["filename"] = file.filename
    return jsonify({"message": "Dataset uploaded successfully", "filename": file.filename,
                    "rows": len(df), "columns": list(df.columns)})


@app.route("/data", methods=["GET"])
def get_data():
    df = get_df()
    if df is None:
        return page("No Data", "Dataset not loaded.", "<p>Upload a CSV first.</p>")
    page_num = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 10))
    start = (page_num - 1) * limit
    end = start + limit
    subset = df.iloc[start:end]
    body_html = f"<p>Showing rows {start+1}-{min(end, len(df))} of {len(df)} total.</p>" + render_table(
        subset.to_dict(orient="records"))
    return page("Sample Data", "Raw rows from the dataset.", body_html)


@app.route("/columns", methods=["GET"])
def get_columns():
    df = get_df()
    if df is None:
        return page("No Data", "Dataset not loaded.", "<p>Upload a CSV first.</p>")
    records = [{"Column": c, "Type": str(t)} for c, t in df.dtypes.items()]
    return page("Column Info", "Every column in the dataset and its data type.", render_table(records))


@app.route("/stats", methods=["GET"])
def get_stats():
    df = get_df()
    if df is None:
        return page("No Data", "Dataset not loaded.", "<p>Upload a CSV first.</p>")
    column = request.args.get("column")
    numeric_df = df.select_dtypes(include="number")

    if column:
        if column not in df.columns or column not in numeric_df.columns:
            return page("Error", "Invalid column.", f"<p>'{column}' is not a valid numeric column.</p>")
        col = df[column]
        body_html = render_kv({
            "Count": int(col.count()),
            "Mean": round(float(col.mean()), 2),
            "Median": round(float(col.median()), 2),
            "Min": round(float(col.min()), 2),
            "Max": round(float(col.max()), 2),
            "Std Dev": round(float(col.std()), 2),
        })
        return page(f"Statistics: {column}", "Summary statistics for one column.", body_html)

    stats = {}
    for col in numeric_df.columns:
        stats[col] = {
            "count": int(numeric_df[col].count()), "mean": float(numeric_df[col].mean()),
            "median": float(numeric_df[col].median()), "min": float(numeric_df[col].min()),
            "max": float(numeric_df[col].max()), "std": float(numeric_df[col].std())
        }
    return page("Full Statistics", "Summary statistics for every numeric column.", render_stats_table(stats))


@app.route("/filter", methods=["GET"])
def filter_data():
    df = get_df()
    if df is None:
        return page("No Data", "Dataset not loaded.", "<p>Upload a CSV first.</p>")
    column = request.args.get("column")
    if not column or column not in df.columns:
        return page("Error", "Missing column.", "<p>Provide a valid 'column' query param.</p>")
    result = df
    value = request.args.get("value")
    min_val = request.args.get("min")
    max_val = request.args.get("max")
    if value is not None:
        result = result[result[column].astype(str) == value]
    if min_val is not None:
        result = result[result[column] >= float(min_val)]
    if max_val is not None:
        result = result[result[column] <= float(max_val)]
    body_html = f"<p><b>{len(result)}</b> rows matched.</p>" + \
        render_table(result.to_dict(orient="records"))
    return page("Filtered Results", f"Filtered by {column}.", body_html)


@app.route("/aggregate", methods=["GET"])
def aggregate_data():
    df = get_df()
    if df is None:
        return page("No Data", "Dataset not loaded.", "<p>Upload a CSV first.</p>")
    group_by = request.args.get("group_by")
    target = request.args.get("target")
    agg = request.args.get("agg", "mean")
    if not group_by or group_by not in df.columns:
        return page("Error", "Missing group_by.", "<p>Provide a valid 'group_by' column.</p>")
    if not target or target not in df.columns:
        return page("Error", "Missing target.", "<p>Provide a valid 'target' column.</p>")
    if agg not in ["mean", "sum", "min", "max", "count"]:
        return page("Error", "Invalid aggregation.", "<p>agg must be one of mean, sum, min, max, count.</p>")
    grouped = df.groupby(group_by)[target].agg(agg)
    records = [{group_by: k, f"{agg}({target})": round(v, 2) if isinstance(v, float) else v}
               for k, v in grouped.to_dict().items()]
    return page("Aggregated Results", f"{agg}({target}) grouped by {group_by}.", render_table(records))


if __name__ == "__main__":
    load_default_dataset()
    app.run(debug=True, port=5000)
