"""Download a pinned KRX-derived daily amount and raw-price snapshot for research."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import pandas as pd
import requests


def collect_liquidity(universe_path="research/kospi-long-results/universe.csv", cache_dir="data_kospi_amount"):
    # Initialize pandas' Arrow extensions once before worker threads read Parquet.
    from pandas.io.parquet import get_engine
    get_engine("pyarrow")
    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True, parents=True)
    meta_path = cache / "source.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        r = requests.get("https://api.github.com/repos/FinanceData/marcap/commits/master", timeout=30)
        r.raise_for_status()
        meta = {"repository": "FinanceData/marcap", "commit": r.json()["sha"],
                "url": "https://github.com/FinanceData/marcap"}
        meta_path.write_text(json.dumps(meta, indent=2))
    u = pd.read_csv(universe_path, dtype={"ticker": str})
    codes = set(u.loc[u.market.eq("KOSPI"), "ticker"])
    def worker(year):
        path = cache / f"{year}.parquet"
        summary = cache / f"{year}_coverage.json"
        if path.exists() and summary.exists():
            return json.loads(summary.read_text())
        url = f"https://raw.githubusercontent.com/FinanceData/marcap/{meta['commit']}/data/marcap-{year}.parquet"
        for attempt in range(3):
            try:
                r = requests.get(url, timeout=90)
                r.raise_for_status()
                import io
                x = pd.read_parquet(io.BytesIO(r.content))
                if "Date" not in x.columns:
                    x = x.reset_index()
                x["Date"] = pd.to_datetime(x.Date)
                x["Code"] = x.Code.astype(str).str.zfill(6)
                kospi = x[x.Market.eq("KOSPI")]
                current = x[x.Code.isin(codes)].copy()
                current[["Date", "Code", "Close", "Volume", "Amount", "Market"]].to_parquet(path, index=False)
                info = {"year": year, "first": str(x.Date.min().date()), "last": str(x.Date.max().date()),
                        "historical_kospi_tickers": kospi.Code.nunique(),
                        "survivor_kospi_tickers": current.loc[current.Market.eq("KOSPI"), "Code"].nunique(),
                        "historical_kospi_rows": len(kospi),
                        "survivor_kospi_rows": int(current.Market.eq("KOSPI").sum()), "source_url": url}
                summary.write_text(json.dumps(info, indent=2))
                return info
            except Exception:
                if attempt == 2:
                    raise
    rows = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(worker, y) for y in range(1995, 2027)]
        for f in as_completed(futures):
            rows.append(f.result())
            print(f"Actual amount years {len(rows)}/32: {rows[-1]['year']}", flush=True)
            pd.DataFrame(rows).sort_values("year").to_csv(cache / "coverage.csv", index=False)


if __name__ == "__main__":
    collect_liquidity()
