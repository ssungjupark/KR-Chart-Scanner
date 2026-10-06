from __future__ import annotations

import re

import FinanceDataReader as fdr
import pandas as pd


# Small fixed panel retained for quick smoke tests.
KOSPI_RESEARCH_PANEL: dict[str, str] = {
    "005930": "Samsung Electronics",
    "000660": "SK hynix",
    "373220": "LG Energy Solution",
    "207940": "Samsung Biologics",
    "005380": "Hyundai Motor",
    "012330": "Hyundai Mobis",
    "035420": "NAVER",
    "035720": "Kakao",
    "068270": "Celltrion",
    "105560": "KB Financial",
    "055550": "Shinhan Financial",
    "005490": "POSCO Holdings",
    "028260": "Samsung C&T",
    "034020": "Doosan Enerbility",
    "012450": "Hanwha Aerospace",
    "267260": "HD Hyundai Electric",
    "010120": "LS ELECTRIC",
    "103590": "Iljin Electric",
    "402340": "SK Square",
    "006400": "Samsung SDI",
}


def _looks_like_preferred(name: str) -> bool:
    text = str(name).strip()
    return bool(
        re.search(r"우$|우B$|우C$|우\([A-Z0-9]+\)$", text)
        or re.search(r"\d우$", text)
    )


def load_krx_universe_frame(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Return current KOSPI/KOSDAQ common-stock listings with market labels.

    The listing itself is current, so historical research using this universe
    still has survivorship bias. Signal-date liquidity is filtered separately.
    """
    listing = fdr.StockListing("KRX")
    if listing is None or listing.empty:
        raise RuntimeError("FinanceDataReader returned an empty KRX listing")

    code_col = next((c for c in ("Code", "Symbol", "Ticker") if c in listing.columns), None)
    name_col = next((c for c in ("Name", "Company") if c in listing.columns), None)
    market_col = next((c for c in ("Market", "MarketId") if c in listing.columns), None)
    if code_col is None or name_col is None:
        raise RuntimeError(f"Unexpected KRX listing columns: {list(listing.columns)}")

    frame = listing.copy()
    if market_col is not None:
        frame = frame[frame[market_col].astype(str).isin(markets)]
        market_values = frame[market_col].astype(str)
    else:
        market_values = pd.Series("UNKNOWN", index=frame.index)

    frame = pd.DataFrame(
        {
            "ticker": frame[code_col].astype(str).str.zfill(6),
            "name": frame[name_col].astype(str),
            "market": market_values,
        }
    )

    # Exclude obvious non-common-equity instruments by name. KRX StockListing
    # usually contains listed equities, but this keeps the research universe conservative.
    bad_pattern = r"스팩|SPAC|리츠|REIT|ETF|ETN"
    frame = frame[~frame["name"].str.contains(bad_pattern, case=False, regex=True, na=False)]
    frame = frame[~frame["name"].map(_looks_like_preferred)]
    frame = frame.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    if max_symbols is not None and max_symbols > 0:
        frame = frame.head(max_symbols).copy()

    return frame


def load_krx_universe(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    max_symbols: int | None = None,
) -> dict[str, str]:
    frame = load_krx_universe_frame(markets=markets, max_symbols=max_symbols)
    return dict(zip(frame["ticker"], frame["name"]))
