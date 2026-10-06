from __future__ import annotations

import re

import FinanceDataReader as fdr
import pandas as pd


KOSPI_RESEARCH_PANEL: dict[str, str] = {
    "005930": "Samsung Electronics", "000660": "SK hynix", "373220": "LG Energy Solution",
    "207940": "Samsung Biologics", "005380": "Hyundai Motor", "012330": "Hyundai Mobis",
    "035420": "NAVER", "035720": "Kakao", "068270": "Celltrion", "105560": "KB Financial",
    "055550": "Shinhan Financial", "005490": "POSCO Holdings", "028260": "Samsung C&T",
    "034020": "Doosan Enerbility", "012450": "Hanwha Aerospace", "267260": "HD Hyundai Electric",
    "010120": "LS ELECTRIC", "103590": "Iljin Electric", "402340": "SK Square", "006400": "Samsung SDI",
}


def _looks_like_preferred(name: str) -> bool:
    text = str(name).strip()
    return bool(re.search(r"우$|우B$|우C$|우\([A-Z0-9]+\)$", text) or re.search(r"\d우$", text))


def _normalize_market(value: object) -> str:
    text = str(value).strip().upper()
    if "KOSDAQ" in text:
        return "KOSDAQ"
    if "KOSPI" in text:
        return "KOSPI"
    return text


def _clean_equity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    bad_pattern = r"스팩|SPAC|리츠|REIT|ETF|ETN"
    out = frame[~frame["name"].str.contains(bad_pattern, case=False, regex=True, na=False)].copy()
    return out[~out["name"].map(_looks_like_preferred)]


def load_krx_universe_frame(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    max_symbols: int | None = None,
) -> pd.DataFrame:
    listing = fdr.StockListing("KRX")
    if listing is None or listing.empty:
        raise RuntimeError("FinanceDataReader returned an empty KRX listing")
    code_col = next((c for c in ("Code", "Symbol", "Ticker") if c in listing.columns), None)
    name_col = next((c for c in ("Name", "Company") if c in listing.columns), None)
    market_col = next((c for c in ("Market", "MarketId") if c in listing.columns), None)
    if code_col is None or name_col is None:
        raise RuntimeError(f"Unexpected KRX listing columns: {list(listing.columns)}")
    market_values = listing[market_col].map(_normalize_market) if market_col else pd.Series("UNKNOWN", index=listing.index)
    frame = pd.DataFrame({
        "ticker": listing[code_col].astype(str).str.zfill(6),
        "name": listing[name_col].astype(str),
        "market": market_values,
        "source": "current",
        "delisting_date": pd.NaT,
    })
    frame = _clean_equity_frame(frame[frame["market"].isin(markets)])
    frame = frame.drop_duplicates("ticker").sort_values("ticker").reset_index(drop=True)
    return frame.head(max_symbols).copy() if max_symbols and max_symbols > 0 else frame


def _prepare_delisted(listing: pd.DataFrame, start: str, end: str, markets: tuple[str, ...]) -> pd.DataFrame:
    if listing is None or listing.empty:
        return pd.DataFrame(columns=["ticker", "name", "market", "source", "delisting_date"])
    code_col = next((c for c in ("Symbol", "Code", "Ticker") if c in listing.columns), None)
    name_col = next((c for c in ("Name", "Company") if c in listing.columns), None)
    market_col = next((c for c in ("Market", "MarketId") if c in listing.columns), None)
    delist_col = next((c for c in ("DelistingDate", "DelistDate") if c in listing.columns), None)
    if not code_col or not name_col:
        return pd.DataFrame(columns=["ticker", "name", "market", "source", "delisting_date"])
    dates = pd.to_datetime(listing[delist_col], errors="coerce") if delist_col else pd.Series(pd.NaT, index=listing.index)
    market_values = listing[market_col].map(_normalize_market) if market_col else pd.Series("UNKNOWN", index=listing.index)
    frame = pd.DataFrame({
        "ticker": listing[code_col].astype(str).str.zfill(6),
        "name": listing[name_col].astype(str),
        "market": market_values,
        "source": "delisted",
        "delisting_date": dates,
    })
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if frame["delisting_date"].notna().any():
        frame = frame[frame["delisting_date"].between(start_ts, end_ts)]
    frame = frame[frame["market"].isin(markets)]
    return _clean_equity_frame(frame)


def load_krx_research_universe(
    start: str,
    end: str,
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    include_delisted: bool = True,
    max_symbols: int | None = None,
) -> pd.DataFrame:
    current = load_krx_universe_frame(markets=markets)
    frames = [current]
    if include_delisted:
        try:
            ranged = fdr.StockListing("KRX-DELISTING", start, end)
        except Exception:
            ranged = pd.DataFrame()
        delisted = _prepare_delisted(ranged, start, end, markets)
        if delisted.empty:
            # Some FinanceDataReader/KRX combinations ignore or reject ranged listing arguments.
            # Full-list fallback is slower but keeps the research universe reproducible.
            full = fdr.StockListing("KRX-DELISTING")
            delisted = _prepare_delisted(full, start, end, markets)
        if not delisted.empty:
            delisted = delisted[~delisted["ticker"].isin(set(current["ticker"]))]
            frames.append(delisted)
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("ticker", keep="first")
    frame = frame.sort_values(["source", "ticker"]).reset_index(drop=True)
    return frame.head(max_symbols).copy() if max_symbols and max_symbols > 0 else frame


def load_krx_universe(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    max_symbols: int | None = None,
) -> dict[str, str]:
    frame = load_krx_universe_frame(markets=markets, max_symbols=max_symbols)
    return dict(zip(frame["ticker"], frame["name"]))
