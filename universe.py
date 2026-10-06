from __future__ import annotations

import re

import FinanceDataReader as fdr
import pandas as pd


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
    return bool(re.search(r"우$|우B$|우C$|우\([A-Z0-9]+\)$", text) or re.search(r"\d우$", text))


def _clean_equity_frame(frame: pd.DataFrame) -> pd.DataFrame:
    bad_pattern = r"스팩|SPAC|리츠|REIT|ETF|ETN"
    out = frame[~frame["name"].str.contains(bad_pattern, case=False, regex=True, na=False)].copy()
    out = out[~out["name"].map(_looks_like_preferred)]
    return out


def load_krx_universe_frame(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Return current KOSPI/KOSDAQ common-stock listings with market labels."""
    listing = fdr.StockListing("KRX")
    if listing is None or listing.empty:
        raise RuntimeError("FinanceDataReader returned an empty KRX listing")

    code_col = next((c for c in ("Code", "Symbol", "Ticker") if c in listing.columns), None)
    name_col = next((c for c in ("Name", "Company") if c in listing.columns), None)
    market_col = next((c for c in ("Market", "MarketId") if c in listing.columns), None)
    if code_col is None or name_col is None:
        raise RuntimeError(f"Unexpected KRX listing columns: {list(listing.columns)}")

    market_values = listing[market_col].astype(str) if market_col else pd.Series("UNKNOWN", index=listing.index)
    frame = pd.DataFrame(
        {
            "ticker": listing[code_col].astype(str).str.zfill(6),
            "name": listing[name_col].astype(str),
            "market": market_values,
            "source": "current",
            "delisting_date": pd.NaT,
        }
    )
    frame = frame[frame["market"].isin(markets)]
    frame = _clean_equity_frame(frame)
    frame = frame.drop_duplicates(subset=["ticker"]).sort_values("ticker").reset_index(drop=True)

    if max_symbols is not None and max_symbols > 0:
        frame = frame.head(max_symbols).copy()
    return frame


def load_krx_research_universe(
    start: str,
    end: str,
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    include_delisted: bool = True,
    max_symbols: int | None = None,
) -> pd.DataFrame:
    """Current equities plus stocks delisted during the research period.

    Adding delisted names materially reduces survivorship bias versus using only
    today's listing. It is still not a perfect point-in-time constituent database.
    """
    current = load_krx_universe_frame(markets=markets, max_symbols=None)
    frames = [current]

    if include_delisted:
        try:
            listing = fdr.StockListing("KRX-DELISTING", start, end)
        except TypeError:
            listing = fdr.StockListing("KRX-DELISTING")

        if listing is not None and not listing.empty:
            code_col = next((c for c in ("Symbol", "Code", "Ticker") if c in listing.columns), None)
            name_col = next((c for c in ("Name", "Company") if c in listing.columns), None)
            market_col = next((c for c in ("Market", "MarketId") if c in listing.columns), None)
            delist_col = next((c for c in ("DelistingDate", "DelistDate") if c in listing.columns), None)
            if code_col and name_col:
                market_values = listing[market_col].astype(str) if market_col else pd.Series("UNKNOWN", index=listing.index)
                delisted = pd.DataFrame(
                    {
                        "ticker": listing[code_col].astype(str).str.zfill(6),
                        "name": listing[name_col].astype(str),
                        "market": market_values,
                        "source": "delisted",
                        "delisting_date": pd.to_datetime(listing[delist_col], errors="coerce") if delist_col else pd.NaT,
                    }
                )
                delisted = delisted[delisted["market"].isin(markets)]
                delisted = _clean_equity_frame(delisted)
                # Current listing wins on duplicate ticker codes.
                delisted = delisted[~delisted["ticker"].isin(set(current["ticker"]))]
                frames.append(delisted)

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates(subset=["ticker"], keep="first").sort_values(["source", "ticker"]).reset_index(drop=True)
    if max_symbols is not None and max_symbols > 0:
        frame = frame.head(max_symbols).copy()
    return frame


def load_krx_universe(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    max_symbols: int | None = None,
) -> dict[str, str]:
    frame = load_krx_universe_frame(markets=markets, max_symbols=max_symbols)
    return dict(zip(frame["ticker"], frame["name"]))
