from collections import Counter
from datetime import datetime, timezone
from threading import Lock
from typing import Dict, List, Tuple

import feedparser
import pandas as pd
import uvicorn
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse


app = FastAPI(title="Market & News Intelligence Dashboard", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXCHANGES = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "FTSE 100": "^FTSE",
    "Nikkei 225": "^N225",
    "Hang Seng": "^HSI",
}

NEWS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://www.ft.com/rss/home/asia",
]

CACHE_LOCK = Lock()
DATA_CACHE: Dict[str, object] = {
    "markets": [],
    "news": [],
    "trends": {},
    "recommendations": [],
    "last_updated": None,
}

scheduler = BackgroundScheduler(timezone="UTC")


def moving_averages(prices: pd.Series) -> Tuple[float, float]:
    window_50 = prices.tail(50).mean()
    window_200 = prices.tail(200).mean()
    return float(window_50), float(window_200)


def fetch_market_metrics() -> List[Dict[str, object]]:
    metrics: List[Dict[str, object]] = []
    for exchange, symbol in EXCHANGES.items():
        history = yf.Ticker(symbol).history(period="1y", interval="1d")
        if history.empty:
            continue

        close_prices = history["Close"].dropna()
        current_price = float(close_prices.iloc[-1])
        previous_price = float(close_prices.iloc[-2]) if len(close_prices) > 1 else current_price
        change = current_price - previous_price
        change_pct = (change / previous_price) * 100 if previous_price else 0.0

        ma_50, ma_200 = moving_averages(close_prices)
        momentum = close_prices.pct_change().tail(30).mean() * 100

        metrics.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "price": round(current_price, 2),
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
                "ma_50": round(ma_50, 2),
                "ma_200": round(ma_200, 2),
                "momentum_30d": round(float(momentum), 2),
                "last_close": close_prices.index[-1].strftime("%Y-%m-%d"),
            }
        )
    return metrics


def fetch_news() -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for feed in NEWS_FEEDS:
        parsed = feedparser.parse(feed)
        for entry in parsed.entries[:10]:
            published = (
                datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).isoformat()
                if hasattr(entry, "published_parsed")
                else None
            )
            items.append(
                {
                    "title": entry.title,
                    "link": entry.link,
                    "source": parsed.feed.get("title", "Unknown"),
                    "published": published,
                }
            )
    return sorted(items, key=lambda x: x.get("published") or "", reverse=True)[:25]


def extract_trending_terms(news_items: List[Dict[str, str]]) -> List[Tuple[str, int]]:
    stop_words = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "will",
        "says",
        "says",
        "into",
        "over",
        "after",
        "into",
        "amid",
    }
    counter: Counter[str] = Counter()
    for item in news_items:
        words = [word.strip(".,:;!?").lower() for word in item["title"].split()]
        for word in words:
            if len(word) <= 3 or word in stop_words:
                continue
            counter[word] += 1
    return counter.most_common(10)


def build_recommendations(markets: List[Dict[str, object]], keywords: List[Tuple[str, int]]) -> List[Dict[str, str]]:
    recommendations: List[Dict[str, str]] = []
    for market in markets:
        price = market["price"]
        ma_50 = market["ma_50"]
        ma_200 = market["ma_200"]
        momentum = market["momentum_30d"]

        if price > ma_50 > ma_200 and momentum > 0:
            stance = "Uptrend intact"
            action = "Consider staggered entries on pullbacks while momentum holds."
        elif price > ma_50 and momentum >= 0:
            stance = "Short-term strength"
            action = "Monitor for sustained closes above the 50-day before sizing in."
        elif price < ma_200 and momentum < 0:
            stance = "Under pressure"
            action = "Keep exposure light; wait for basing above the 50-day average."
        else:
            stance = "Range-bound"
            action = "Favor smaller allocations and tighter risk controls until trend resolves."

        recommendations.append(
            {
                "market": market["exchange"],
                "symbol": market["symbol"],
                "stance": stance,
                "action": action,
                "note": f"30d momentum: {momentum:.2f}%, price vs 50d: {price - ma_50:.2f}",
            }
        )

    if keywords:
        keyword_hint = ", ".join([word for word, _ in keywords[:3]])
        recommendations.append(
            {
                "market": "Cross-market themes",
                "symbol": "Global",
                "stance": "News-sensitive",
                "action": "Track position sizing around headline risk and liquidity windows.",
                "note": f"Top recurring terms: {keyword_hint}",
            }
        )

    return recommendations


def refresh_data() -> Dict[str, object]:
    markets = fetch_market_metrics()
    news_items = fetch_news()
    keywords = extract_trending_terms(news_items)
    recommendations = build_recommendations(markets, keywords)

    with CACHE_LOCK:
        DATA_CACHE.update(
            {
                "markets": markets,
                "news": news_items,
                "trends": {
                    "keywords": keywords,
                    "count": len(news_items),
                },
                "recommendations": recommendations,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }
        )
    return DATA_CACHE


@app.get("/api/summary")
def get_summary():
    if not DATA_CACHE["markets"]:
        refresh_data()
    return DATA_CACHE


@app.post("/api/refresh")
def manual_refresh():
    try:
        return refresh_data()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
def serve_index(request: Request):
    index_path = request.scope.get("root_path", "") + "/static/index.html"
    try:
        with open("project 5/stock-dashboard/static/index.html", "r", encoding="utf-8") as index_file:
            return HTMLResponse(index_file.read())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Dashboard not found") from exc


@app.on_event("startup")
def start_scheduler():
    refresh_data()
    if not scheduler.running:
        scheduler.add_job(refresh_data, "interval", minutes=30, id="refresh-job", replace_existing=True)
        scheduler.start()


@app.on_event("shutdown")
def shutdown_scheduler():
    if scheduler.running:
        scheduler.shutdown()


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
