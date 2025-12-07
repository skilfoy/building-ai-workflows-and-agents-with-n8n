# Market & News Intelligence Dashboard

This lightweight FastAPI web app tracks major stock exchanges, global business news, and trending themes to generate
research-only positioning notes. Data refreshes automatically every 30 minutes and can also be triggered on demand.

## Features
- Snapshots for S&P 500, NASDAQ, Dow Jones, FTSE 100, Nikkei 225, and Hang Seng via Yahoo Finance.
- Global business headlines from Reuters, BBC, and Financial Times RSS feeds.
- Trend extraction from recurring headline keywords.
- Simple rule-based stance and action guidance based on moving averages and 30-day momentum.
- Manual refresh endpoint plus scheduled background updates.

## Getting started
1. **Install dependencies**
   ```bash
   pip install -r "project 5/stock-dashboard/requirements.txt"
   ```

2. **Run the app**
   ```bash
   uvicorn main:app --reload --app-dir "project 5/stock-dashboard" --host 0.0.0.0 --port 8000
   ```

3. **Open the dashboard**
   Visit `http://localhost:8000` in your browser.

4. **Manual refresh API**
   ```bash
   curl -X POST http://localhost:8000/api/refresh
   ```

## Notes
- This dashboard is for internal research and informational purposes only; it is not investment advice.
- The default scheduler refresh interval is 30 minutes. Adjust the `scheduler.add_job` interval in `main.py` if needed.
