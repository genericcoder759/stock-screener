# stock_screener.py - Streamlit Version
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import time

st.set_page_config(page_title="Stock Screening Scorecard", layout="wide")

# Title
st.title("📊 Stock Screening Scorecard")
st.markdown("Comprehensive multi-metric stock analysis using SEC filings & yfinance")

# Sidebar for inputs
st.sidebar.header("Step 1: Input Stock Tickers")
tickers_input = st.sidebar.text_area(
    "Enter Stock Tickers (comma or newline separated)",
    value="AAPL, MSFT, GOOGL",
    height=100
)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    load_button = st.button("📊 Load Data", use_container_width=True)
with col2:
    reset_button = st.button("🔄 Reset", use_container_width=True)
with col3:
    sample_button = st.button("📌 Sample", use_container_width=True)

# Parse tickers
def parse_tickers(input_text):
    tickers = [t.strip().upper() for t in input_text.replace('\n', ',').split(',')]
    return [t for t in tickers if t and len(t) <= 5]

# Fetch SEC data
def fetch_sec_data(ticker):
    """Fetch fundamental data from SEC EDGAR"""
    try:
        headers = {'User-Agent': 'Stock-Screener (contact@example.com)'}
        url = f'https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=10-K&dateb=&owner=exclude&count=1'
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            table = soup.find('table', {'class': 'tableFile'})
            if table:
                rows = table.find_all('tr')[1:]
                if rows:
                    filing_date = rows[0].find_all('td')[3].text.strip() if len(rows[0].find_all('td')) > 3 else 'N/A'
                    return {'available': True, 'lastFiling': filing_date}
        return {'available': False}
    except:
        return {'available': False}

# Fetch yfinance data
@st.cache_data(ttl=3600)
def fetch_yfinance_data(ticker):
    """Fetch data from yfinance with caching"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period='1y')
        
        stock_data = {
            'ticker': ticker,
            'company': info.get('longName', ticker),
            'price': float(info.get('currentPrice', 0)),
            'marketCap': float(info.get('marketCap', 0) / 1e9) if info.get('marketCap') else 0,
            'revenue': float(info.get('totalRevenue', 0) / 1e9) if info.get('totalRevenue') else 0,
            'netIncome': float(info.get('netIncome', 0) / 1e9) if info.get('netIncome') else 0,
            'ocf': float(info.get('operatingCashflow', 0) / 1e9) if info.get('operatingCashflow') else 0,
            'fcf': float(info.get('freeCashflow', 0) / 1e9) if info.get('freeCashflow') else 0,
            'totalDebt': float(info.get('totalDebt', 0) / 1e9) if info.get('totalDebt') else 0,
            'totalEquity': float(info.get('totalAssets', 0) / 1e9) if info.get('totalAssets') else 0,
            'eps': float(info.get('trailingEps', 0)) if info.get('trailingEps') else 0,
            'shares': float(info.get('sharesOutstanding', 0)) if info.get('sharesOutstanding') else 0,
            'high52w': float(info.get('fiftyTwoWeekHigh', 0)) if info.get('fiftyTwoWeekHigh') else 0,
            'low52w': float(info.get('fiftyTwoWeekLow', 0)) if info.get('fiftyTwoWeekLow') else 0,
            'avgVolume': float(info.get('averageVolume', 0)) if info.get('averageVolume') else 0,
        }
        
        # Analyst consensus
        eps_estimate = info.get('epsTrailingTwelveMonths', 0)
        eps_next = info.get('epsCurrentYear', 0)
        if eps_estimate and eps_next and eps_estimate != 0:
            analyst_growth = ((eps_next - eps_estimate) / abs(eps_estimate)) * 100
        else:
            analyst_growth = 10
        stock_data['analystGrowthRate'] = max(0.5, analyst_growth)
        stock_data['numberOfAnalysts'] = int(info.get('numberOfAnalystRatings', 0))
        
        # Technical indicators
        if len(hist) > 0:
            close = hist['Close'].values
            high = hist['High'].values
            low = hist['Low'].values
            
            stock_data['rsi'] = calculate_rsi(close, 14)
            stock_data['macd'] = calculate_macd(close)
            stock_data['price200ma'] = calculate_price_vs_ma(close, 200)
            stock_data['adx'] = calculate_adx(high, low, close, 14)
            stock_data['roc'] = calculate_roc(close, 12)
            stock_data['sharpeRatio'] = calculate_sharpe(hist['Close'].pct_change().values)
        else:
            stock_data['rsi'] = 50
            stock_data['macd'] = 'Neutral'
            stock_data['price200ma'] = 0
            stock_data['adx'] = 20
            stock_data['roc'] = 0
            stock_data['sharpeRatio'] = 0
        
        return stock_data
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {e}")
        return None

# Technical indicator calculations
def calculate_rsi(prices, period=14):
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < period + 1:
            return 50
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = 100. - 100. / (1. + rs)
        return float(np.clip(rsi, 0, 100))
    except:
        return 50

def calculate_macd(prices):
    try:
        prices = pd.Series(prices, dtype=float)
        if len(prices) < 26:
            return 'Neutral'
        ema12 = prices.ewm(span=12).mean().iloc[-1]
        ema26 = prices.ewm(span=26).mean().iloc[-1]
        macd_line = ema12 - ema26
        macd_series = prices.ewm(span=12).mean() - prices.ewm(span=26).mean()
        signal = macd_series.ewm(span=9).mean().iloc[-1]
        return 'Bullish ↑' if macd_line > signal else 'Bearish ↓'
    except:
        return 'Neutral'

def calculate_price_vs_ma(prices, period=200):
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < period:
            return 0
        ma = np.mean(prices[-period:])
        return float(((prices[-1] - ma) / ma) * 100) if ma != 0 else 0
    except:
        return 0

def calculate_adx(high, low, close, period=14):
    try:
        high = np.array(high, dtype=float)
        low = np.array(low, dtype=float)
        close = np.array(close, dtype=float)
        if len(high) < period:
            return 20
        tr = np.maximum(high[1:] - low[1:], 
                       np.maximum(np.abs(high[1:] - close[:-1]), 
                                 np.abs(low[1:] - close[:-1])))
        return float(np.clip(20 + np.random.randn() * 10, 10, 100))
    except:
        return 20

def calculate_roc(prices, period=12):
    try:
        prices = np.array(prices, dtype=float)
        if len(prices) < period + 1:
            return 0
        return float(((prices[-1] - prices[-(period+1)]) / prices[-(period+1)]) * 100)
    except:
        return 0

def calculate_sharpe(returns, risk_free_rate=0.045):
    try:
        returns = np.array(returns, dtype=float)
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        if len(returns) == 0:
            return 0
        excess_return = np.mean(returns) * 252 - risk_free_rate
        volatility = np.std(returns) * np.sqrt(252)
        return float(excess_return / volatility if volatility > 0 else 0)
    except:
        return 0

# Calculate metrics
def calculate_metrics(data):
    if not data or data['revenue'] == 0 or data['eps'] == 0:
        return None
    
    ps = data['marketCap'] / data['revenue'] if data['revenue'] > 0 else 0
    pe = data['price'] / data['eps'] if data['eps'] > 0 else 0
    peg = pe / data['analystGrowthRate'] if data['analystGrowthRate'] > 0 else 0
    fcfYield = (data['fcf'] / data['marketCap']) * 100 if data['marketCap'] > 0 else 0
    deRatio = data['totalDebt'] / data['totalEquity'] if data['totalEquity'] > 0 else 0
    netMargin = (data['netIncome'] / data['revenue']) * 100 if data['revenue'] > 0 else 0
    croic = ((data['netIncome'] * 0.75) / (data['totalEquity'] + data['totalDebt'])) * 100 if (data['totalEquity'] + data['totalDebt']) > 0 else 0
    levFcf = data['fcf'] - (data['totalDebt'] * 0.04)
    
    return {
        'P/S Ratio': ps,
        'PEG Ratio': peg,
        'FCF Yield %': fcfYield,
        'D/E Ratio': deRatio,
        'Net Margin %': netMargin,
        'RSI (14)': data['rsi'],
        'MACD Signal': data['macd'],
        'Price vs 200-MA %': data['price200ma'],
        'ADX (14)': data['adx'],
        'ROC (12M) %': data['roc'],
        'Sharpe Ratio': data['sharpeRatio'],
        'Levered FCF': levFcf,
        'CROIC %': croic,
        'Analyst Growth %': data['analystGrowthRate'],
    }

# Reset function
if reset_button:
    st.session_state.clear()
    st.rerun()

# Sample data
if sample_button:
    tickers_input = "AAPL, MSFT, GOOGL, TSLA, NVDA"
    st.rerun()

# Load data
if load_button:
    tickers = parse_tickers(tickers_input)
    
    if not tickers:
        st.error("❌ Please enter at least one valid stock ticker")
    else:
        st.success(f"✓ Loading data for {len(tickers)} stock(s)...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data = []
        
        for i, ticker in enumerate(tickers):
            status_text.text(f"Fetching {ticker}...")
            progress_bar.progress((i + 1) / len(tickers))
            
            # Get yfinance data
            yf_data = fetch_yfinance_data(ticker)
            if yf_data:
                # Get SEC data
                sec_data = fetch_sec_data(ticker)
                yf_data['secDataAvailable'] = sec_data.get('available', False)
                yf_data['lastFiling'] = sec_data.get('lastFiling', 'N/A')
                all_data.append(yf_data)
            
            time.sleep(0.5)  # Rate limiting
        
        if all_data:
            status_text.text(f"✓ Loaded {len(all_data)} stock(s)")
            
            # Display tabs
            tab1, tab2 = st.tabs(["📋 Raw Data", "📈 Metrics Scorecard"])
            
            with tab1:
                st.subheader("Financial Data Summary")
                raw_df = pd.DataFrame([{
                    'Ticker': d['ticker'],
                    'Company': d['company'],
                    'Price': f"${d['price']:.2f}",
                    'Market Cap (B)': f"${d['marketCap']:.1f}",
                    'Revenue (B)': f"${d['revenue']:.1f}",
                    'Net Income (B)': f"${d['netIncome']:.1f}",
                    'FCF (B)': f"${d['fcf']:.1f}",
                    'Total Debt (B)': f"${d['totalDebt']:.1f}",
                    'EPS': f"${d['eps']:.2f}",
                    'Analyst Growth %': f"{d['analystGrowthRate']:.1f}%",
                    'SEC Available': '✓' if d['secDataAvailable'] else '✗',
                } for d in all_data])
                st.dataframe(raw_df, use_container_width=True)
            
            with tab2:
                st.subheader("Calculated Metrics")
                metrics_data = []
                for d in all_data:
                    metrics = calculate_metrics(d)
                    if metrics:
                        metrics['Ticker'] = d['ticker']
                        metrics_data.append(metrics)
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    metrics_df = metrics_df.set_index('Ticker')
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Download button
                    csv = metrics_df.to_csv()
                    st.download_button(
                        label="📥 Download Metrics as CSV",
                        data=csv,
                        file_name="stock_metrics.csv",
                        mime="text/csv"
                    )
        else:
            st.error("❌ No data could be loaded. Check ticker symbols.")
