import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Stock Screener")
st.markdown("Professional stock analysis with SEC filings integration")

# Sidebar
st.sidebar.header("Input Parameters")
tickers_input = st.sidebar.text_input(
    "Enter stock tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL",
    help="e.g., AAPL, MSFT, GOOGL"
)

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

def check_sec_filing(ticker):
    """Check if company has recent 10-K filing"""
    try:
        # SEC EDGAR API endpoint
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company={ticker}&type=10-K&dateb=&owner=exclude&count=1&output=json"
        response = requests.get(url, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if 'filings' in data and data['filings'].get('filings'):
                filing = data['filings']['filings'][0]
                filing_date = filing.get('filingDate', 'Unknown')
                return f"✅ 10-K filed: {filing_date}"
            else:
                return "❌ No recent 10-K found"
        else:
            return "⚠️ Could not verify"
    except:
        return "⚠️ SEC check unavailable"

def calculate_metrics(ticker):
    """Calculate all metrics for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        info = stock.info
        
        if hist.empty:
            return None
        
        # Price metrics
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[0]
        price_change = ((current_price - prev_price) / prev_price) * 100
        
        # Valuation metrics
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', None)
        ps_ratio = info.get('priceToSalesTrailing12Months', None)
        pb_ratio = info.get('priceToBook', None)
        eps = info.get('trailingEps', None)
        dividend_yield = info.get('dividendYield', 0) or 0
        
        # Growth metrics
        revenue_growth = info.get('revenueGrowth', None)
        earnings_growth = info.get('earningsGrowth', None)
        
        # Debt metrics
        debt_to_equity = info.get('debtToEquity', None)
        
        # Technical metrics
        ma_50 = hist['Close'].tail(50).mean()
        ma_200 = hist['Close'].tail(200).mean()
        price_to_ma_50 = current_price / ma_50 if ma_50 > 0 else 0
        price_to_ma_200 = current_price / ma_200 if ma_200 > 0 else 0
        
        # Volatility
        returns = hist['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # RSI calculation
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs.iloc[-1])) if rs.iloc[-1] != 0 else 0
        
        # MACD
        ema_12 = hist['Close'].ewm(span=12).mean()
        ema_26 = hist['Close'].ewm(span=26).mean()
        macd = ema_12 - ema_26
        macd_value = macd.iloc[-1]
        
        # SEC filing status
        sec_status = check_sec_filing(ticker)
        
        # Format output
        data_dict = {
            'Ticker': ticker,
            'Price': f"${current_price:.2f}",
            '1Y Change %': f"{price_change:.2f}%",
            'P/E Ratio': f"{pe_ratio:.2f}" if pe_ratio else "N/A",
            'P/S Ratio': f"{ps_ratio:.2f}" if ps_ratio else "N/A",
            'P/B Ratio': f"{pb_ratio:.2f}" if pb_ratio else "N/A",
            'EPS': f"{eps:.2f}" if eps else "N/A",
            'Market Cap': f"${market_cap/1e9:.2f}B" if market_cap > 0 else "N/A",
            'Dividend Yield': f"{dividend_yield*100:.2f}%" if dividend_yield else "0%",
            'D/E Ratio': f"{debt_to_equity:.2f}" if debt_to_equity else "N/A",
            'Revenue Growth': f"{revenue_growth*100:.2f}%" if revenue_growth else "N/A",
            'Earnings Growth': f"{earnings_growth*100:.2f}%" if earnings_growth else "N/A",
            'Price/50MA': f"{price_to_ma_50:.2f}",
            'Price/200MA': f"{price_to_ma_200:.2f}",
            'Volatility': f"{volatility*100:.2f}%",
            'RSI (14)': f"{rsi:.2f}",
            'MACD': f"{macd_value:.4f}",
            'SEC Filing': sec_status,
        }
        
        return data_dict
    except Exception as e:
        st.warning(f"Error processing {ticker}: {str(e)}")
        return None

# Load data button
if st.sidebar.button("📊 Load Data", use_container_width=True):
    if not tickers:
        st.error("Please enter at least one ticker symbol")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data = []
        
        for idx, ticker in enumerate(tickers):
            status_text.text(f"Loading {ticker}...")
            progress_bar.progress((idx + 1) / len(tickers))
            
            result = calculate_metrics(ticker)
            if result:
                all_data.append(result)
        
        status_text.empty()
        progress_bar.empty()
        
        if all_data:
            # Display results in tabs
            tab1, tab2, tab3 = st.tabs(["📋 Financial Data", "📈 Technical Analysis", "🏛️ SEC Filings"])
            
            with tab1:
                st.subheader("Financial Metrics")
                df_financial = pd.DataFrame([{
                    'Ticker': d['Ticker'],
                    'Price': d['Price'],
                    '1Y Change': d['1Y Change %'],
                    'P/E': d['P/E Ratio'],
                    'P/S': d['P/S Ratio'],
                    'P/B': d['P/B Ratio'],
                    'EPS': d['EPS'],
                    'Market Cap': d['Market Cap'],
                    'Div Yield': d['Dividend Yield'],
                    'D/E Ratio': d['D/E Ratio'],
                } for d in all_data])
                st.dataframe(df_financial, use_container_width=True, hide_index=True)
            
            with tab2:
                st.subheader("Technical Indicators")
                df_technical = pd.DataFrame([{
                    'Ticker': d['Ticker'],
                    'Price/50MA': d['Price/50MA'],
                    'Price/200MA': d['Price/200MA'],
                    'Volatility': d['Volatility'],
                    'RSI (14)': d['RSI (14)'],
                    'MACD': d['MACD'],
                } for d in all_data])
                st.dataframe(df_technical, use_container_width=True, hide_index=True)
                
                st.markdown("""
                ### Technical Indicators Explained:
                - **Price/50MA & Price/200MA**: Price vs moving averages (support/resistance)
                - **RSI**: Momentum (>70 = overbought, <30 = oversold)
                - **MACD**: Trend indicator (positive = bullish)
                - **Volatility**: Price fluctuation
                """)
            
            with tab3:
                st.subheader("SEC EDGAR Filing Status")
                df_sec = pd.DataFrame([{
                    'Ticker': d['Ticker'],
                    'Revenue Growth': d['Revenue Growth'],
                    'Earnings Growth': d['Earnings Growth'],
                    'Filing Status': d['SEC Filing'],
                } for d in all_data])
                st.dataframe(df_sec, use_container_width=True, hide_index=True)
                
                st.markdown("""
                ### What This Means:
                - ✅ **10-K Filed**: Company has filed annual report
                - ❌ **No Recent 10-K**: Check company website
                - ⚠️ **Unable to Verify**: SEC server may be slow
                
                [View SEC EDGAR Database](https://www.sec.gov/edgar)
                """)
            
            # Download full data as CSV
            st.divider()
            full_df = pd.DataFrame(all_data)
            csv = full_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Analysis (CSV)",
                data=csv,
                file_name=f"stock_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.error("No data found for any of the entered tickers")

else:
    st.info("👈 Enter tickers in the sidebar and click 'Load Data' to get started!")
    st.markdown("""
    ### Features:
    ✅ Real-time stock prices
    ✅ 15+ financial metrics
    ✅ Technical indicators (RSI, MACD, Moving Averages)
    ✅ SEC 10-K filing verification
    ✅ CSV export
    
    ### Example Tickers:
    - AAPL (Apple)
    - MSFT (Microsoft)
    - GOOGL (Google)
    - TSLA (Tesla)
    - AMZN (Amazon)
    """)
