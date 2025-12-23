import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Stock Screener")
st.markdown("Professional stock analysis tool with real-time data")

# Sidebar
st.sidebar.header("Input Parameters")
tickers_input = st.sidebar.text_input(
    "Enter stock tickers (comma-separated)",
    value="AAPL, MSFT, GOOGL",
    help="e.g., AAPL, MSFT, GOOGL"
)

# Parse tickers
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

# Load data button
if st.sidebar.button("📊 Load Data", use_container_width=True):
    if not tickers:
        st.error("Please enter at least one ticker symbol")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_data = []
        
        for idx, ticker in enumerate(tickers):
            try:
                status_text.text(f"Loading {ticker}...")
                progress_bar.progress((idx + 1) / len(tickers))
                
                # Download data
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")
                info = stock.info
                
                if hist.empty:
                    st.warning(f"No data found for {ticker}")
                    continue
                
                # Calculate metrics
                current_price = hist['Close'].iloc[-1]
                prev_price = hist['Close'].iloc[0]
                price_change = ((current_price - prev_price) / prev_price) * 100
                
                # Basic metrics
                market_cap = info.get('marketCap', 0)
                pe_ratio = info.get('trailingPE', 'N/A')
                dividend_yield = info.get('dividendYield', 0)
                eps = info.get('trailingEps', 'N/A')
                
                # Technical metrics
                ma_200 = hist['Close'].tail(200).mean()
                price_to_ma = current_price / ma_200 if ma_200 > 0 else 0
                
                # Volatility
                returns = hist['Close'].pct_change()
                volatility = returns.std() * np.sqrt(252)
                
                # RSI calculation
                delta = hist['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss if loss.iloc[-1] != 0 else 0
                rsi = 100 - (100 / (1 + rs.iloc[-1])) if rs.iloc[-1] != 0 else 0
                
                data_dict = {
                    'Ticker': ticker,
                    'Price': f"${current_price:.2f}",
                    'Change %': f"{price_change:.2f}%",
                    'P/E Ratio': pe_ratio if isinstance(pe_ratio, str) else f"{pe_ratio:.2f}",
                    'Market Cap': f"${market_cap/1e9:.2f}B" if market_cap > 0 else "N/A",
                    'Dividend Yield': f"{dividend_yield*100:.2f}%" if dividend_yield else "0%",
                    'EPS': eps if isinstance(eps, str) else f"{eps:.2f}",
                    'Price/200MA': f"{price_to_ma:.2f}",
                    'Volatility': f"{volatility*100:.2f}%",
                    'RSI (14)': f"{rsi:.2f}",
                }
                
                all_data.append(data_dict)
                
            except Exception as e:
                st.warning(f"Error loading {ticker}: {str(e)}")
        
        status_text.empty()
        progress_bar.empty()
        
        if all_data:
            # Display results in tabs
            tab1, tab2 = st.tabs(["📋 Data Table", "📈 Analysis"])
            
            with tab1:
                df = pd.DataFrame(all_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Download CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"stock_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            
            with tab2:
                st.markdown("### Key Metrics Explained")
                st.markdown("""
                - **P/E Ratio**: Price-to-Earnings (lower = cheaper relative to earnings)
                - **Market Cap**: Total company value
                - **Dividend Yield**: Annual dividend as % of stock price
                - **Price/200MA**: Price vs 200-day moving average (support level)
                - **RSI**: Momentum indicator (>70 = overbought, <30 = oversold)
                - **Volatility**: Stock price fluctuation (%)
                """)
                
                st.info("💡 **Tip**: Refresh the page to load different tickers!")
        else:
            st.error("No data found for any of the entered tickers")

else:
    st.info("👈 Enter tickers in the sidebar and click 'Load Data' to get started!")
    st.markdown("""
    ### How to use:
    1. Enter stock ticker symbols (comma-separated)
    2. Click "📊 Load Data"
    3. View results in tabs
    4. Download CSV for analysis
    
    **Example tickers:**
    - AAPL (Apple)
    - MSFT (Microsoft)
    - GOOGL (Google)
    - TSLA (Tesla)
    - AMZN (Amazon)
    """)

