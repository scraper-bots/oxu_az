import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test direct stock data extraction
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

print("Testing yfinance data extraction...")
print(f"Current time: {datetime.now()}")

for symbol in symbols:
    try:
        print(f"\nTesting {symbol}...")
        ticker = yf.Ticker(symbol)
        
        # Try different periods
        periods = ['1mo', '3mo', '6mo', '1y', '2y']
        
        for period in periods:
            try:
                hist = ticker.history(period=period)
                if not hist.empty:
                    print(f"  {period}: SUCCESS - {len(hist)} records")
                    print(f"    Latest price: ${hist['Close'].iloc[-1]:.2f}")
                    print(f"    Date range: {hist.index[0].date()} to {hist.index[-1].date()}")
                    break
                else:
                    print(f"  {period}: EMPTY")
            except Exception as e:
                print(f"  {period}: ERROR - {e}")
        
        # Test info
        try:
            info = ticker.info
            print(f"  Company: {info.get('longName', 'N/A')}")
            print(f"  Market Cap: ${info.get('marketCap', 0):,}")
        except Exception as e:
            print(f"  Info ERROR: {e}")
            
    except Exception as e:
        print(f"ERROR with {symbol}: {e}")

print("\nTesting S&P 500 symbols fetch...")
try:
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)
    df = tables[0]
    symbols = df['Symbol'].tolist()
    print(f"Found {len(symbols)} S&P 500 symbols")
    print(f"First 10: {symbols[:10]}")
except Exception as e:
    print(f"ERROR fetching S&P 500: {e}")