import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class StockDataExtractor:
    def __init__(self):
        self.data = {}
    
    def get_stock_data(self, symbol, period="2y", interval="1d"):
        """
        Extract stock data for a given symbol
        period: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                print(f"No data found for {symbol}")
                return None
            
            # Get additional info
            info = ticker.info
            
            # Calculate technical indicators
            hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
            hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
            hist['RSI'] = self.calculate_rsi(hist['Close'])
            hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
            
            # Price change metrics
            hist['Price_Change'] = hist['Close'].pct_change()
            hist['Price_Change_5d'] = hist['Close'].pct_change(periods=5)
            hist['Price_Change_20d'] = hist['Close'].pct_change(periods=20)
            
            # Volatility
            hist['Volatility'] = hist['Price_Change'].rolling(window=20).std()
            
            return {
                'symbol': symbol,
                'data': hist,
                'info': info,
                'current_price': hist['Close'].iloc[-1],
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'sector': info.get('sector', 'Unknown')
            }
            
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_multiple_stocks(self, symbols, period="2y", delay=1):
        """Get data for multiple stocks with delay to avoid rate limiting"""
        results = []
        
        for symbol in symbols:
            print(f"Fetching data for {symbol}...")
            data = self.get_stock_data(symbol, period)
            if data:
                results.append(data)
            time.sleep(delay)  # Rate limiting
        
        return results
    
    def get_sp500_symbols(self):
        """Get S&P 500 symbols from Wikipedia"""
        try:
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            tables = pd.read_html(url)
            df = tables[0]
            return df['Symbol'].tolist()
        except Exception as e:
            print(f"Error fetching S&P 500 symbols: {e}")
            # Fallback to common stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'UNH']
    
    def save_data(self, data, filename):
        """Save extracted data to file"""
        processed_data = []
        for stock in data:
            processed_data.append({
                'symbol': stock['symbol'],
                'current_price': float(stock['current_price']),
                'market_cap': stock['market_cap'],
                'pe_ratio': stock['pe_ratio'],
                'sector': stock['sector'],
                'data_shape': stock['data'].shape
            })
        
        with open(filename, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Data saved to {filename}")

if __name__ == "__main__":
    extractor = StockDataExtractor()
    
    # Get S&P 500 symbols
    print("Fetching S&P 500 symbols...")
    symbols = extractor.get_sp500_symbols()
    print(f"Found {len(symbols)} symbols")
    
    # Extract data for first 50 stocks (adjust as needed)
    print("Extracting stock data...")
    stock_data = extractor.get_multiple_stocks(symbols[:50], period="2y", delay=0.5)
    
    # Save data
    extractor.save_data(stock_data, "stock_data_summary.json")
    
    print(f"Successfully extracted data for {len(stock_data)} stocks")