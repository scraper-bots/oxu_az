import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv
import json

load_dotenv()

class PolygonScraper:
    def __init__(self):
        self.api_key = os.getenv('POLYGON')
        self.base_url = "https://api.polygon.io"
        self.session = requests.Session()
        
    def get_stock_tickers(self, market='stocks', limit=5000):
        """Get all available stock tickers"""
        url = f"{self.base_url}/v3/reference/tickers"
        params = {
            'market': market,
            'active': 'true',
            'limit': limit,
            'apikey': self.api_key
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            print(f"Error fetching tickers: {response.status_code}")
            return []
    
    def get_stock_details(self, ticker):
        """Get detailed information about a stock"""
        url = f"{self.base_url}/v3/reference/tickers/{ticker}"
        params = {'apikey': self.api_key}
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', {})
        else:
            print(f"Error fetching details for {ticker}: {response.status_code}")
            return {}
    
    def get_historical_data(self, ticker, start_date, end_date, timespan='day', multiplier=1):
        """Get historical stock data"""
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{start_date}/{end_date}"
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,
            'apikey': self.api_key
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            print(f"Error fetching historical data for {ticker}: {response.status_code}")
            return []
    
    def get_real_time_quote(self, ticker):
        """Get real-time quote for a stock"""
        url = f"{self.base_url}/v2/last/nbbo/{ticker}"
        params = {'apikey': self.api_key}
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', {})
        else:
            print(f"Error fetching quote for {ticker}: {response.status_code}")
            return {}
    
    def get_stock_news(self, ticker=None, limit=1000):
        """Get stock news"""
        url = f"{self.base_url}/v2/reference/news"
        params = {
            'limit': limit,
            'apikey': self.api_key
        }
        
        if ticker:
            params['ticker'] = ticker
            
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', [])
        else:
            print(f"Error fetching news: {response.status_code}")
            return []
    
    def get_market_holidays(self, year=None):
        """Get market holidays"""
        url = f"{self.base_url}/v1/marketstatus/upcoming"
        params = {'apikey': self.api_key}
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching market holidays: {response.status_code}")
            return {}
    
    def get_stock_financials(self, ticker, limit=100):
        """Get stock financials"""
        url = f"{self.base_url}/vX/reference/financials"
        params = {
            'ticker': ticker,
            'limit': limit,
            'apikey': self.api_key
        }
        
        response = self.session.get(url, params=params)
        if response.status_code == 200:
            return response.json().get('results', [])
        else:
            print(f"Error fetching financials for {ticker}: {response.status_code}")
            return []
    
    def scrape_comprehensive_data(self, tickers=None, days_back=365):
        """Scrape comprehensive data for multiple stocks"""
        if not tickers:
            print("Fetching all available tickers...")
            ticker_data = self.get_stock_tickers()
            tickers = [t['ticker'] for t in ticker_data[:100]]  # Limit to first 100 for demo
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        all_data = {}
        
        for i, ticker in enumerate(tickers):
            print(f"Processing {ticker} ({i+1}/{len(tickers)})")
            
            stock_data = {
                'ticker': ticker,
                'details': self.get_stock_details(ticker),
                'historical_data': self.get_historical_data(ticker, start_date, end_date),
                'current_quote': self.get_real_time_quote(ticker),
                'news': self.get_stock_news(ticker, limit=50),
                'financials': self.get_stock_financials(ticker, limit=10)
            }
            
            all_data[ticker] = stock_data
            
            # Rate limiting - free tier has 5 requests per minute
            time.sleep(12)  # Wait 12 seconds between requests
        
        return all_data
    
    def save_data_to_files(self, data, base_filename='polygon_data'):
        """Save scraped data to various file formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save as JSON
        json_filename = f"{base_filename}_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        # Create CSV files for historical data
        all_historical = []
        for ticker, stock_data in data.items():
            historical = stock_data.get('historical_data', [])
            for record in historical:
                record['ticker'] = ticker
                record['date'] = datetime.fromtimestamp(record['t']/1000).strftime('%Y-%m-%d')
                all_historical.append(record)
        
        if all_historical:
            df_historical = pd.DataFrame(all_historical)
            csv_filename = f"{base_filename}_historical_{timestamp}.csv"
            df_historical.to_csv(csv_filename, index=False)
            print(f"Historical data saved to {csv_filename}")
        
        # Create CSV for stock details
        details_data = []
        for ticker, stock_data in data.items():
            details = stock_data.get('details', {})
            if details:
                details['ticker'] = ticker
                details_data.append(details)
        
        if details_data:
            df_details = pd.DataFrame(details_data)
            details_filename = f"{base_filename}_details_{timestamp}.csv"
            df_details.to_csv(details_filename, index=False)
            print(f"Stock details saved to {details_filename}")
        
        print(f"All data saved to {json_filename}")
        return json_filename

def main():
    scraper = PolygonScraper()
    
    # Example usage - scrape data for specific tickers
    popular_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX', 'SPY', 'QQQ']
    
    print("Starting comprehensive stock data scraping...")
    data = scraper.scrape_comprehensive_data(tickers=popular_tickers, days_back=90)
    
    # Save data
    filename = scraper.save_data_to_files(data)
    print(f"Data scraping completed! Check {filename}")
    
    # Get general market news
    print("\nFetching market news...")
    news = scraper.get_stock_news(limit=100)
    news_filename = f"market_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(news_filename, 'w') as f:
        json.dump(news, f, indent=2, default=str)
    print(f"Market news saved to {news_filename}")

if __name__ == "__main__":
    main()