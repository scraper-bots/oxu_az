import pandas as pd
import numpy as np
from stock_prediction_model import StockPredictionModel
from stock_data_extractor import StockDataExtractor
import json

class InvestmentScreener:
    def __init__(self):
        self.model = StockPredictionModel()
        self.extractor = StockDataExtractor()
        
    def load_model(self, model_path):
        """Load pre-trained model"""
        self.model.load_model(model_path)
    
    def screen_stocks(self, stock_data, min_predicted_return=0.05, max_pe_ratio=25, min_market_cap=1e9):
        """Screen stocks based on criteria"""
        predictions = self.model.predict_stock_returns(stock_data)
        
        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(predictions)
        
        # Apply screening criteria
        screened = df[
            (df['predicted_return_20d'] >= min_predicted_return) &
            (df['pe_ratio'] <= max_pe_ratio) &
            (df['pe_ratio'] > 0) &  # Exclude negative PE ratios
            (df['market_cap'] >= min_market_cap)
        ].copy()
        
        # Calculate additional metrics
        screened['predicted_price_20d'] = screened['current_price'] * (1 + screened['predicted_return_20d'])
        screened['potential_profit_pct'] = screened['predicted_return_20d'] * 100
        
        # Sort by predicted return
        screened = screened.sort_values('predicted_return_20d', ascending=False)
        
        return screened
    
    def analyze_undervalued_stocks(self, stock_data):
        """Find potentially undervalued stocks with growth potential"""
        predictions = self.model.predict_stock_returns(stock_data)
        df = pd.DataFrame(predictions)
        
        # Calculate valuation metrics
        df['is_undervalued'] = (df['pe_ratio'] < 15) & (df['pe_ratio'] > 0)
        df['high_growth_potential'] = df['predicted_return_20d'] > 0.1
        df['large_cap'] = df['market_cap'] > 10e9
        
        # Score stocks
        df['investment_score'] = (
            df['predicted_return_20d'] * 0.4 +
            (1 / (df['pe_ratio'] + 1)) * 0.3 +
            (df['market_cap'] / 1e12) * 0.2 +
            df['is_undervalued'].astype(int) * 0.1
        )
        
        # Filter for investment candidates
        candidates = df[
            (df['predicted_return_20d'] > 0.03) &
            (df['pe_ratio'] > 0) &
            (df['pe_ratio'] < 30) &
            (df['market_cap'] > 1e9)
        ].copy()
        
        candidates = candidates.sort_values('investment_score', ascending=False)
        
        return candidates
    
    def generate_investment_report(self, screened_stocks, top_n=20):
        """Generate investment report"""
        top_stocks = screened_stocks.head(top_n)
        
        report = {
            'summary': {
                'total_screened': len(screened_stocks),
                'top_picks': len(top_stocks),
                'avg_predicted_return': float(top_stocks['predicted_return_20d'].mean()),
                'avg_pe_ratio': float(top_stocks['pe_ratio'].mean()),
                'avg_market_cap': float(top_stocks['market_cap'].mean())
            },
            'top_stocks': []
        }
        
        for _, stock in top_stocks.iterrows():
            report['top_stocks'].append({
                'symbol': stock['symbol'],
                'current_price': float(stock['current_price']),
                'predicted_return_20d': float(stock['predicted_return_20d']),
                'potential_profit_pct': float(stock['potential_profit_pct']),
                'pe_ratio': float(stock['pe_ratio']),
                'market_cap': float(stock['market_cap']),
                'sector': stock['sector']
            })
        
        return report
    
    def save_report(self, report, filename):
        """Save investment report to file"""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Investment report saved to {filename}")
    
    def get_real_time_screening(self, symbols=None, top_n=10):
        """Get real-time screening results"""
        if symbols is None:
            symbols = self.extractor.get_sp500_symbols()[:200]  # Limit for demo
        
        print(f"Extracting real-time data for {len(symbols)} stocks...")
        stock_data = self.extractor.get_multiple_stocks(symbols, period="1y", delay=0.3)
        
        print("Screening stocks...")
        screened = self.screen_stocks(stock_data)
        
        print("Analyzing undervalued stocks...")
        undervalued = self.analyze_undervalued_stocks(stock_data)
        
        print("Generating report...")
        report = self.generate_investment_report(screened, top_n)
        
        return {
            'screened_stocks': screened,
            'undervalued_stocks': undervalued,
            'investment_report': report
        }

class PortfolioOptimizer:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
    
    def optimize_portfolio(self, screened_stocks, max_positions=10, max_position_size=0.2):
        """Optimize portfolio allocation"""
        top_stocks = screened_stocks.head(max_positions)
        
        # Simple equal weight with some risk adjustment
        weights = []
        for _, stock in top_stocks.iterrows():
            # Adjust weight based on predicted return and PE ratio
            base_weight = 1 / len(top_stocks)
            return_bonus = min(stock['predicted_return_20d'] * 2, 0.5)
            pe_penalty = max(stock['pe_ratio'] / 100, 0.1)
            
            adjusted_weight = base_weight * (1 + return_bonus - pe_penalty)
            weights.append(min(adjusted_weight, max_position_size))
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate allocation
        portfolio = []
        for i, (_, stock) in enumerate(top_stocks.iterrows()):
            allocation = self.initial_capital * weights[i]
            shares = int(allocation / stock['current_price'])
            
            portfolio.append({
                'symbol': stock['symbol'],
                'shares': shares,
                'allocation': allocation,
                'current_price': stock['current_price'],
                'weight': weights[i],
                'predicted_return': stock['predicted_return_20d'],
                'sector': stock['sector']
            })
        
        return portfolio
    
    def calculate_portfolio_metrics(self, portfolio):
        """Calculate portfolio metrics"""
        total_value = sum(p['allocation'] for p in portfolio)
        expected_return = sum(p['allocation'] * p['predicted_return'] for p in portfolio) / total_value
        
        # Sector diversification
        sectors = {}
        for p in portfolio:
            sector = p['sector']
            if sector not in sectors:
                sectors[sector] = 0
            sectors[sector] += p['weight']
        
        return {
            'total_value': total_value,
            'expected_return': expected_return,
            'num_positions': len(portfolio),
            'sector_allocation': sectors,
            'cash_remaining': self.initial_capital - total_value
        }

if __name__ == "__main__":
    # Initialize screener
    screener = InvestmentScreener()
    
    # Load pre-trained model (if available)
    try:
        screener.load_model('stock_prediction_model.pkl')
    except:
        print("No pre-trained model found. Please train the model first.")
        exit()
    
    # Run real-time screening
    results = screener.get_real_time_screening(top_n=15)
    
    # Save results
    screener.save_report(results['investment_report'], 'investment_report.json')
    results['screened_stocks'].to_csv('screened_stocks.csv', index=False)
    results['undervalued_stocks'].to_csv('undervalued_stocks.csv', index=False)
    
    # Optimize portfolio
    optimizer = PortfolioOptimizer(initial_capital=10000)
    portfolio = optimizer.optimize_portfolio(results['screened_stocks'])
    portfolio_metrics = optimizer.calculate_portfolio_metrics(portfolio)
    
    # Display results
    print("\n" + "="*50)
    print("INVESTMENT SCREENING RESULTS")
    print("="*50)
    
    print(f"\nTop 10 Investment Candidates:")
    for i, stock in enumerate(results['screened_stocks'].head(10).iterrows()):
        stock = stock[1]
        print(f"{i+1}. {stock['symbol']}: {stock['potential_profit_pct']:.1f}% predicted return")
    
    print(f"\nPortfolio Metrics:")
    print(f"Expected Return: {portfolio_metrics['expected_return']:.2%}")
    print(f"Number of Positions: {portfolio_metrics['num_positions']}")
    print(f"Cash Remaining: ${portfolio_metrics['cash_remaining']:.2f}")
    
    print(f"\nSector Allocation:")
    for sector, weight in portfolio_metrics['sector_allocation'].items():
        print(f"  {sector}: {weight:.1%}")
    
    print("\nFiles generated:")
    print("- investment_report.json")
    print("- screened_stocks.csv")
    print("- undervalued_stocks.csv")