#!/usr/bin/env python3

"""
Investment Bot - Stock Price Prediction and Investment Screening System
Usage: python main.py [--train] [--screen] [--symbols SYMBOLS]
"""

import argparse
import sys
import os
from stock_data_extractor import StockDataExtractor
from stock_prediction_model import StockPredictionModel
from investment_screener import InvestmentScreener, PortfolioOptimizer

def train_model(num_stocks=100):
    """Train the prediction model"""
    print("="*60)
    print("TRAINING STOCK PREDICTION MODEL")
    print("="*60)
    
    extractor = StockDataExtractor()
    model = StockPredictionModel()
    
    # Get stock symbols
    print("Fetching stock symbols...")
    symbols = extractor.get_sp500_symbols()
    
    # Extract data
    print(f"Extracting data for {num_stocks} stocks...")
    stock_data = extractor.get_multiple_stocks(symbols[:num_stocks], period="2y", delay=0.5)
    
    if not stock_data:
        print("No stock data extracted. Exiting.")
        return False
    
    # Create features and train model
    print("Creating features...")
    features_df = model.create_features(stock_data)
    
    print("Preparing data...")
    X, y = model.prepare_data(features_df)
    
    print("Training model...")
    model.train_model(X, y, model_type='random_forest')
    
    # Save model
    model.save_model('stock_prediction_model.pkl')
    
    print("Model training completed!")
    return True

def screen_investments(custom_symbols=None, top_n=15):
    """Screen investments using trained model"""
    print("="*60)
    print("SCREENING INVESTMENT OPPORTUNITIES")
    print("="*60)
    
    # Check if model exists
    if not os.path.exists('stock_prediction_model.pkl'):
        print("No trained model found. Please train the model first using --train")
        return False
    
    # Initialize screener
    screener = InvestmentScreener()
    screener.load_model('stock_prediction_model.pkl')
    
    # Get symbols
    if custom_symbols:
        symbols = custom_symbols.split(',')
    else:
        symbols = screener.extractor.get_sp500_symbols()[:200]  # Limit for performance
    
    # Run screening
    print(f"Screening {len(symbols)} stocks...")
    results = screener.get_real_time_screening(symbols, top_n=top_n)
    
    # Save detailed results
    screener.save_report(results['investment_report'], 'investment_report.json')
    results['screened_stocks'].to_csv('screened_stocks.csv', index=False)
    results['undervalued_stocks'].to_csv('undervalued_stocks.csv', index=False)
    
    # Create portfolio
    optimizer = PortfolioOptimizer(initial_capital=10000)
    portfolio = optimizer.optimize_portfolio(results['screened_stocks'])
    portfolio_metrics = optimizer.calculate_portfolio_metrics(portfolio)
    
    # Display results
    print_results(results, portfolio_metrics)
    
    return True

def print_results(results, portfolio_metrics):
    """Print screening results"""
    print("\n" + "="*60)
    print("INVESTMENT SCREENING RESULTS")
    print("="*60)
    
    screened = results['screened_stocks']
    undervalued = results['undervalued_stocks']
    report = results['investment_report']
    
    print(f"\nSUMMARY:")
    print(f"Total stocks screened: {report['summary']['total_screened']}")
    print(f"Average predicted return: {report['summary']['avg_predicted_return']:.2%}")
    print(f"Average P/E ratio: {report['summary']['avg_pe_ratio']:.1f}")
    
    print(f"\nTOP 10 INVESTMENT CANDIDATES:")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Price':<8} {'Pred Return':<12} {'P/E':<6} {'Market Cap':<12} {'Sector':<15}")
    print("-" * 80)
    
    for i, (_, stock) in enumerate(screened.head(10).iterrows()):
        market_cap_b = stock['market_cap'] / 1e9
        print(f"{stock['symbol']:<8} ${stock['current_price']:<7.2f} {stock['predicted_return_20d']:<11.2%} "
              f"{stock['pe_ratio']:<6.1f} ${market_cap_b:<11.1f}B {stock['sector']:<15}")
    
    print(f"\nTOP 5 UNDERVALUED STOCKS:")
    print("-" * 80)
    for i, (_, stock) in enumerate(undervalued.head(5).iterrows()):
        if 'investment_score' in stock:
            print(f"{stock['symbol']:<8} Score: {stock['investment_score']:<6.3f} "
                  f"Return: {stock['predicted_return_20d']:<8.2%} P/E: {stock['pe_ratio']:<6.1f}")
    
    print(f"\nPORTFOLIO OPTIMIZATION:")
    print(f"Expected Return: {portfolio_metrics['expected_return']:.2%}")
    print(f"Number of Positions: {portfolio_metrics['num_positions']}")
    print(f"Cash Remaining: ${portfolio_metrics['cash_remaining']:.2f}")
    
    print(f"\nSector Allocation:")
    for sector, weight in sorted(portfolio_metrics['sector_allocation'].items(), 
                                key=lambda x: x[1], reverse=True):
        print(f"  {sector}: {weight:.1%}")
    
    print(f"\nFILES GENERATED:")
    print("- investment_report.json (detailed report)")
    print("- screened_stocks.csv (all screened stocks)")
    print("- undervalued_stocks.csv (undervalued analysis)")
    
    print(f"\nDISCLAIMER:")
    print("This is for educational purposes only. Past performance does not guarantee future results.")
    print("Please do your own research before making investment decisions.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Investment Bot - Stock Analysis and Screening')
    parser.add_argument('--train', action='store_true', help='Train the prediction model')
    parser.add_argument('--screen', action='store_true', help='Screen investment opportunities')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of stock symbols to analyze')
    parser.add_argument('--top', type=int, default=15, help='Number of top stocks to show (default: 15)')
    parser.add_argument('--stocks', type=int, default=100, help='Number of stocks to use for training (default: 100)')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not args.train and not args.screen:
        print("Investment Bot - Stock Analysis and Screening System")
        print("\nUsage:")
        print("  python main.py --train              # Train the model")
        print("  python main.py --screen             # Screen investments")
        print("  python main.py --screen --symbols AAPL,MSFT,GOOGL")
        print("  python main.py --train --stocks 200 # Train with 200 stocks")
        print("\nOptions:")
        print("  --train          Train the prediction model")
        print("  --screen         Screen investment opportunities")
        print("  --symbols        Comma-separated stock symbols to analyze")
        print("  --top N          Show top N stocks (default: 15)")
        print("  --stocks N       Use N stocks for training (default: 100)")
        return
    
    try:
        if args.train:
            success = train_model(args.stocks)
            if not success:
                sys.exit(1)
        
        if args.screen:
            success = screen_investments(args.symbols, args.top)
            if not success:
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()