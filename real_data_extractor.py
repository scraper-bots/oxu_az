import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class RealStockDataExtractor:
    def __init__(self):
        self.retry_delay = 2  # seconds between retries
        self.max_retries = 3
        
    def get_stock_data_with_retry(self, symbol, period="1y", interval="1d"):
        """Get stock data with retry logic"""
        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                
                # Try to get historical data
                hist = ticker.history(period=period, interval=interval)
                
                if hist.empty:
                    print(f"  No data for {symbol} (attempt {attempt + 1})")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    continue
                
                # Try to get info with error handling
                try:
                    info = ticker.info
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('trailingPE', 0)
                    sector = info.get('sector', 'Unknown')
                    longName = info.get('longName', symbol)
                except Exception as e:
                    print(f"  Info error for {symbol}: {e}")
                    # Use fallback values
                    market_cap = 1000000000  # Default 1B
                    pe_ratio = 20  # Default PE
                    sector = 'Unknown'
                    longName = symbol
                
                # Calculate technical indicators
                if len(hist) >= 50:
                    hist['SMA_20'] = hist['Close'].rolling(window=20).mean()
                    hist['SMA_50'] = hist['Close'].rolling(window=50).mean()
                    hist['RSI'] = self.calculate_rsi(hist['Close'])
                    hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
                    hist['Price_Change'] = hist['Close'].pct_change()
                    hist['Price_Change_5d'] = hist['Close'].pct_change(periods=5)
                    hist['Price_Change_20d'] = hist['Close'].pct_change(periods=20)
                    hist['Volatility'] = hist['Price_Change'].rolling(window=20).std()
                    
                    return {
                        'symbol': symbol,
                        'name': longName,
                        'data': hist,
                        'current_price': hist['Close'].iloc[-1],
                        'market_cap': market_cap,
                        'pe_ratio': pe_ratio if pe_ratio > 0 else 20,
                        'sector': sector
                    }
                else:
                    print(f"  Insufficient data for {symbol}")
                    
            except Exception as e:
                print(f"  Error for {symbol} (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                continue
        
        return None
    
    def calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def extract_real_stock_data(self, symbols):
        """Extract real stock data for given symbols"""
        results = []
        
        print(f"Extracting real data for {len(symbols)} stocks...")
        
        for i, symbol in enumerate(symbols):
            print(f"[{i+1}/{len(symbols)}] Processing {symbol}...")
            
            data = self.get_stock_data_with_retry(symbol)
            if data:
                results.append(data)
                print(f"  ✓ Success: {data['name']} - ${data['current_price']:.2f}")
            else:
                print(f"  ✗ Failed: {symbol}")
            
            # Rate limiting
            time.sleep(1)
        
        return results

def create_features_for_ml(stock_data):
    """Create features for machine learning"""
    features_list = []
    
    for stock in stock_data:
        df = stock['data'].copy()
        symbol = stock['symbol']
        
        # Skip if not enough data
        if len(df) < 60:
            continue
            
        # Create target variable (future price change)
        df['future_return_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['future_return_20d'] = df['Close'].shift(-20) / df['Close'] - 1
        
        # Technical indicators
        df['price_vs_sma20'] = df['Close'] / df['SMA_20']
        df['price_vs_sma50'] = df['Close'] / df['SMA_50']
        df['sma_ratio'] = df['SMA_20'] / df['SMA_50']
        
        # Volume indicators
        df['volume_ratio'] = df['Volume'] / df['Volume_MA']
        
        # Price momentum
        df['momentum_5d'] = df['Close'] / df['Close'].shift(5)
        df['momentum_20d'] = df['Close'] / df['Close'].shift(20)
        
        # Volatility features
        df['volatility_ratio'] = df['Volatility'] / df['Volatility'].shift(20)
        
        # Market cap and PE ratio (static features)
        df['market_cap'] = stock['market_cap']
        df['pe_ratio'] = stock['pe_ratio']
        
        # High-low spread
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        
        # Price position in range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Add symbol as categorical feature
        df['symbol'] = symbol
        
        # Remove rows with NaN values
        df = df.dropna()
        
        if len(df) > 0:
            features_list.append(df)
    
    if not features_list:
        raise ValueError("No valid data found for feature creation")
        
    return pd.concat(features_list, ignore_index=True)

def train_prediction_model(features_df):
    """Train the stock prediction model"""
    
    # Select feature columns
    feature_cols = [
        'price_vs_sma20', 'price_vs_sma50', 'sma_ratio', 'RSI',
        'volume_ratio', 'momentum_5d', 'momentum_20d', 'volatility_ratio',
        'market_cap', 'pe_ratio', 'hl_spread', 'price_position',
        'Price_Change', 'Price_Change_5d', 'Price_Change_20d'
    ]
    
    # Filter available columns
    available_cols = [col for col in feature_cols if col in features_df.columns]
    
    X = features_df[available_cols].copy()
    y = features_df['future_return_20d'].copy()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Remove extreme outliers
    y = y.clip(lower=y.quantile(0.01), upper=y.quantile(0.99))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': available_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances:")
    print(importance_df.head(10))
    
    return model, scaler, available_cols

def predict_stock_returns(model, scaler, feature_cols, stock_data):
    """Predict future returns for stocks"""
    predictions = []
    
    for stock in stock_data:
        try:
            df = stock['data'].copy()
            symbol = stock['symbol']
            
            # Skip if not enough data
            if len(df) < 50:
                continue
            
            # Create features for the latest data point
            latest_data = df.iloc[-1:].copy()
            
            # Calculate the same features as in training
            latest_data['price_vs_sma20'] = latest_data['Close'] / latest_data['SMA_20']
            latest_data['price_vs_sma50'] = latest_data['Close'] / latest_data['SMA_50']
            latest_data['sma_ratio'] = latest_data['SMA_20'] / latest_data['SMA_50']
            latest_data['volume_ratio'] = latest_data['Volume'] / stock['data']['Volume_MA'].iloc[-1]
            latest_data['momentum_5d'] = latest_data['Close'] / df['Close'].iloc[-6]
            latest_data['momentum_20d'] = latest_data['Close'] / df['Close'].iloc[-21]
            latest_data['volatility_ratio'] = latest_data['Volatility'] / df['Volatility'].iloc[-21]
            latest_data['market_cap'] = stock['market_cap']
            latest_data['pe_ratio'] = stock['pe_ratio']
            latest_data['hl_spread'] = (latest_data['High'] - latest_data['Low']) / latest_data['Close']
            latest_data['price_position'] = (latest_data['Close'] - latest_data['Low']) / (latest_data['High'] - latest_data['Low'])
            
            # Select features
            X = latest_data[feature_cols].copy()
            X = X.fillna(X.median())
            
            # Scale and predict
            X_scaled = scaler.transform(X)
            prediction = model.predict(X_scaled)[0]
            
            predictions.append({
                'symbol': symbol,
                'name': stock['name'],
                'current_price': float(stock['current_price']),
                'predicted_return_20d': float(prediction),
                'market_cap': stock['market_cap'],
                'pe_ratio': stock['pe_ratio'],
                'sector': stock['sector'],
                'predicted_price_20d': float(stock['current_price'] * (1 + prediction)),
                'potential_profit_pct': float(prediction * 100)
            })
            
        except Exception as e:
            print(f"Error predicting for {symbol}: {e}")
            continue
    
    return predictions

def screen_investments(predictions):
    """Screen stocks for investment opportunities"""
    df = pd.DataFrame(predictions)
    
    if df.empty:
        return df
    
    # Investment screening criteria
    screened = df[
        (df['predicted_return_20d'] > 0.05) &  # At least 5% return
        (df['pe_ratio'] < 40) &  # P/E ratio less than 40
        (df['pe_ratio'] > 5) &   # P/E ratio greater than 5
        (df['market_cap'] > 10000000000)  # Market cap > $10B
    ].copy()
    
    if screened.empty:
        print("No stocks meet the screening criteria. Showing top performers...")
        screened = df.nlargest(10, 'predicted_return_20d')
    
    # Sort by predicted return
    screened = screened.sort_values('predicted_return_20d', ascending=False)
    
    # Calculate investment score
    screened['investment_score'] = (
        screened['predicted_return_20d'] * 0.4 +
        (1 / (screened['pe_ratio'] / 20)) * 0.3 +
        (screened['market_cap'] / 1e12) * 0.2 +
        (screened['predicted_return_20d'] > 0.1).astype(int) * 0.1
    )
    
    return screened

def main():
    """Main function to run the complete analysis"""
    print("=" * 60)
    print("REAL STOCK INVESTMENT ANALYSIS")
    print("=" * 60)
    
    # Define stock symbols to analyze
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
        'JPM', 'V', 'WMT', 'PG', 'JNJ', 'HD', 'BAC', 'NFLX',
        'DIS', 'KO', 'PFE', 'CRM', 'PYPL', 'ADBE', 'NKE',
        'INTC', 'CMCSA', 'VZ', 'ORCL', 'IBM', 'QCOM', 'AMD'
    ]
    
    # Extract real stock data
    print("\n1. Extracting real stock data...")
    extractor = RealStockDataExtractor()
    stock_data = extractor.extract_real_stock_data(symbols)
    
    if not stock_data:
        print("❌ No stock data was successfully extracted!")
        return
    
    print(f"✓ Successfully extracted data for {len(stock_data)} stocks")
    
    # Create features
    print("\n2. Creating features for machine learning...")
    try:
        features_df = create_features_for_ml(stock_data)
        print(f"✓ Created {len(features_df)} feature records")
    except Exception as e:
        print(f"❌ Error creating features: {e}")
        return
    
    # Train model
    print("\n3. Training prediction model...")
    try:
        model, scaler, feature_cols = train_prediction_model(features_df)
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return
    
    # Make predictions
    print("\n4. Making predictions...")
    predictions = predict_stock_returns(model, scaler, feature_cols, stock_data)
    
    if not predictions:
        print("❌ No predictions generated!")
        return
    
    print(f"✓ Generated predictions for {len(predictions)} stocks")
    
    # Screen investments
    print("\n5. Screening investment opportunities...")
    screened = screen_investments(predictions)
    
    # Display results
    print("\n" + "=" * 60)
    print("INVESTMENT SCREENING RESULTS")
    print("=" * 60)
    
    print(f"\nTOP INVESTMENT OPPORTUNITIES:")
    print("-" * 80)
    print(f"{'Symbol':<8} {'Company':<20} {'Current $':<10} {'Pred Return':<12} {'Target $':<10} {'P/E':<6}")
    print("-" * 80)
    
    for _, stock in screened.head(10).iterrows():
        company_name = stock['name'][:18] + '..' if len(stock['name']) > 20 else stock['name']
        print(f"{stock['symbol']:<8} {company_name:<20} ${stock['current_price']:<9.2f} {stock['predicted_return_20d']:<11.2%} "
              f"${stock['predicted_price_20d']:<9.2f} {stock['pe_ratio']:<6.1f}")
    
    print(f"\nUNDERVALUED STOCKS (Low P/E with Good Returns):")
    print("-" * 80)
    
    undervalued = screened[screened['pe_ratio'] < 20].head(5)
    if not undervalued.empty:
        for _, stock in undervalued.iterrows():
            print(f"{stock['symbol']:<8} P/E: {stock['pe_ratio']:<6.1f} "
                  f"Return: {stock['predicted_return_20d']:<8.2%} "
                  f"Current: ${stock['current_price']:<7.2f}")
    else:
        print("No undervalued stocks found with current criteria.")
    
    print(f"\nHIGH GROWTH POTENTIAL (>8% predicted return):")
    print("-" * 80)
    
    high_growth = screened[screened['predicted_return_20d'] > 0.08].head(5)
    if not high_growth.empty:
        for _, stock in high_growth.iterrows():
            print(f"{stock['symbol']:<8} Return: {stock['predicted_return_20d']:<8.2%} "
                  f"Current: ${stock['current_price']:<7.2f} "
                  f"Target: ${stock['predicted_price_20d']:<7.2f}")
    else:
        print("No high growth stocks found with current criteria.")
    
    # Save results
    screened.to_csv('real_investment_opportunities.csv', index=False)
    
    # Create investment report
    report = {
        'analysis_date': datetime.now().isoformat(),
        'summary': {
            'total_analyzed': len(predictions),
            'investment_opportunities': len(screened),
            'avg_predicted_return': float(screened['predicted_return_20d'].mean()) if len(screened) > 0 else 0,
            'best_opportunity': screened.iloc[0]['symbol'] if len(screened) > 0 else None,
        },
        'top_picks': screened.head(10).to_dict('records')
    }
    
    with open('real_investment_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📊 ANALYSIS COMPLETE:")
    print(f"- Analyzed {len(predictions)} stocks with real data")
    print(f"- Found {len(screened)} investment opportunities")
    if len(screened) > 0:
        print(f"- Best opportunity: {screened.iloc[0]['symbol']} ({screened.iloc[0]['predicted_return_20d']:.1%} return)")
        print(f"- Average predicted return: {screened['predicted_return_20d'].mean():.1%}")
    
    print(f"\n📁 FILES GENERATED:")
    print("- real_investment_opportunities.csv")
    print("- real_investment_report.json")
    
    print(f"\n⚠️  IMPORTANT DISCLAIMER:")
    print("This analysis is based on historical data and technical indicators.")
    print("Past performance does not guarantee future results.")
    print("Always consult with a financial advisor before making investment decisions.")

if __name__ == "__main__":
    main()