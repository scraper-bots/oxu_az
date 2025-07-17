import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from stock_data_extractor import StockDataExtractor

class StockPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_features(self, stock_data):
        """Create features for machine learning model"""
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
            df['pe_ratio'] = stock['pe_ratio'] if stock['pe_ratio'] > 0 else np.nan
            
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
    
    def prepare_data(self, features_df, target_column='future_return_20d'):
        """Prepare data for training"""
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
        y = features_df[target_column].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Remove extreme outliers
        y = y.clip(lower=y.quantile(0.01), upper=y.quantile(0.99))
        
        self.feature_columns = available_cols
        return X, y
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the prediction model"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Choose model
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print(importance_df.head(10))
        
        return self.model
    
    def predict_stock_returns(self, stock_data):
        """Predict future returns for stocks"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        predictions = []
        
        for stock in stock_data:
            try:
                df = stock['data'].copy()
                symbol = stock['symbol']
                
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
                latest_data['pe_ratio'] = stock['pe_ratio'] if stock['pe_ratio'] > 0 else np.nan
                latest_data['hl_spread'] = (latest_data['High'] - latest_data['Low']) / latest_data['Close']
                latest_data['price_position'] = (latest_data['Close'] - latest_data['Low']) / (latest_data['High'] - latest_data['Low'])
                
                # Select features
                X = latest_data[self.feature_columns].copy()
                X = X.fillna(X.median())
                
                # Scale and predict
                X_scaled = self.scaler.transform(X)
                prediction = self.model.predict(X_scaled)[0]
                
                predictions.append({
                    'symbol': symbol,
                    'current_price': float(stock['current_price']),
                    'predicted_return_20d': float(prediction),
                    'market_cap': stock['market_cap'],
                    'pe_ratio': stock['pe_ratio'],
                    'sector': stock['sector']
                })
                
            except Exception as e:
                print(f"Error predicting for {symbol}: {e}")
                continue
        
        return predictions
    
    def save_model(self, filename):
        """Save trained model"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename):
        """Load trained model"""
        data = joblib.load(filename)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        print(f"Model loaded from {filename}")

if __name__ == "__main__":
    # Extract stock data
    extractor = StockDataExtractor()
    symbols = extractor.get_sp500_symbols()
    
    print("Extracting stock data for model training...")
    stock_data = extractor.get_multiple_stocks(symbols[:100], period="2y", delay=0.5)
    
    # Create and train model
    model = StockPredictionModel()
    
    print("Creating features...")
    features_df = model.create_features(stock_data)
    
    print("Preparing data...")
    X, y = model.prepare_data(features_df)
    
    print("Training model...")
    model.train_model(X, y, model_type='random_forest')
    
    # Save model
    model.save_model('stock_prediction_model.pkl')
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict_stock_returns(stock_data)
    
    # Save predictions
    predictions_df = pd.DataFrame(predictions)
    predictions_df.to_csv('stock_predictions.csv', index=False)
    
    print(f"Predictions saved for {len(predictions)} stocks")