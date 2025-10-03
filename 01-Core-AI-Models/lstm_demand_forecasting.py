
"""
LSTM-based Demand Forecasting for Distribution Management
Author: Senior AI Manager - Unilever DMS Team
Purpose: Predict product demand across distributor network using time series analysis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class DemandForecastingModel:
    def __init__(self, sequence_length=30, n_features=5):
        """
        Initialize LSTM model for demand forecasting

        Args:
            sequence_length (int): Number of time steps to look back
            n_features (int): Number of features (product, region, seasonality, etc.)
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        self.scaler = MinMaxScaler()

    def create_sequences(self, data, target_col='demand'):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []

        for i in range(len(data) - self.sequence_length):
            seq = data[i:(i + self.sequence_length)]
            target = data[i + self.sequence_length][target_col]
            sequences.append(seq)
            targets.append(target)

        return np.array(sequences), np.array(targets)

    def build_model(self):
        """Build LSTM architecture for demand forecasting"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1, activation='relu')  # Demand cannot be negative
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        self.model = model
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """Train the LSTM model"""
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            shuffle=False
        )

        return history

    def predict(self, X_test):
        """Make demand predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        predictions = self.model.predict(X_test)
        return predictions.flatten()

    def calculate_business_metrics(self, actual, predicted):
        """Calculate business-relevant metrics"""
        mae = mean_absolute_error(actual, predicted)
        mse = mean_squared_error(actual, predicted)
        rmse = np.sqrt(mse)

        # Business metrics
        accuracy = 100 - (mae / np.mean(actual) * 100)
        forecast_bias = np.mean(predicted - actual)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'Accuracy_%': accuracy,
            'Forecast_Bias': forecast_bias,
            'Total_Demand_Actual': np.sum(actual),
            'Total_Demand_Predicted': np.sum(predicted)
        }

# Example usage for DMS implementation
def implement_demand_forecasting():
    """
    Example implementation for Unilever DMS
    Shows integration with distributor data and business logic
    """

    # Sample data structure for DMS
    sample_data = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=365, freq='D'),
        'product_code': ['PROD_001'] * 365,
        'distributor_id': ['DIST_001'] * 365,
        'region': ['North'] * 365,
        'demand': np.random.randint(50, 500, 365),  # Sample demand data
        'seasonality': np.sin(np.arange(365) * 2 * np.pi / 365),  # Seasonal pattern
        'promotion': np.random.choice([0, 1], 365, p=[0.8, 0.2]),  # Promotion flag
        'competitor_activity': np.random.uniform(0, 1, 365),
        'weather_index': np.random.uniform(0.5, 1.5, 365)
    })

    print("🚀 DMS Demand Forecasting Implementation")
    print("=" * 50)
    print(f"📊 Dataset Shape: {sample_data.shape}")
    print(f"📅 Date Range: {sample_data['date'].min()} to {sample_data['date'].max()}")
    print(f"📦 Average Daily Demand: {sample_data['demand'].mean():.2f}")
    print(f"📈 Demand Volatility (CV): {(sample_data['demand'].std()/sample_data['demand'].mean()):.2%}")

    # Feature engineering for business context
    sample_data['day_of_week'] = sample_data['date'].dt.dayofweek
    sample_data['month'] = sample_data['date'].dt.month
    sample_data['is_weekend'] = (sample_data['day_of_week'] >= 5).astype(int)

    # Business insights
    weekend_uplift = sample_data.groupby('is_weekend')['demand'].mean()
    print(f"📊 Weekend vs Weekday Demand: {weekend_uplift[1]/weekend_uplift[0]:.2%} uplift")

    return sample_data

if __name__ == "__main__":
    # Demonstrate the model
    data = implement_demand_forecasting()

    # Initialize model
    model = DemandForecastingModel()
    print("\n✅ LSTM Demand Forecasting Model Initialized")
    print("🎯 Ready for distributor network deployment")
    print("🏢 Integration: SAP ERP → Feature Store → Real-time Prediction")
