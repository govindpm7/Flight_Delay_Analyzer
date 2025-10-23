"""
Prediction Module
Orchestrates feature engineering and model inference
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from datetime import date

from .feature_engine import FeatureEngine


class PredictionResult:
    """
    Encapsulates prediction results with metadata
    Clean interface for frontend consumption
    """
    
    def __init__(self, predicted_delay: float, confidence_mae: float,
                 carrier: str, origin: str, dest: str,
                 carrier_stats: Dict, origin_stats: Dict, dest_stats: Dict):
        self.predicted_delay = predicted_delay
        self.confidence_mae = confidence_mae
        self.carrier = carrier
        self.origin = origin
        self.dest = dest
        self.carrier_stats = carrier_stats
        self.origin_stats = origin_stats
        self.dest_stats = dest_stats
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for easy frontend consumption"""
        return {
            'predicted_delay': round(self.predicted_delay, 1),
            'confidence_mae': round(self.confidence_mae, 1),
            'lower_bound': round(self.predicted_delay - self.confidence_mae, 1),
            'upper_bound': round(self.predicted_delay + self.confidence_mae, 1),
            'carrier': self.carrier,
            'origin': self.origin,
            'dest': self.dest,
            'carrier_avg_delay': round(self.carrier_stats.get('avg_delay', 0), 1),
            'origin_avg_delay': round(self.origin_stats.get('avg_delay', 0), 1),
            'origin_delay_rate': round(self.origin_stats.get('delay_rate', 0) * 100, 1),
            'dest_avg_delay': round(self.dest_stats.get('avg_delay', 0), 1),
            'dest_delay_rate': round(self.dest_stats.get('delay_rate', 0) * 100, 1),
        }


class FlightDelayPredictor:
    """
    Main prediction orchestrator
    Coordinates feature engineering and model inference
    """
    
    def __init__(self, model, feature_engine: FeatureEngine, mae: float = 12.5, label_encoders: Optional[dict] = None):
        """
        Initialize predictor with model and feature engine
        
        Args:
            model: Trained model with predict() method
            feature_engine: FeatureEngine instance
            mae: Model's mean absolute error for confidence bands
            label_encoders: Dictionary of label encoders for categorical features
        """
        self.model = model
        self.feature_engine = feature_engine
        self.mae = mae
        self.label_encoders = label_encoders or {}
    
    def predict(self, carrier: str, origin: str, dest: str,
                flight_date: date, dep_hour: int) -> PredictionResult:
        """
        Generate prediction for a single flight
        
        Args:
            carrier: Airline code
            origin: Origin airport code
            dest: Destination airport code
            flight_date: Flight date
            dep_hour: Departure hour (0-23)
        
        Returns:
            PredictionResult object with all prediction details
        """
        # Build features
        features = self.feature_engine.build_features(
            carrier, origin, dest, flight_date, dep_hour
        )
        
        # Convert to DataFrame for model
        X = pd.DataFrame([features])
        
        # Apply label encoders to categorical features
        if self.label_encoders:
            for col, encoder in self.label_encoders.items():
                if col in X.columns:
                    try:
                        # Handle unknown categories by using the most frequent class
                        X[col] = X[col].map(lambda x: encoder.transform([x])[0] if x in encoder.classes_ else 0)
                    except Exception:
                        # If encoding fails, use 0 as default
                        X[col] = 0
        
        # Generate prediction
        try:
            predicted_delay = float(self.model.predict(X)[0])
            predicted_delay = max(0, predicted_delay)  # Non-negative delays
        except Exception as e:
            raise RuntimeError(f"Model prediction failed: {e}")
        
        # Get stats for result context
        carrier_stats = self.feature_engine.get_carrier_stats(carrier) or {}
        origin_stats = self.feature_engine.get_airport_stats(origin) or {}
        dest_stats = self.feature_engine.get_airport_stats(dest) or {}
        
        # Create result object
        result = PredictionResult(
            predicted_delay=predicted_delay,
            confidence_mae=self.mae,
            carrier=carrier,
            origin=origin,
            dest=dest,
            carrier_stats=carrier_stats,
            origin_stats=origin_stats,
            dest_stats=dest_stats
        )
        
        return result
    
    def predict_batch(self, flights: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions for multiple flights
        
        Args:
            flights: DataFrame with columns [carrier, origin, dest, flight_date, dep_hour]
        
        Returns:
            DataFrame with original data plus prediction columns
        """
        predictions = []
        
        for _, flight in flights.iterrows():
            result = self.predict(
                carrier=flight['carrier'],
                origin=flight['origin'],
                dest=flight['dest'],
                flight_date=flight['flight_date'],
                dep_hour=flight['dep_hour']
            )
            predictions.append(result.to_dict())
        
        return pd.DataFrame(predictions)
