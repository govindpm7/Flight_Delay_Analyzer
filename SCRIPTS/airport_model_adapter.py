#!/usr/bin/env python3
"""
Simple adapter for the new airport-level model
"""
import joblib
import json
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any

class AirportModelPredictor:
    """Simple predictor for the new airport-level model"""
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.model = joblib.load(f"{model_dir}/model.pkl")
        self.preprocessor = joblib.load(f"{model_dir}/preprocessor.pkl")
        
        with open(f"{model_dir}/metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load lookup data
        self.bts_airport = pd.read_csv(f"{model_dir}/bts_lookup_airport.csv")
        self.bts_carrier = pd.read_csv(f"{model_dir}/bts_lookup_carrier.csv")
    
    def predict(self, carrier: str, origin: str, dest: str, flight_date: date, dep_hour: int) -> Dict[str, Any]:
        """Make a dynamic prediction - only uses available BTS data, excludes missing components"""
        
        # Check data availability first
        carrier_info = self.bts_carrier[self.bts_carrier['OP_CARRIER'] == carrier]
        origin_info = self.bts_airport[self.bts_airport['ORIGIN'] == origin]
        dest_info = self.bts_airport[self.bts_airport['ORIGIN'] == dest]
        
        data_availability = {
            'carrier': not carrier_info.empty,
            'origin': not origin_info.empty, 
            'dest': not dest_info.empty
        }
        
        # If no data is available, return a message instead of prediction
        if not any(data_availability.values()):
            return {
                'predicted_delay': None,
                'confidence_mae': None,
                'lower_bound': None,
                'upper_bound': None,
                'carrier_avg_delay': None,
                'origin_delay_rate': None,
                'dest_delay_rate': None,
                'data_availability': data_availability,
                'using_fallback_data': False,
                'prediction_available': False,
                'message': f"No historical data found for carrier {carrier}, origin {origin}, or destination {dest}. Cannot generate prediction."
            }
        
        # Build features dynamically based on available data
        # Start with all required columns with default values
        features = {
            'airport': origin,
            'carrier': carrier,
            'year': flight_date.year,
            'month': flight_date.month,
            'dep_hour': dep_hour,
            'is_weekend': 1 if flight_date.weekday() in (5, 6) else 0,
            # Default values for all model-required columns
            'arr_flights': 1000.0,
            'arr_del15': 200.0,
            'arr_cancelled': 10.0,
            'arr_diverted': 2.0,
            'total_delay_rate': 0.2,
            'carrier_delay_rate': 0.1,
            'weather_delay_rate': 0.05,
            'nas_delay_rate': 0.08,
            'security_delay_rate': 0.01,
            'late_aircraft_delay_rate': 0.06,
            'avg_carrier_delay': 50.0,
            'avg_weather_delay': 30.0,
            'avg_nas_delay': 40.0,
            'avg_security_delay': 10.0,
            'avg_late_aircraft_delay': 45.0
        }
        
        # Add carrier-specific features only if carrier data is available
        if data_availability['carrier']:
            try:
                carrier_data = carrier_info.iloc[0]
                features.update({
                    'carrier_delay_rate': carrier_data['carrier_delay_rate_origin'],
                    'avg_carrier_delay': carrier_data['avg_carrier_delay_origin'],
                    'carrier_weather_delay_rate': carrier_data.get('weather_delay_rate_origin', 0.05),
                    'carrier_nas_delay_rate': carrier_data.get('nas_delay_rate_origin', 0.08)
                })
            except (KeyError, IndexError):
                data_availability['carrier'] = False
        
        # Add origin airport features only if origin data is available
        if data_availability['origin']:
            try:
                origin_data = origin_info.iloc[0]
                features.update({
                    'total_delay_rate': origin_data['total_delay_rate_origin'],
                    'avg_delay_minutes': origin_data['avg_delay_minutes_origin'],
                    'weather_delay_rate': origin_data.get('weather_delay_rate_origin', 0.05),
                    'nas_delay_rate': origin_data.get('nas_delay_rate_origin', 0.08),
                    'security_delay_rate': origin_data.get('security_delay_rate_origin', 0.01),
                    'late_aircraft_delay_rate': origin_data.get('late_aircraft_delay_rate_origin', 0.06)
                })
            except (KeyError, IndexError):
                data_availability['origin'] = False
        
        # Add destination airport features only if destination data is available
        if data_availability['dest']:
            try:
                dest_data = dest_info.iloc[0]
                features.update({
                    'dest_delay_rate': dest_data['total_delay_rate_origin'],
                    'dest_avg_delay': dest_data['avg_delay_minutes_origin'],
                    'dest_weather_delay_rate': dest_data.get('weather_delay_rate_origin', 0.05),
                    'dest_nas_delay_rate': dest_data.get('nas_delay_rate_origin', 0.08)
                })
            except (KeyError, IndexError):
                data_availability['dest'] = False
        
        # Create DataFrame
        X = pd.DataFrame([features])
        
        try:
            # Make prediction
            prediction = self.model.predict(X)[0]
            
            # Calculate confidence interval using MAE
            mae = self.metadata.get('selected_mae', 5.0)
            lower_bound = max(0, prediction - mae)
            upper_bound = prediction + mae
            
            # Get additional info for display
            carrier_avg_delay = features.get('avg_carrier_delay', 0)
            origin_delay_rate = features.get('total_delay_rate', 0) * 100
            dest_delay_rate = features.get('dest_delay_rate', 0) * 100
            
            return {
                'predicted_delay': prediction,
                'confidence_mae': mae,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'carrier_avg_delay': carrier_avg_delay,
                'origin_delay_rate': origin_delay_rate,
                'dest_delay_rate': dest_delay_rate,
                'data_availability': data_availability,
                'using_fallback_data': False,
                'prediction_available': True,
                'message': f"Prediction based on available data: {', '.join([k for k, v in data_availability.items() if v])}"
            }
            
        except Exception as e:
            return {
                'predicted_delay': None,
                'confidence_mae': None,
                'lower_bound': None,
                'upper_bound': None,
                'carrier_avg_delay': None,
                'origin_delay_rate': None,
                'dest_delay_rate': None,
                'data_availability': data_availability,
                'using_fallback_data': False,
                'prediction_available': False,
                'message': f"Prediction failed: {str(e)}"
            }

def create_airport_predictor(model_dir: str):
    """Create airport-level predictor"""
    try:
        predictor = AirportModelPredictor(model_dir)
        return predictor, predictor.metadata
    except Exception as e:
        print(f"Error creating airport predictor: {e}")
        return None, None
