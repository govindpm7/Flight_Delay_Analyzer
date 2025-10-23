"""
Feature Engineering Module
Handles all feature derivation from minimal inputs
"""
import pandas as pd
from typing import Dict, Optional
from datetime import date
import os
import sys

# Add current directory to path for advanced feature imports
sys.path.append(os.path.dirname(__file__))


class FeatureEngine:
    """
    Transforms minimal flight inputs into complete feature vectors
    All business logic for feature engineering lives here
    """
    
    def __init__(self, bts_airport_df: pd.DataFrame, 
                 bts_carrier_df: pd.DataFrame, 
                 lookup_df: Optional[pd.DataFrame] = None,
                 use_advanced_features: bool = True):
        """
        Initialize with pre-loaded lookup tables
        
        Args:
            bts_airport_df: Airport statistics from BTS
            bts_carrier_df: Carrier statistics from BTS  
            lookup_df: Optional route/flight lookup data
            use_advanced_features: Whether to apply PCA-derived features
        """
        # Store default values first
        self.default_airport_delay = bts_airport_df['avg_delay_minutes_origin'].mean()
        self.default_carrier_delay = bts_carrier_df['avg_delay_minutes_origin'].mean()
        self.default_distance = 500.0
        
        # Build caches
        self.airport_cache = self._build_airport_cache(bts_airport_df)
        self.carrier_cache = self._build_carrier_cache(bts_carrier_df)
        self.route_cache = self._build_route_cache(lookup_df)
        
        # Initialize advanced feature engineer if enabled
        self.use_advanced_features = use_advanced_features
        if use_advanced_features:
            try:
                from feature_engineering_advanced import AdvancedFeatureEngineer
                self.advanced_engineer = AdvancedFeatureEngineer()
                print("Advanced features enabled")
            except Exception as e:
                print(f"Could not load advanced features: {e}")
                self.use_advanced_features = False
    
    def _build_airport_cache(self, df: pd.DataFrame) -> Dict:
        """Build fast lookup dictionary for airport stats"""
        cache = {}
        for _, row in df.iterrows():
            cache[row['ORIGIN']] = {
                'avg_delay': row['avg_delay_minutes_origin'],
                'delay_rate': row['total_delay_rate_origin'],
                'carrier_delay_rate': row['carrier_delay_rate_origin'],
                'weather_delay_rate': row['weather_delay_rate_origin'],
                'nas_delay_rate': row['nas_delay_rate_origin'],
                'security_delay_rate': row['security_delay_rate_origin'],
                'late_aircraft_delay_rate': row['late_aircraft_delay_rate_origin'],
                'avg_carrier_delay': row['avg_carrier_delay_origin'],
                'avg_weather_delay': row['avg_weather_delay_origin'],
                'avg_nas_delay': row['avg_nas_delay_origin'],
                'avg_security_delay': row['avg_security_delay_origin'],
                'avg_late_aircraft_delay': row['avg_late_aircraft_delay_origin'],
            }
        return cache
    
    def _build_carrier_cache(self, df: pd.DataFrame) -> Dict:
        """Build fast lookup dictionary for carrier stats"""
        cache = {}
        for _, row in df.iterrows():
            cache[row['OP_CARRIER']] = {
                'avg_delay': row['avg_delay_minutes_origin'],
                'delay_rate': row['total_delay_rate_origin'],
                'carrier_delay_rate': row['carrier_delay_rate_origin'],
                'weather_delay_rate': row['weather_delay_rate_origin'],
                'nas_delay_rate': row['nas_delay_rate_origin'],
                'security_delay_rate': row['security_delay_rate_origin'],
                'late_aircraft_delay_rate': row['late_aircraft_delay_rate_origin'],
            }
        return cache
    
    def _build_route_cache(self, df: Optional[pd.DataFrame]) -> Dict:
        """Build fast lookup dictionary for route stats"""
        cache = {}
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                route_key = f"{row.get('ORIGIN', '')}-{row.get('DEST', '')}"
                cache[route_key] = {
                    'distance': row.get('DISTANCE', self.default_distance),
                    'avg_delay': row.get('avg_delay', 0.0),
                }
        return cache
    
    def build_features(self, carrier: str, origin: str, dest: str, 
                      flight_date: date, dep_hour: int) -> Dict:
        """
        Build complete feature vector from minimal inputs
        Now includes PCA-derived advanced features if enabled
        
        Args:
            carrier: Airline code (e.g., "AA")
            origin: Origin airport code (e.g., "LAX")
            dest: Destination airport code (e.g., "JFK")
            flight_date: Flight date
            dep_hour: Departure hour (0-23)
        
        Returns:
            Complete feature dictionary ready for model
        """
        # Normalize inputs
        carrier = carrier.upper()
        origin = origin.upper()
        dest = dest.upper()
        
        # Derive temporal features
        dow = flight_date.weekday()
        month = flight_date.month
        is_weekend = 1 if dow in (5, 6) else 0
        
        # Lookup carrier stats
        carrier_info = self.carrier_cache.get(carrier, {
            'avg_delay': self.default_carrier_delay,
            'delay_rate': 0.20
        })
        
        # Lookup airport stats
        origin_info = self.airport_cache.get(origin, {
            'avg_delay': self.default_airport_delay,
            'delay_rate': 0.18
        })
        
        dest_info = self.airport_cache.get(dest, {
            'avg_delay': self.default_airport_delay,
            'delay_rate': 0.18
        })
        
        # Lookup route stats
        route_key = f"{origin}-{dest}"
        route_info = self.route_cache.get(route_key, {
            'distance': self.default_distance,
            'avg_delay': 10.0
        })
        
        # Build comprehensive feature dictionary in the exact order expected by the model
        # This order must match the feature_columns in metadata.json
        features = {
            'dep_hour': int(dep_hour),
            'dow': int(dow),
            'month': int(month),
            'is_weekend': int(is_weekend),
            'DISTANCE': float(route_info['distance']),
            'route': route_key,
            'ORIGIN': origin,
            'DEST': dest,
            'OP_CARRIER': carrier,
            'route_avg_delay': float(route_info['avg_delay']),
            'origin_avg_delay': float(origin_info['avg_delay']),
            'airline_avg_delay': float(carrier_info['avg_delay']),
            'total_delay_rate_origin': float(origin_info.get('delay_rate', 0)),
            'carrier_delay_rate_origin': float(origin_info.get('carrier_delay_rate', 0)),
            'weather_delay_rate_origin': float(origin_info.get('weather_delay_rate', 0)),
            'nas_delay_rate_origin': float(origin_info.get('nas_delay_rate', 0)),
            'security_delay_rate_origin': float(origin_info.get('security_delay_rate', 0)),
            'late_aircraft_delay_rate_origin': float(origin_info.get('late_aircraft_delay_rate', 0)),
            'avg_delay_minutes_origin': float(origin_info.get('avg_delay', 0)),
            'avg_carrier_delay_origin': float(origin_info.get('avg_carrier_delay', 0)),
            'avg_weather_delay_origin': float(origin_info.get('avg_weather_delay', 0)),
            'avg_nas_delay_origin': float(origin_info.get('avg_nas_delay', 0)),
            'avg_security_delay_origin': float(origin_info.get('avg_security_delay', 0)),
            'avg_late_aircraft_delay_origin': float(origin_info.get('avg_late_aircraft_delay', 0)),
            'total_delay_rate_dest': float(dest_info.get('delay_rate', 0)),
            'carrier_delay_rate_dest': float(dest_info.get('carrier_delay_rate', 0)),
            'weather_delay_rate_dest': float(dest_info.get('weather_delay_rate', 0)),
            'nas_delay_rate_dest': float(dest_info.get('nas_delay_rate', 0)),
            'security_delay_rate_dest': float(dest_info.get('security_delay_rate', 0)),
            'late_aircraft_delay_rate_dest': float(dest_info.get('late_aircraft_delay_rate', 0)),
            'avg_delay_minutes_dest': float(dest_info.get('avg_delay', 0)),
            'avg_carrier_delay_dest': float(dest_info.get('avg_carrier_delay', 0)),
            'avg_weather_delay_dest': float(dest_info.get('avg_weather_delay', 0)),
            'avg_nas_delay_dest': float(dest_info.get('avg_nas_delay', 0)),
            'avg_security_delay_dest': float(dest_info.get('avg_security_delay', 0)),
            'avg_late_aircraft_delay_dest': float(dest_info.get('avg_late_aircraft_delay', 0)),
        }
        
        # Apply advanced features if enabled
        if self.use_advanced_features and hasattr(self, 'advanced_engineer'):
            try:
                # Convert to DataFrame
                df = pd.DataFrame([features])
                
                # Apply advanced engineering (only high priority features)
                df_extended = self.advanced_engineer.engineer_all_features(
                    df, 
                    priority='high'
                )
                
                # Convert back to dict
                features = df_extended.iloc[0].to_dict()
            except Exception as e:
                print(f"Warning: Advanced feature engineering failed: {e}")
        
        return features
    
    def get_airport_stats(self, airport_code: str) -> Optional[Dict]:
        """Get airport statistics for display purposes"""
        return self.airport_cache.get(airport_code.upper())
    
    def get_carrier_stats(self, carrier_code: str) -> Optional[Dict]:
        """Get carrier statistics for display purposes"""
        return self.carrier_cache.get(carrier_code.upper())
