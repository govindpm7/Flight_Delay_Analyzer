"""
Model Loading Module
Handles loading of models and lookup data
"""
import os
import json
import pickle
import pandas as pd
from typing import Tuple, Optional

from .feature_engine import FeatureEngine
from .predictor import FlightDelayPredictor


class ModelArtifacts:
    """Container for all model artifacts"""
    
    def __init__(self, model, metadata: dict, 
                 bts_airport: pd.DataFrame, 
                 bts_carrier: pd.DataFrame,
                 lookup: Optional[pd.DataFrame] = None,
                 label_encoders: Optional[dict] = None):
        self.model = model
        self.metadata = metadata
        self.bts_airport = bts_airport
        self.bts_carrier = bts_carrier
        self.lookup = lookup
        self.label_encoders = label_encoders


def load_model_artifacts(artifact_dir: str = "OUTPUTS") -> Optional[ModelArtifacts]:
    """
    Load all model artifacts from directory
    
    Args:
        artifact_dir: Directory containing model artifacts
    
    Returns:
        ModelArtifacts object or None if loading fails
    """
    try:
        # Load model
        model_path = os.path.join(artifact_dir, "model.pkl")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load metadata
        meta_path = os.path.join(artifact_dir, "metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Load BTS data
        bts_airport = pd.read_csv(os.path.join(artifact_dir, "bts_lookup_airport.csv"))
        bts_carrier = pd.read_csv(os.path.join(artifact_dir, "bts_lookup_carrier.csv"))
        
        # Load optional lookup data
        lookup_path = os.path.join(artifact_dir, "flight_lookup.csv")
        lookup = pd.read_csv(lookup_path) if os.path.exists(lookup_path) else None
        
        # Load label encoders
        encoders_path = os.path.join(artifact_dir, "label_encoders.pkl")
        label_encoders = None
        if os.path.exists(encoders_path):
            with open(encoders_path, "rb") as f:
                label_encoders = pickle.load(f)
        
        return ModelArtifacts(model, metadata, bts_airport, bts_carrier, lookup, label_encoders)
        
    except Exception as e:
        print(f"Failed to load model artifacts: {e}")
        return None


def create_predictor(artifact_dir: str = "OUTPUTS") -> Tuple[Optional[FlightDelayPredictor], Optional[dict]]:
    """
    Create a ready-to-use predictor instance
    
    Args:
        artifact_dir: Directory containing model artifacts
    
    Returns:
        Tuple of (FlightDelayPredictor instance, metadata) or (None, None) if initialization fails
    """
    artifacts = load_model_artifacts(artifact_dir)
    
    if artifacts is None:
        return None, None
    
    # Create feature engine
    feature_engine = FeatureEngine(
        bts_airport_df=artifacts.bts_airport,
        bts_carrier_df=artifacts.bts_carrier,
        lookup_df=artifacts.lookup
    )
    
    # Get MAE from metadata
    mae = artifacts.metadata.get('selected_mae', 12.5)
    
    # Create predictor
    predictor = FlightDelayPredictor(
        model=artifacts.model,
        feature_engine=feature_engine,
        mae=mae,
        label_encoders=artifacts.label_encoders
    )
    
    return predictor, artifacts.metadata
