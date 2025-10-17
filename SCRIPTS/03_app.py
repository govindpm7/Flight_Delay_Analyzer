from __future__ import annotations

import json
import os
from datetime import date
from typing import Optional

import pickle
import numpy as np
import pandas as pd
import streamlit as st

try:
    from serpapi import GoogleSearch  # optional, only used if key is provided
except Exception:  # keep app running without serpapi installed
    GoogleSearch = None

from utils import parse_flight_number, ConfidenceBand, confidence_badge_level


ARTIFACT_DIR = "OUTPUTS"
MODEL_PATH = os.path.join(ARTIFACT_DIR, "model.pkl")
META_PATH = os.path.join(ARTIFACT_DIR, "metadata.json")
LOOKUP_PATH = os.path.join(ARTIFACT_DIR, "flight_lookup.csv")
BTS_AIRPORT_PATH = os.path.join(ARTIFACT_DIR, "bts_lookup_airport.csv")
BTS_CARRIER_PATH = os.path.join(ARTIFACT_DIR, "bts_lookup_carrier.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_metadata():
    if not os.path.exists(META_PATH):
        return None
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_lookup():
    if not os.path.exists(LOOKUP_PATH):
        return None
    try:
        df = pd.read_csv(LOOKUP_PATH)
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_bts_airport_lookup():
    if not os.path.exists(BTS_AIRPORT_PATH):
        return None
    try:
        df = pd.read_csv(BTS_AIRPORT_PATH)
        return df
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_bts_carrier_lookup():
    if not os.path.exists(BTS_CARRIER_PATH):
        return None
    try:
        df = pd.read_csv(BTS_CARRIER_PATH)
        return df
    except Exception:
        return None


def search_flight_route(flight_number: str, api_key: str) -> Optional[dict]:
    """Search for flight route information using SERP API"""
    if not api_key or GoogleSearch is None:
        return None
    
    try:
        params = {
            "engine": "google",
            "q": f"{flight_number} flight route origin destination",
            "api_key": api_key,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Extract route information from search results
        if "organic_results" in results:
            for result in results["organic_results"][:3]:  # Check first 3 results
                title = result.get("title", "").lower()
                snippet = result.get("snippet", "").lower()
                
                # Look for airport codes in the content
                import re
                # Look for both IATA (3 letters) and ICAO (4 letters starting with K) codes
                iata_pattern = r'\b[A-Z]{3}\b'
                icao_pattern = r'\bK[A-Z]{3}\b'
                
                # Find IATA codes first (preferred)
                iata_airports = re.findall(iata_pattern, title + " " + snippet)
                # Find ICAO codes and convert to IATA
                icao_airports = re.findall(icao_pattern, title + " " + snippet)
                
                airports = []
                
                # Add IATA codes
                for code in iata_airports:
                    if code not in ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BUT', 'NOT', 'WHAT', 'ALL', 'WERE', 'WHEN', 'YOUR', 'SAID', 'EACH', 'WHICH', 'THEIR', 'TIME', 'WILL', 'ABOUT', 'IF', 'UP', 'OUT', 'MANY', 'THEN', 'THEM', 'THESE', 'SO', 'SOME', 'HER', 'WOULD', 'MAKE', 'LIKE', 'INTO', 'HIM', 'HAS', 'TWO', 'MORE', 'GO', 'NO', 'WAY', 'COULD', 'MY', 'THAN', 'FIRST', 'BEEN', 'CALL', 'WHO', 'ITS', 'NOW', 'FIND', 'LONG', 'DOWN', 'DAY', 'DID', 'GET', 'COME', 'MADE', 'MAY', 'PART']:
                        airports.append(code)
                
                # Convert ICAO to IATA (remove 'K' prefix)
                for code in icao_airports:
                    iata_code = code[1:]  # Remove 'K'
                    if iata_code not in airports:
                        airports.append(iata_code)
                
                if len(airports) >= 2:
                    return {
                        "origin": airports[0],
                        "destination": airports[1],
                        "title": result.get("title", ""),
                        "snippet": result.get("snippet", ""),
                        "link": result.get("link", "")
                    }
    except Exception as e:
        st.error(f"Search failed: {e}")
    
    return None


def save_search_history(search_result: dict, flight_number: str):
    """Save search result to session state history"""
    if "search_history" not in st.session_state:
        st.session_state.search_history = []
    
    search_result["flight_number"] = flight_number
    search_result["timestamp"] = pd.Timestamp.now().isoformat()
    st.session_state.search_history.append(search_result)


def display_delay_cause_analysis(airport: str, carrier: str, bts_airport: pd.DataFrame, bts_carrier: pd.DataFrame):
    """Display BTS delay cause analysis for airport and carrier"""
    st.subheader("📊 Delay Cause Analysis (BTS Data)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Airport Analysis**")
        airport_data = bts_airport[bts_airport['ORIGIN'] == airport.upper()]
        if not airport_data.empty:
            data = airport_data.iloc[0]
            
            # Delay rates
            st.metric("Total Delay Rate", f"{data['total_delay_rate_origin']:.1%}")
            st.metric("Avg Delay Minutes", f"{data['avg_delay_minutes_origin']:.1f} min")
            
            # Delay causes
            causes = {
                'Carrier Issues': data['carrier_delay_rate_origin'],
                'Weather': data['weather_delay_rate_origin'],
                'NAS (Air Traffic)': data['nas_delay_rate_origin'],
                'Security': data['security_delay_rate_origin'],
                'Late Aircraft': data['late_aircraft_delay_rate_origin']
            }
            
            st.markdown("**Delay Causes:**")
            for cause, rate in causes.items():
                st.write(f"• {cause}: {rate:.1%}")
        else:
            st.info(f"No BTS data available for airport {airport}")
    
    with col2:
        st.markdown("**Carrier Analysis**")
        carrier_data = bts_carrier[bts_carrier['OP_CARRIER'] == carrier.upper()]
        if not carrier_data.empty:
            data = carrier_data.iloc[0]
            
            # Delay rates
            st.metric("Total Delay Rate", f"{data['total_delay_rate_origin']:.1%}")
            st.metric("Avg Delay Minutes", f"{data['avg_delay_minutes_origin']:.1f} min")
            
            # Delay causes
            causes = {
                'Carrier Issues': data['carrier_delay_rate_origin'],
                'Weather': data['weather_delay_rate_origin'],
                'NAS (Air Traffic)': data['nas_delay_rate_origin'],
                'Security': data['security_delay_rate_origin'],
                'Late Aircraft': data['late_aircraft_delay_rate_origin']
            }
            
            st.markdown("**Delay Causes:**")
            for cause, rate in causes.items():
                st.write(f"• {cause}: {rate:.1%}")
        else:
            st.info(f"No BTS data available for carrier {carrier}")


def display_delay_cause_insights(bts_airport: pd.DataFrame, bts_carrier: pd.DataFrame):
    """Display general delay cause insights from BTS data"""
    st.subheader("🔍 Delay Cause Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Top Delayed Airports**")
        top_airports = bts_airport.nlargest(5, 'total_delay_rate_origin')
        for _, row in top_airports.iterrows():
            st.write(f"• {row['ORIGIN']}: {row['total_delay_rate_origin']:.1%} delay rate")
    
    with col2:
        st.markdown("**Top Delayed Carriers**")
        top_carriers = bts_carrier.nlargest(5, 'total_delay_rate_origin')
        for _, row in top_carriers.iterrows():
            st.write(f"• {row['OP_CARRIER']}: {row['total_delay_rate_origin']:.1%} delay rate")
    
    # Overall delay cause breakdown
    st.markdown("**Overall Delay Cause Breakdown**")
    if not bts_airport.empty:
        avg_causes = {
            'Carrier Issues': bts_airport['carrier_delay_rate_origin'].mean(),
            'Weather': bts_airport['weather_delay_rate_origin'].mean(),
            'NAS (Air Traffic)': bts_airport['nas_delay_rate_origin'].mean(),
            'Security': bts_airport['security_delay_rate_origin'].mean(),
            'Late Aircraft': bts_airport['late_aircraft_delay_rate_origin'].mean()
        }
        
        # Create a simple bar chart
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 6))
        causes = list(avg_causes.keys())
        rates = list(avg_causes.values())
        
        bars = ax.bar(causes, rates, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57'])
        ax.set_ylabel('Average Delay Rate')
        ax.set_title('Average Delay Causes Across All Airports')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{rate:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)


def main():
    st.set_page_config(page_title="Flight Delay Prediction (MVP)", page_icon="✈️", layout="centered")
    st.title("Flight Delay Prediction (MVP)")
    st.caption("Historical-patterns baseline with SERP API route search.")

    model = load_model()
    meta = load_metadata()
    lookup = load_lookup()
    bts_airport = load_bts_airport_lookup()
    bts_carrier = load_bts_carrier_lookup()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Flight Prediction", "Delay Analysis", "Search History"])

    with st.sidebar:
        st.subheader("Model Status")
        if meta is None or model is None:
            st.error("Model artifacts not found. Run training first (see README).")
        else:
            st.success(f"Loaded model: {meta.get('best_model','?')} (MAE ~ {meta.get('selected_mae','?'):.1f} min)")
            st.caption(f"Train: {meta.get('n_train',0):,}  Test: {meta.get('n_test',0):,}")

        st.markdown("---")
        st.subheader("SERP API Configuration")
        default_key = os.getenv("SERPAPI_API_KEY", "")
        key = st.text_input("SERPAPI_API_KEY", value=default_key, type="password", help="Required for route search functionality.")
        if key:
            st.session_state["SERPAPI_API_KEY"] = key
            st.caption("Key set for this session.")
        else:
            st.caption("Provide a SERP API key for route search.")

        # Optional quick check
        if st.button("Test SERP Key", use_container_width=True):
            if not key:
                st.warning("No key provided.")
            elif GoogleSearch is None:
                st.error("serpapi package not installed. See requirements.txt")
            else:
                try:
                    params = {
                        "engine": "google",
                        "q": "site:google.com flights",
                        "api_key": key,
                    }
                    _ = GoogleSearch(params).get_dict()
                    st.success("SERP API key appears valid.")
                except Exception as e:
                    st.error(f"SERP API check failed: {e}")

    with tab1:
        st.subheader("Enter Flight Details")
        c1, c2 = st.columns([2, 1])
        with c1:
            flight_input = st.text_input("Flight number (e.g., AA1234)", value="UA1234")
        with c2:
            flight_date = st.date_input("Date", value=date.today())

        # Search for route information
        search_result = None
        if flight_input and st.session_state.get("SERPAPI_API_KEY"):
            if st.button("Search Route", use_container_width=True):
                with st.spinner("Searching for flight route..."):
                    search_result = search_flight_route(flight_input, st.session_state["SERPAPI_API_KEY"])
                    if search_result:
                        save_search_history(search_result, flight_input)
                        st.success(f"Found route: {search_result['origin']} → {search_result['destination']}")
                    else:
                        st.warning("No route information found. Please enter manually.")

        # Manual entry fields
        st.subheader("Flight Details")
        col1, col2, col3 = st.columns(3)
        origin = col1.text_input("Origin (IATA)", value=search_result.get("origin", "") if search_result else "")
        dest = col2.text_input("Destination (IATA)", value=search_result.get("destination", "") if search_result else "")
        dep_hour = col3.number_input("Departure hour (0-23)", min_value=0, max_value=23, value=13)

        predict_btn = st.button("Predict Delay")

        if predict_btn:
            if model is None:
                st.error("Model not available. Please run training.")
                return

            parsed = parse_flight_number(flight_input)
            if not parsed:
                st.warning("Invalid flight format. Expect like 'AA1234'.")
                return
            carrier, number = parsed

            # Build single-row feature frame matching training features
            route = f"{origin.upper()}-{dest.upper()}"
            dow = pd.Timestamp(flight_date).weekday()
            month = pd.Timestamp(flight_date).month
            is_weekend = 1 if dow in (5, 6) else 0

            # Get historical averages from lookup if available
            route_avg_delay = 0.0
            origin_avg_delay = 0.0
            airline_avg_delay = 0.0
            
            if lookup is not None:
                # Try to get route-specific averages
                route_data = lookup[lookup["route"] == route]
                if not route_data.empty:
                    route_avg_delay = route_data["avg_delay"].iloc[0] if "avg_delay" in route_data.columns else 0.0
                
                # Try to get origin-specific averages
                origin_data = lookup[lookup["ORIGIN"] == origin.upper()]
                if not origin_data.empty:
                    origin_avg_delay = origin_data["avg_delay"].iloc[0] if "avg_delay" in origin_data.columns else 0.0
                
                # Try to get airline-specific averages
                airline_data = lookup[lookup["OP_CARRIER"] == carrier]
                if not airline_data.empty:
                    airline_avg_delay = airline_data["avg_delay"].iloc[0] if "avg_delay" in airline_data.columns else 0.0

            # Get BTS delay cause features
            bts_features = {}
            if bts_airport is not None:
                airport_data = bts_airport[bts_airport['ORIGIN'] == origin.upper()]
                if not airport_data.empty:
                    data = airport_data.iloc[0]
                    bts_features.update({
                        'total_delay_rate_origin': data.get('total_delay_rate_origin', 0),
                        'carrier_delay_rate_origin': data.get('carrier_delay_rate_origin', 0),
                        'weather_delay_rate_origin': data.get('weather_delay_rate_origin', 0),
                        'nas_delay_rate_origin': data.get('nas_delay_rate_origin', 0),
                        'security_delay_rate_origin': data.get('security_delay_rate_origin', 0),
                        'late_aircraft_delay_rate_origin': data.get('late_aircraft_delay_rate_origin', 0),
                        'avg_delay_minutes_origin': data.get('avg_delay_minutes_origin', 0),
                        'avg_carrier_delay_origin': data.get('avg_carrier_delay_origin', 0),
                        'avg_weather_delay_origin': data.get('avg_weather_delay_origin', 0),
                        'avg_nas_delay_origin': data.get('avg_nas_delay_origin', 0),
                        'avg_security_delay_origin': data.get('avg_security_delay_origin', 0),
                        'avg_late_aircraft_delay_origin': data.get('avg_late_aircraft_delay_origin', 0),
                    })
            
            if bts_carrier is not None:
                carrier_data = bts_carrier[bts_carrier['OP_CARRIER'] == carrier.upper()]
                if not carrier_data.empty:
                    data = carrier_data.iloc[0]
                    bts_features.update({
                        'total_delay_rate_dest': data.get('total_delay_rate_origin', 0),
                        'carrier_delay_rate_dest': data.get('carrier_delay_rate_origin', 0),
                        'weather_delay_rate_dest': data.get('weather_delay_rate_origin', 0),
                        'nas_delay_rate_dest': data.get('nas_delay_rate_origin', 0),
                        'security_delay_rate_dest': data.get('security_delay_rate_origin', 0),
                        'late_aircraft_delay_rate_dest': data.get('late_aircraft_delay_rate_origin', 0),
                        'avg_delay_minutes_dest': data.get('avg_delay_minutes_origin', 0),
                        'avg_carrier_delay_dest': data.get('avg_carrier_delay_origin', 0),
                        'avg_weather_delay_dest': data.get('avg_weather_delay_origin', 0),
                        'avg_nas_delay_dest': data.get('avg_nas_delay_origin', 0),
                        'avg_security_delay_dest': data.get('avg_security_delay_origin', 0),
                        'avg_late_aircraft_delay_dest': data.get('avg_late_aircraft_delay_origin', 0),
                    })
            
            # Fill missing BTS features with zeros
            bts_feature_names = [
                'total_delay_rate_origin', 'carrier_delay_rate_origin', 'weather_delay_rate_origin',
                'nas_delay_rate_origin', 'security_delay_rate_origin', 'late_aircraft_delay_rate_origin',
                'avg_delay_minutes_origin', 'avg_carrier_delay_origin', 'avg_weather_delay_origin',
                'avg_nas_delay_origin', 'avg_security_delay_origin', 'avg_late_aircraft_delay_origin',
                'total_delay_rate_dest', 'carrier_delay_rate_dest', 'weather_delay_rate_dest',
                'nas_delay_rate_dest', 'security_delay_rate_dest', 'late_aircraft_delay_rate_dest',
                'avg_delay_minutes_dest', 'avg_carrier_delay_dest', 'avg_weather_delay_dest',
                'avg_nas_delay_dest', 'avg_security_delay_dest', 'avg_late_aircraft_delay_dest'
            ]
            
            for feature in bts_feature_names:
                if feature not in bts_features:
                    bts_features[feature] = 0.0

            # Get actual distance from lookup data
            distance = 500.0  # Default fallback
            if lookup is not None:
                # Try to get distance from route data
                route_data = lookup[lookup["route"] == route]
                if not route_data.empty and "DISTANCE" in route_data.columns:
                    distance = route_data["DISTANCE"].iloc[0]
                else:
                    # Try to get distance from origin-destination pair
                    origin_dest_data = lookup[(lookup["ORIGIN"] == origin.upper()) & (lookup["DEST"] == dest.upper())]
                    if not origin_dest_data.empty and "DISTANCE" in origin_dest_data.columns:
                        distance = origin_dest_data["DISTANCE"].iloc[0]

            # Create feature dictionary with all features
            feature_dict = {
                "dep_hour": int(dep_hour),
                "dow": int(dow),
                "month": int(month),
                "is_weekend": int(is_weekend),
                "DISTANCE": float(distance),
                "route": route,
                "ORIGIN": origin.upper(),
                "DEST": dest.upper(),
                "OP_CARRIER": carrier,
                "route_avg_delay": float(route_avg_delay),
                "origin_avg_delay": float(origin_avg_delay),
                "airline_avg_delay": float(airline_avg_delay),
            }
            
            # Add BTS features
            feature_dict.update(bts_features)
            
            X = pd.DataFrame([feature_dict])

            try:
                y_hat = float(model.predict(X)[0])
            except Exception as exc:
                st.error(f"Unable to generate prediction: {exc}")
                return

            mae = float(meta.get("selected_mae", 20.0)) if meta else 20.0
            band = ConfidenceBand(point_minutes=y_hat, half_width_minutes=mae)
            badge = confidence_badge_level(mae)

            st.markdown("---")
            st.subheader(f"Flight {flight_input.upper()} on {flight_date.isoformat()} — {origin.upper()} → {dest.upper()}")
            st.metric(label=f"Predicted Delay (± MAE)", value=f"{y_hat:.0f} min", delta=f"± {mae:.0f} min")
            st.caption(f"Confidence: {badge}")

            # Feature contributions (simple proxy): show input features in a table
            with st.expander("Inputs used"):
                st.dataframe(X.T, use_container_width=True)

            # Display delay cause analysis if BTS data is available
            if bts_airport is not None and bts_carrier is not None:
                display_delay_cause_analysis(origin, carrier, bts_airport, bts_carrier)

            st.info("This MVP uses historical patterns with SERP API route search and BTS delay cause data for enhanced accuracy.")

    with tab2:
        st.subheader("Delay Analysis")
        
        if bts_airport is None or bts_carrier is None:
            st.warning("BTS delay cause data not available. Please run the BTS data processing pipeline first.")
        else:
            display_delay_cause_insights(bts_airport, bts_carrier)
            
            # Interactive analysis
            st.markdown("---")
            st.subheader("🔍 Interactive Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Airport Analysis**")
                selected_airport = st.selectbox(
                    "Select Airport",
                    options=sorted(bts_airport['ORIGIN'].unique()),
                    key="airport_selector"
                )
                
                if selected_airport:
                    airport_data = bts_airport[bts_airport['ORIGIN'] == selected_airport].iloc[0]
                    st.metric("Total Delay Rate", f"{airport_data['total_delay_rate_origin']:.1%}")
                    st.metric("Avg Delay Minutes", f"{airport_data['avg_delay_minutes_origin']:.1f} min")
                    
                    # Delay cause breakdown
                    causes = {
                        'Carrier Issues': airport_data['carrier_delay_rate_origin'],
                        'Weather': airport_data['weather_delay_rate_origin'],
                        'NAS (Air Traffic)': airport_data['nas_delay_rate_origin'],
                        'Security': airport_data['security_delay_rate_origin'],
                        'Late Aircraft': airport_data['late_aircraft_delay_rate_origin']
                    }
                    
                    st.markdown("**Delay Causes:**")
                    for cause, rate in causes.items():
                        st.write(f"• {cause}: {rate:.1%}")
            
            with col2:
                st.markdown("**Carrier Analysis**")
                selected_carrier = st.selectbox(
                    "Select Carrier",
                    options=sorted(bts_carrier['OP_CARRIER'].unique()),
                    key="carrier_selector"
                )
                
                if selected_carrier:
                    carrier_data = bts_carrier[bts_carrier['OP_CARRIER'] == selected_carrier].iloc[0]
                    st.metric("Total Delay Rate", f"{carrier_data['total_delay_rate_origin']:.1%}")
                    st.metric("Avg Delay Minutes", f"{carrier_data['avg_delay_minutes_origin']:.1f} min")
                    
                    # Delay cause breakdown
                    causes = {
                        'Carrier Issues': carrier_data['carrier_delay_rate_origin'],
                        'Weather': carrier_data['weather_delay_rate_origin'],
                        'NAS (Air Traffic)': carrier_data['nas_delay_rate_origin'],
                        'Security': carrier_data['security_delay_rate_origin'],
                        'Late Aircraft': carrier_data['late_aircraft_delay_rate_origin']
                    }
                    
                    st.markdown("**Delay Causes:**")
                    for cause, rate in causes.items():
                        st.write(f"• {cause}: {rate:.1%}")

    with tab3:
        st.subheader("Search History")
        
        if "search_history" not in st.session_state or not st.session_state.search_history:
            st.info("No search history yet. Search for flight routes to see them here.")
        else:
            # Display search history
            history_df = pd.DataFrame(st.session_state.search_history)
            
            # Show recent searches
            st.dataframe(
                history_df[["flight_number", "origin", "destination", "timestamp"]].sort_values("timestamp", ascending=False),
                use_container_width=True
            )
            
            # Download button
            csv = history_df.to_csv(index=False)
            st.download_button(
                label="Download Search History as CSV",
                data=csv,
                file_name=f"flight_search_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Clear history button
            if st.button("Clear History", type="secondary"):
                st.session_state.search_history = []
                st.rerun()


if __name__ == "__main__":
    main()

