"""
Streamlit Frontend - Lightweight Display Layer with Dropdowns
All business logic delegated to modeling module
"""
from __future__ import annotations

import os
from datetime import date
import streamlit as st
import pandas as pd

try:
    from serpapi import GoogleSearch
except Exception:
    GoogleSearch = None

from utils import parse_flight_number
from modeling.model_loader import create_predictor
from acquisition.getSERP import search_flight_comprehensive


# Configuration
ARTIFACT_DIR = "../OUTPUTS/improved_model"  # Use improved model by default

# Airline mappings for user-friendly display
AIRLINE_MAPPINGS = {
            "American Airlines": "AA",
            "Delta Air Lines": "DL", 
            "United Airlines": "UA",
            "Southwest Airlines": "WN",
            "JetBlue Airways": "B6",
            "Alaska Airlines": "AS",
            "Hawaiian Airlines": "HA",
            "Frontier Airlines": "F9",
            "Spirit Airlines": "NK",
            "Allegiant Air": "G4",
            "Sun Country Airlines": "SY",
            "Breeze Airways": "MX",
            "Avelo Airlines": "XP",
            "Cape Air": "9K",
            "Republic Airways": "YX",
            "SkyWest Airlines": "OO",
            "Envoy Air": "MQ",
            "Mesa Airlines": "YV",
            "Endeavor Air": "9E",
            "PSA Airlines": "OH"
        }
        


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Load predictor once and cache"""
    try:
        predictor, metadata = create_predictor(ARTIFACT_DIR)
        if predictor is None:
            st.error(f"Failed to load predictor from: {ARTIFACT_DIR}")
            st.error(f"Current working directory: {os.getcwd()}")
            st.error(f"Looking for OUTPUTS at: {os.path.abspath(ARTIFACT_DIR)}")
        return predictor, metadata
    except Exception as e:
        st.error(f"Error loading predictor: {e}")
        return None, None


def main():
    st.set_page_config(
        page_title="Flight Delay Prediction", 
        
        layout="centered"
    )
    
    st.title("Flight Delay Prediction")
    st.caption("ML-powered delay prediction using BTS historical data")
    
    # Load predictor
    predictor, metadata = load_predictor()
    
    # Sidebar: Model info
    with st.sidebar:
        st.subheader("Model Status")
        if predictor is None:
            st.error("Model not loaded. Run training pipeline first.")
            st.caption(f"Looking in: {ARTIFACT_DIR}")
            st.caption(f"Working dir: {os.getcwd()}")
        else:
            st.success(f"✓ {metadata.get('best_model', 'Model')} loaded")
            st.metric("Model MAE", f"{metadata.get('selected_mae', 0):.1f} min")
            st.caption(f"Trained on {metadata.get('n_train', 0):,} flights")
            st.caption(f"Loaded from: {ARTIFACT_DIR}")
        
        st.markdown("---")
        st.subheader("Search History")
        history_count = len(st.session_state.get("search_history", []))
        st.metric("Saved Predictions", f"{history_count}")
        if history_count > 0:
            st.caption("View in Search History tab")
        
        st.markdown("---")
        st.subheader("SERP API Key")
        api_key = st.text_input(
            "API Key (optional)", 
            value=os.getenv("SERPAPI_API_KEY", ""),
            type="password"
        )
        if api_key:
            st.session_state["SERPAPI_API_KEY"] = api_key
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Predict", "Analytics", "Search History"])
    
    with tab1:
        # Check if we need to reload a previous prediction
        if "reload_flight" in st.session_state:
            reload_entry = st.session_state["reload_flight"]
            st.info(f"🔄 Reloading: {reload_entry['flight_number']} ({reload_entry['origin']} → {reload_entry['destination']})")
            del st.session_state["reload_flight"]
        
        # Input section with dropdowns
        st.subheader("Flight Information")
        
        # Airline and Flight Number Selection
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            # Airline dropdown
            selected_airline_name = st.selectbox(
                "Select Airline",
                options=list(AIRLINE_MAPPINGS.keys()),
                index=2,  # Default to United Airlines
                key="airline_selector"
            )
            selected_airline_code = AIRLINE_MAPPINGS[selected_airline_name]
        
        with col2:
            # Flight number manual input
            flight_number = st.text_input(
                "Flight Number", 
                value="1234",
                placeholder="e.g., 1234",
                key="flight_number_input"
            )
        
        with col3:
            flight_date = st.date_input("Date", value=date.today())
        
        # Combine airline and flight number
        flight_input = f"{selected_airline_code}{flight_number}"
        
        # Display the combined flight number
        st.info(f"🔍 Selected: **{flight_input}** ({selected_airline_name})")
        
        # SERP API search (optional)
        search_result = None
        if st.session_state.get("SERPAPI_API_KEY"):
            if st.button("Search Route", use_container_width=True):
                with st.spinner("Searching..."):
                    search_result = search_flight_comprehensive(
                        flight_number=flight_input,
                        flight_date=flight_date,
                        api_key=st.session_state["SERPAPI_API_KEY"],
                        include_status=True
                    )
                    if search_result and not search_result.get("error"):
                        st.success(
                            f"Found: {search_result.get('origin', 'N/A')} → {search_result.get('destination', 'N/A')}"
                        )
                    elif search_result and search_result.get("error"):
                        st.error(search_result["error"])
        
        # Manual input fields
        st.markdown("---")
        st.subheader("Route Information")
        
        col1, col2, col3 = st.columns(3)
        origin = col1.text_input(
            "Origin Airport (IATA)", 
            value=search_result.get("origin", "") if search_result else "",
            placeholder="e.g., LAX"
        )
        dest = col2.text_input(
            "Destination Airport (IATA)",
            value=search_result.get("destination", "") if search_result else "",
            placeholder="e.g., JFK"
        )
        dep_hour = col3.number_input(
            "Departure Hour", 
            min_value=0, 
            max_value=23, 
            value=13
        )
        
        # Predict button
        if st.button("Predict Delay", type="primary", use_container_width=True):
            if predictor is None:
                st.error("❌ Predictor not available")
                return

            if not origin or not dest:
                st.error("Please provide origin and destination airports")
                return
            
            # Generate prediction (all logic in modeling layer)
            try:
                with st.spinner("Generating prediction..."):
                    result = predictor.predict(
                        carrier=selected_airline_code,
                        origin=origin,
                        dest=dest,
                        flight_date=flight_date,
                        dep_hour=dep_hour
                    )
                
                # Display results
                st.markdown("---")
                st.subheader(f"✈️ {flight_input}: {origin} → {dest}")
                
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
                result_dict = result.to_dict()
                
                col1.metric(
                    "Predicted Delay",
                    f"{result_dict['predicted_delay']:.0f} min"
                )
                col2.metric(
                    "Confidence Range",
                    f"± {result_dict['confidence_mae']:.0f} min"
                )
                col3.metric(
                    "Est. Range",
                    f"{result_dict['lower_bound']:.0f}-{result_dict['upper_bound']:.0f} min"
                )
                
                # Key factors
                st.markdown("### 🔍 Key Factors")
                
                factors = {
                    f"**{selected_airline_code} Performance**": f"{result_dict['carrier_avg_delay']:.1f} min avg",
                    f"**{origin} Operations**": f"{result_dict['origin_delay_rate']:.1f}% delay rate",
                    f"**{dest} Operations**": f"{result_dict['dest_delay_rate']:.1f}% delay rate",
                }
                
                for factor, value in factors.items():
                    st.write(f"{factor}: {value}")
                
                # Additional context
                with st.expander("ℹ️ Understanding This Prediction"):
                    st.markdown("""
                    **How it works:**
                    - Historical carrier performance
                    - Airport operational efficiency
                    - Time-of-day patterns
                    - Day-of-week trends
                    
                    **Confidence band (±MAE)** represents the model's typical error range.
                    """)
                
                # Save to search history
                if "search_history" not in st.session_state:
                    st.session_state.search_history = []
                
                # Create history entry
                history_entry = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "flight_number": flight_input,
                    "airline": selected_airline_name,
                    "airline_code": selected_airline_code,
                    "origin": origin,
                    "destination": dest,
                    "flight_date": flight_date.strftime("%Y-%m-%d"),
                    "departure_hour": dep_hour,
                    "predicted_delay": result_dict['predicted_delay'],
                    "confidence_mae": result_dict['confidence_mae'],
                    "lower_bound": result_dict['lower_bound'],
                    "upper_bound": result_dict['upper_bound'],
                    "carrier_avg_delay": result_dict['carrier_avg_delay'],
                    "origin_delay_rate": result_dict['origin_delay_rate'],
                    "dest_delay_rate": result_dict['dest_delay_rate']
                }
                
                # Add to history (avoid duplicates)
                history_key = f"{flight_input}_{origin}_{dest}_{flight_date}_{dep_hour}"
                existing = [h for h in st.session_state.search_history 
                           if h.get("history_key") == history_key]
                
                if not existing:
                    history_entry["history_key"] = history_key
                    st.session_state.search_history.append(history_entry)
                    st.success("✅ Prediction saved to search history!")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab2:
        st.subheader("📊 Model Analytics")
        
        if predictor is None:
            st.warning("Load model to see analytics")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Model Type", metadata.get('best_model', 'N/A'))
                st.metric("Training Samples", f"{metadata.get('n_train', 0):,}")
            
            with col2:
                st.metric("Test Samples", f"{metadata.get('n_test', 0):,}")
                st.metric("MAE Performance", f"{metadata.get('selected_mae', 0):.2f} min")
            
            st.info("Model trained on BTS historical delay patterns")

    with tab3:
        st.subheader("Search History")
        
        if "search_history" not in st.session_state or not st.session_state.search_history:
            st.info("No search history yet. Make some predictions to see them here!")
        else:
            # Filter and sort options
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                filter_option = st.selectbox(
                    "Filter by",
                    ["All", "Recent (Last 10)", "This Week", "High Delays (>20 min)", "Low Delays (<10 min)"],
                    key="history_filter"
                )
            
            with col2:
                sort_option = st.selectbox(
                    "Sort by",
                    ["Most Recent", "Oldest First", "Flight Number", "Predicted Delay"],
                    key="history_sort"
                )
            
            with col3:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.search_history = []
                    st.rerun()
            
            # Apply filters
            history = st.session_state.search_history.copy()
            
            if filter_option == "Recent (Last 10)":
                history = history[-10:]
            elif filter_option == "This Week":
                week_ago = pd.Timestamp.now() - pd.Timedelta(days=7)
                history = [h for h in history if pd.Timestamp(h["timestamp"]) >= week_ago]
            elif filter_option == "High Delays (>20 min)":
                history = [h for h in history if h["predicted_delay"] > 20]
            elif filter_option == "Low Delays (<10 min)":
                history = [h for h in history if h["predicted_delay"] < 10]
            
            # Apply sorting
            if sort_option == "Most Recent":
                history = sorted(history, key=lambda x: x["timestamp"], reverse=True)
            elif sort_option == "Oldest First":
                history = sorted(history, key=lambda x: x["timestamp"])
            elif sort_option == "Flight Number":
                history = sorted(history, key=lambda x: x["flight_number"])
            elif sort_option == "Predicted Delay":
                history = sorted(history, key=lambda x: x["predicted_delay"], reverse=True)
            
            # Display history as cards
            st.write(f"**{len(history)} prediction(s) found**")
            
            for i, entry in enumerate(history):
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**✈️ {entry['flight_number']}** ({entry['airline']})")
                        st.caption(f"{entry['origin']} → {entry['destination']}")
                        st.caption(f"📅 {entry['flight_date']} at {entry['departure_hour']:02d}:00")
                    
                    with col2:
                        delay_color = "🟢" if entry['predicted_delay'] < 10 else "🟡" if entry['predicted_delay'] < 20 else "🔴"
                        st.markdown(f"{delay_color} **{entry['predicted_delay']:.0f} min**")
                        st.caption(f"±{entry['confidence_mae']:.0f} min confidence")
                    
                    with col3:
                        st.markdown(f"**Range:** {entry['lower_bound']:.0f}-{entry['upper_bound']:.0f} min")
                        st.caption(f"Carrier avg: {entry['carrier_avg_delay']:.1f} min")
                    
                    with col4:
                        if st.button("🔄", key=f"reload_{i}", help="Reload this prediction"):
                            # Pre-fill the form with this entry's data
                            st.session_state["reload_flight"] = entry
                            st.rerun()
                    
                    st.markdown("---")
            
            # Export options
            st.markdown("### 📥 Export History")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV export
                if history:
                    df = pd.DataFrame(history)
                csv = df.to_csv(index=False)
                
                st.download_button(
                        label="📊 Download as CSV",
                    data=csv,
                        file_name=f"flight_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # JSON export
                if history:
                    import json
                json_str = json.dumps(history, indent=2)
                
                st.download_button(
                        label="📄 Download as JSON",
                    data=json_str,
                        file_name=f"flight_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            # Statistics
            if history:
                st.markdown("### 📈 History Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_delay = sum(h["predicted_delay"] for h in history) / len(history)
                    st.metric("Average Delay", f"{avg_delay:.1f} min")
                
                with col2:
                    max_delay = max(h["predicted_delay"] for h in history)
                    st.metric("Highest Delay", f"{max_delay:.0f} min")
                
                with col3:
                    min_delay = min(h["predicted_delay"] for h in history)
                    st.metric("Lowest Delay", f"{min_delay:.0f} min")
                
                with col4:
                    total_predictions = len(history)
                    st.metric("Total Predictions", f"{total_predictions}")


if __name__ == "__main__":
    main()