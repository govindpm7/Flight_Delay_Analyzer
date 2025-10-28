"""
Streamlit Frontend - Lightweight Display Layer with Dropdowns
All business logic delegated to modeling module
"""
from __future__ import annotations

import os
from datetime import date
import streamlit as st
import pandas as pd
import plotly.express as px

try:
    from serpapi import GoogleSearch
except Exception:
    GoogleSearch = None

from utils import parse_flight_number
from modeling.model_loader import create_predictor
try:
    from acquisition.getSERP import search_flight_comprehensive, search_flights_between_airports
except ImportError as e:
    st.error(f"Import error: {e}")
    search_flight_comprehensive = None
    search_flights_between_airports = None
from airport_model_adapter import create_airport_predictor


# Configuration
ARTIFACT_DIR = "../OUTPUTS/airport_model"  # Use airport-level model by default

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
def load_predictor(_model_dir=ARTIFACT_DIR):
    """Load predictor once and cache"""
    try:
        # Try new airport model first, fallback to old model
        if "airport_model" in _model_dir:
            predictor, metadata = create_airport_predictor(_model_dir)
        else:
            predictor, metadata = create_predictor(_model_dir)
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
    predictor, metadata = load_predictor(ARTIFACT_DIR)
    
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
        
        # Define available airports and carriers (moved from later in code)
        available_airports = ['ATL', 'DEN', 'IAD', 'LAX']
        available_carriers = ['9E', 'AA', 'AS', 'B6', 'C5', 'DL', 'F9', 'G4', 'G7', 'HA', 'MQ', 'NK', 'OH', 'OO', 'QX', 'UA', 'WN', 'YV', 'YX', 'ZW']
        carrier_names = {
            '9E': 'Endeavor Air',
            'AA': 'American Airlines',
            'AS': 'Alaska Airlines', 
            'B6': 'JetBlue Airways',
            'C5': 'CommutAir',
            'DL': 'Delta Air Lines',
            'F9': 'Frontier Airlines',
            'G4': 'Allegiant Air',
            'G7': 'GoJet Airlines',
            'HA': 'Hawaiian Airlines',
            'MQ': 'Envoy Air',
            'NK': 'Spirit Airlines',
            'OH': 'PSA Airlines',
            'OO': 'SkyWest Airlines',
            'QX': 'Horizon Air',
            'UA': 'United Airlines',
            'WN': 'Southwest Airlines',
            'YV': 'Mesa Airlines',
            'YX': 'Republic Airways',
            'ZW': 'Air Wisconsin'
        }
        
        st.markdown("---")
        st.subheader("BTS Data Coverage")
        st.metric("Airports", f"{len(available_airports)}")
        st.metric("Carriers", f"{len(available_carriers)}")
        st.caption("All predictions use only carriers with BTS historical data")
        
        st.markdown("---")
        st.subheader("Available Carriers")
        st.caption("Carriers with BTS data:")
        for carrier in available_carriers[:10]:  # Show first 10
            st.caption(f"• {carrier_names.get(carrier, carrier)} ({carrier})")
        if len(available_carriers) > 10:
            st.caption(f"... and {len(available_carriers) - 10} more")
        
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
    tab1, tab2, tab3, tab4 = st.tabs(["Predict", "Analytics", "Search History", "SERP API"])
    
    with tab1:
        # Check if we need to reload a previous prediction
        if "reload_flight" in st.session_state:
            reload_entry = st.session_state["reload_flight"]
            st.info(f"🔄 Reloading: {reload_entry['flight_number']} ({reload_entry['origin']} → {reload_entry['destination']})")
            del st.session_state["reload_flight"]
        
        # Route-First Flight Search Workflow
        st.subheader("Smart Flight Search")
        st.caption("Select your route first, then find available flights with BTS data")
        
        # Airport Selection (Manual or Auto-filled)
        st.subheader("Route Selection")
        
        # Airport names mapping
        airport_names = {
            'ATL': 'Atlanta (ATL) - Hub',
            'DEN': 'Denver (DEN) - Hub', 
            'IAD': 'Washington Dulles (IAD) - Hub',
            'LAX': 'Los Angeles (LAX) - Hub'
        }
        
        col1, col2, col3 = st.columns(3)
        origin = col1.selectbox(
            "Origin Airport", 
            options=available_airports,
            format_func=lambda x: airport_names[x],
            index=0,
            key="origin_airport_main"
        )
        
        # Filter destination options to exclude the selected origin
        dest_options = [airport for airport in available_airports if airport != origin]
        dest_default_index = 0 if origin != available_airports[1] else 0
        
        dest = col2.selectbox(
            "Destination Airport",
            options=dest_options,
            format_func=lambda x: airport_names[x],
            index=dest_default_index,
            key="dest_airport_main"
        )
        flight_date = col3.date_input("Date", value=date.today(), key="flight_date_main")
        
        # Validation: Ensure origin and destination are different
        if origin == dest:
            st.error("❌ **Invalid Route**: Origin and destination cannot be the same airport!")
            st.warning("Please select different airports for origin and destination.")
            st.stop()
        
        # Display selected route
        st.info(f"🛫 Route: **{origin} → {dest}** on {flight_date.strftime('%Y-%m-%d')}")
        
        # Initialize session state for airline search results
        if "airline_search_results" not in st.session_state:
            st.session_state["airline_search_results"] = None
        if "selected_airline" not in st.session_state:
            st.session_state["selected_airline"] = None
        
        # Route-Based Flight Search Section
        st.subheader("Find Flights for This Route")
        
        # Step 1: Search for flights on this route
        if st.button("🔍 Find Available Flights", use_container_width=True, type="primary"):
            if st.session_state.get("SERPAPI_API_KEY"):
                if search_flights_between_airports is None:
                    st.error("❌ SERP API functions not available. Check import errors.")
                else:
                    with st.spinner(f"Searching for flights from {origin} to {dest}..."):
                        try:
                            # Search for flights between the selected airports
                            search_result = search_flights_between_airports(
                                origin=origin,
                                dest=dest, 
                                api_key=st.session_state["SERPAPI_API_KEY"]
                            )
                        except Exception as e:
                            st.error(f"Error searching for flights: {str(e)}")
                            search_result = None
                    
                    if search_result and search_result.get("route_found"):
                        # Filter to only show carriers with BTS data
                        serp_airlines = search_result.get("available_airlines", [])
                        bts_airlines = []
                        flights_with_bts = []
                        
                        for airline in serp_airlines:
                            if airline in available_carriers:
                                bts_airlines.append(airline)
                                # Find flights for this airline
                                airline_flights = [f for f in search_result.get("flights_found", []) if f.get("carrier") == airline]
                                flights_with_bts.extend(airline_flights)
                        
                        if bts_airlines:
                            st.session_state["route_flights"] = {
                                "available_airlines": bts_airlines,
                                "flights_found": flights_with_bts,
                                "origin": origin,
                                "destination": dest,
                                "bts_data_available": True,
                                "total_flights": len(flights_with_bts)
                            }
                            st.success(f"✅ Found {len(flights_with_bts)} flights with BTS data from {len(bts_airlines)} airlines")
                            
                            # Show filtered out airlines
                            filtered_out = [a for a in serp_airlines if a not in available_carriers]
                            if filtered_out:
                                st.info(f"ℹ️ Filtered out {len(filtered_out)} airlines without BTS data: {', '.join(filtered_out)}")
                        else:
                            st.warning(f"⚠️ No airlines from SERP API have BTS data for {origin} → {dest}")
                            st.info("💡 Try using 'Show Airlines with BTS Data' button instead")
                    else:
                        st.warning(f"❌ No flights found for {origin} → {dest}")
            else:
                st.warning("⚠️ SERP API key required for flight search. Add it in the sidebar.")
        
        # Step 2: Show available flights if found
        if "route_flights" in st.session_state:
            route_data = st.session_state["route_flights"]
            if route_data["origin"] == origin and route_data["destination"] == dest:
                st.subheader("Available Flights")
                
                # Group flights by airline
                flights_by_airline = {}
                for flight in route_data["flights_found"]:
                    carrier = flight.get("carrier")
                    if carrier not in flights_by_airline:
                        flights_by_airline[carrier] = []
                    flights_by_airline[carrier].append(flight)
                
                # Display flights by airline
                for carrier, flights in flights_by_airline.items():
                    airline_name = carrier_names.get(carrier, carrier)
                    with st.expander(f"✈️ {airline_name} ({carrier}) - {len(flights)} flights"):
                        for i, flight in enumerate(flights):
                            flight_num = flight.get("flight_number", f"{carrier}{i+1}")
                            departure_time = flight.get("departure_time", "Time TBD")
                            departure_hour = flight.get("departure_hour")
                            
                            # Create button text with departure time
                            button_text = f"Select {flight_num}"
                            if departure_time:
                                button_text += f" ({departure_time})"
                            
                            if st.button(button_text, key=f"select_{flight_num}"):
                                # Store selected flight info
                                st.session_state["selected_flight"] = {
                                    "flight_number": flight_num,
                                    "carrier": carrier,
                                    "origin": origin,
                                    "destination": dest,
                                    "title": flight.get("title", ""),
                                    "link": flight.get("link", ""),
                                    "departure_time": departure_time,
                                    "departure_hour": departure_hour
                                }
                                st.session_state["selected_airline"] = carrier
                                
                                # Make additional SERP API call to get specific flight departure time
                                if st.session_state.get("SERPAPI_API_KEY") and search_flight_comprehensive is not None:
                                    with st.spinner(f"Getting departure time for {flight_num}..."):
                                        try:
                                            # Search for specific flight details
                                            flight_details = search_flight_comprehensive(
                                                flight_number=flight_num,
                                                flight_date=flight_date,
                                                api_key=st.session_state["SERPAPI_API_KEY"],
                                                include_status=True
                                            )
                                            
                                            # Extract departure time from Google Flights data
                                            if flight_details.get("google_flights_data", {}).get("departure_time"):
                                                gf_departure = flight_details["google_flights_data"]["departure_time"]
                                                st.session_state["auto_fill_dep_hour"] = gf_departure
                                                st.session_state["selected_flight"]["departure_time"] = gf_departure
                                                st.session_state["selected_flight"]["departure_hour"] = gf_departure
                                                st.success(f"✅ Selected: {flight_num} ({airline_name}) - Departure: {gf_departure}")
                                            else:
                                                # Fallback to original time if available
                                                if departure_hour is not None:
                                                    st.session_state["auto_fill_dep_hour"] = departure_hour
                                                st.success(f"✅ Selected: {flight_num} ({airline_name}) - Departure: {departure_time or 'Time TBD'}")
                                        except Exception as e:
                                            st.warning(f"Could not get departure time for {flight_num}: {str(e)}")
                                            # Fallback to original time if available
                                            if departure_hour is not None:
                                                st.session_state["auto_fill_dep_hour"] = departure_hour
                                            st.success(f"✅ Selected: {flight_num} ({airline_name}) - Departure: {departure_time or 'Time TBD'}")
                                else:
                                    # Fallback to original time if available
                                    if departure_hour is not None:
                                        st.session_state["auto_fill_dep_hour"] = departure_hour
                                    st.success(f"✅ Selected: {flight_num} ({airline_name}) - Departure: {departure_time or 'Time TBD'}")
                                
                                st.rerun()
        
        # Fallback: Show BTS carriers if no SERP search
        if st.button("📊 Show Airlines with BTS Data", use_container_width=True):
            route_carriers = [carrier for carrier in available_carriers]
            st.session_state["airline_search_results"] = {
                "available_airlines": route_carriers,
                "airline_count": len(route_carriers),
                "origin": origin,
                "destination": dest,
                "bts_data_available": True
            }
            st.success(f"✅ Found {len(route_carriers)} airlines with BTS historical data for {origin} → {dest}")
        
        # Optional: SERP API integration for additional flight info
        if st.session_state.get("SERPAPI_API_KEY"):
            if st.button("🔍 Search Additional Flight Info (SERP API)", use_container_width=True):
                with st.spinner("Searching for additional flight information..."):
                    search_result = search_flights_between_airports(
                        origin=origin,
                        dest=dest, 
                        api_key=st.session_state["SERPAPI_API_KEY"]
                    )
                    
                    if search_result and search_result.get("route_found"):
                        # Filter SERP results to only include carriers with BTS data
                        serp_airlines = search_result.get("available_airlines", [])
                        filtered_airlines = []
                        
                        for airline in serp_airlines:
                            # Check if this airline exists in our BTS data
                            if airline in available_carriers:
                                filtered_airlines.append(airline)
                        
                        if filtered_airlines:
                            st.session_state["airline_search_results"] = {
                                "available_airlines": filtered_airlines,
                                "airline_count": len(filtered_airlines),
                                "origin": origin,
                                "destination": dest,
                                "bts_data_available": True,
                                "serp_source": True,
                                "original_serp_count": len(serp_airlines)
                            }
                            st.success(f"✅ Found {len(filtered_airlines)} airlines with BTS data (filtered from {len(serp_airlines)} SERP results)")
                            
                            # Show which airlines were filtered out
                            filtered_out = [a for a in serp_airlines if a not in available_carriers]
                            if filtered_out:
                                st.info(f"ℹ️ Filtered out {len(filtered_out)} airlines without BTS data: {', '.join(filtered_out)}")
                        else:
                            st.warning(f"⚠️ No airlines from SERP API have BTS data. Found {len(serp_airlines)} airlines but none match our BTS carriers.")
                            st.info("💡 Try using 'Show Airlines with BTS Data' button instead for guaranteed accurate predictions.")
                    else:
                        st.warning("No additional flight info found via SERP API")
        else:
            st.info("💡 **Tip**: Add SERP API key in sidebar for additional flight information")
        
        # Show airline selection if we have search results
        if st.session_state.get("airline_search_results"):
            search_results = st.session_state["airline_search_results"]
            available_airlines = search_results.get("available_airlines", [])
            
            if available_airlines:
                # Create airline display names using BTS carrier names
                airline_display_names = {}
                for code in available_airlines:
                    # Use BTS carrier names first, then fallback to AIRLINE_MAPPINGS
                    display_name = carrier_names.get(code)
                    if not display_name:
                        # Fallback to AIRLINE_MAPPINGS
                        for name, airline_code in AIRLINE_MAPPINGS.items():
                            if airline_code == code:
                                display_name = name
                                break
                    if not display_name:
                        display_name = f"{code} Airlines"  # Final fallback
                    airline_display_names[code] = display_name
                
                # Airline selection dropdown
                selected_airline_code = st.selectbox(
                    "Select Airline",
                    options=available_airlines,
                    format_func=lambda x: f"{airline_display_names.get(x, x)} ({x})",
                    key="airline_selector_main"
                )
                
                # Store selected airline
                st.session_state["selected_airline"] = selected_airline_code
                
                # Display selected airline with BTS data indicator
                st.info(f"✈️ Selected: **{airline_display_names.get(selected_airline_code, selected_airline_code)}** ({selected_airline_code})")
                
                # Show BTS data availability indicator
                if search_results.get("bts_data_available"):
                    if search_results.get("serp_source"):
                        st.success(f"✅ **SERP API + BTS Data** - Found {search_results.get('original_serp_count', 0)} airlines, showing {len(available_airlines)} with BTS data")
                    else:
                        st.success("✅ **BTS Historical Data Available** - Prediction will be highly accurate")
                else:
                    st.warning("⚠️ **Limited Data** - Prediction may be less accurate")
                
                # Show additional flight info if available
                if search_results.get("flights_found"):
                    with st.expander("📋 Available Flights", expanded=False):
                        for flight in search_results["flights_found"][:5]:  # Show first 5
                            st.write(f"• {flight['flight_number']} - {flight.get('title', 'Flight info')}")
            else:
                st.error("No airlines found for this route")
        
        # Display detailed SERP results if available
        if "last_search_result" in st.session_state and st.session_state["last_search_result"]:
            search_result = st.session_state["last_search_result"]
            
            with st.expander("🔍 SERP API Search Results", expanded=False):
                st.subheader("Flight Information")
                
                # Basic flight info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Origin", search_result.get('origin', 'N/A'))
                with col2:
                    st.metric("Destination", search_result.get('destination', 'N/A'))
                with col3:
                    st.metric("Route Found", "✅" if search_result.get('route_found') else "❌")
                
                # Flight status if available
                if search_result.get('flight_status'):
                    st.subheader("Real-time Status")
                    status = search_result['flight_status']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Status", status.get('status', 'N/A'))
                    with col2:
                        st.metric("Scheduled Departure", status.get('scheduled_departure', 'N/A'))
                    with col3:
                        st.metric("Actual Departure", status.get('actual_departure', 'N/A'))
                    with col4:
                        st.metric("Gate", status.get('gate', 'N/A'))
                
                # Google Flights data if available
                if search_result.get('google_flights_data'):
                    st.subheader("Google Flights Data")
                    flights_data = search_result['google_flights_data']
                    
                    if isinstance(flights_data, list) and flights_data:
                        # Display as a table
                        df_flights = pd.DataFrame(flights_data)
                        st.dataframe(df_flights, use_container_width=True)
                    else:
                        st.json(flights_data)
                
                # Debug information
                if search_result.get('debug_info'):
                    st.subheader("Debug Information")
                    st.json(search_result['debug_info'])
                
                # Raw search result (collapsible)
                with st.expander("Raw SERP Response"):
                    st.json(search_result)
        
        # Departure time selection
        st.markdown("---")
        st.subheader("Departure Time")
        
        # Auto-fill departure hour if available from selected flight
        default_dep_hour = 13
        if "auto_fill_dep_hour" in st.session_state:
            default_dep_hour = st.session_state["auto_fill_dep_hour"]
        
        if "selected_flight" in st.session_state:
            selected_flight = st.session_state["selected_flight"]
            departure_time = selected_flight.get("departure_time", "Time TBD")
            st.info(f"✈️ **Selected Flight**: {selected_flight['flight_number']} ({carrier_names.get(selected_flight['carrier'], selected_flight['carrier'])}) - Departure: {departure_time}")
            
            # Manual refresh button for departure time
            if st.session_state.get("SERPAPI_API_KEY") and search_flight_comprehensive is not None:
                if st.button("🔄 Refresh Departure Time", help="Get updated departure time for selected flight"):
                    with st.spinner(f"Getting departure time for {selected_flight['flight_number']}..."):
                        try:
                            flight_details = search_flight_comprehensive(
                                flight_number=selected_flight['flight_number'],
                                flight_date=flight_date,
                                api_key=st.session_state["SERPAPI_API_KEY"],
                                include_status=True
                            )
                            
                            if flight_details.get("google_flights_data", {}).get("departure_time"):
                                gf_departure = flight_details["google_flights_data"]["departure_time"]
                                st.session_state["auto_fill_dep_hour"] = gf_departure
                                st.session_state["selected_flight"]["departure_time"] = gf_departure
                                st.session_state["selected_flight"]["departure_hour"] = gf_departure
                                st.success(f"✅ Updated departure time: {gf_departure}")
                                st.rerun()
                            else:
                                st.warning("❌ Could not find departure time for this flight")
                        except Exception as e:
                            st.error(f"Error getting departure time: {str(e)}")
            
            if "auto_fill_dep_hour" in st.session_state:
                st.success(f"🕐 **Auto-filled**: Departure time from flight search")
        
        dep_hour = st.number_input(
            "Departure Hour (0-23)", 
            min_value=0, 
            max_value=23, 
            value=default_dep_hour
        )
        
        # Predict button
        if st.button("Predict Delay", type="primary", use_container_width=True):
            if predictor is None:
                st.error("❌ Predictor not available")
                return

            if not origin or not dest:
                st.error("Please provide origin and destination airports")
                return
            
            if origin == dest:
                st.error("❌ **Invalid Route**: Origin and destination cannot be the same airport!")
                st.warning("Please select different airports for origin and destination.")
                return
            
            # Get selected airline from the new workflow
            selected_airline_code = st.session_state.get("selected_airline")
            if not selected_airline_code:
                st.error("Please select an airline first by clicking 'Search Airlines for This Route'")
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
                st.subheader(f"{selected_airline_code} Flight: {origin} → {dest}")
                
                # Convert result to dict if needed
                result_dict = result if isinstance(result, dict) else result.to_dict()
                
                # Check if prediction is available
                if not result_dict.get('prediction_available', True):
                    st.error(f"❌ **No Prediction Available**: {result_dict.get('message', 'Cannot generate prediction.')}")
                    return
                
                # Show data availability information
                data_availability = result_dict.get('data_availability', {})
                available_data = [k for k, v in data_availability.items() if v]
                missing_data = [k for k, v in data_availability.items() if not v]
                
                if missing_data:
                    st.info(f"ℹ️ **Data Available**: {', '.join(available_data)} | **Missing**: {', '.join(missing_data)}")
                else:
                    st.success(f"✅ **Complete Data Available**: All historical data found for this route")
                
                # Main metrics
                col1, col2, col3 = st.columns(3)
                
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
                
                # Key factors - only show available data
                st.markdown("### 🔍 Key Factors")
                
                factors = {}
                
                # Only show carrier performance if data is available
                if data_availability.get('carrier', False) and result_dict.get('carrier_avg_delay'):
                    factors[f"**{selected_airline_code} Performance**"] = f"{result_dict['carrier_avg_delay']:.1f} min avg"
                
                # Only show origin airport performance if data is available
                if data_availability.get('origin', False) and result_dict.get('origin_delay_rate'):
                    factors[f"**{origin} Operations**"] = f"{result_dict['origin_delay_rate']:.1f}% delay rate"
                
                # Only show destination airport performance if data is available
                if data_availability.get('dest', False) and result_dict.get('dest_delay_rate'):
                    factors[f"**{dest} Operations**"] = f"{result_dict['dest_delay_rate']:.1f}% delay rate"
                
                if factors:
                    for factor, value in factors.items():
                        st.write(f"{factor}: {value}")
                else:
                    st.info("No specific performance data available for this route")
                
                # Additional context
                with st.expander("ℹ️ Understanding This Prediction"):
                    st.markdown(f"""
                    **How it works:**
                    - **Dynamic Prediction**: Only uses available historical data
                    - **Available Data**: {', '.join(available_data) if available_data else 'None'}
                    - **Missing Data**: {', '.join(missing_data) if missing_data else 'None'}
                    - Time-of-day and seasonal patterns
                    - Day-of-week trends
                    
                    **Confidence band (±MAE)** represents the model's typical error range.
                    
                    **Note**: This prediction is based on the specific data available for your route. 
                    Missing data components are excluded rather than using industry averages.
                    """)
                
                # Save to search history
                if "search_history" not in st.session_state:
                    st.session_state.search_history = []
                
                # Create flight number for history
                flight_number = f"{selected_airline_code}1234"  # Generate a flight number
                
                # Get airline display name
                selected_airline_name = None
                for name, airline_code in AIRLINE_MAPPINGS.items():
                    if airline_code == selected_airline_code:
                        selected_airline_name = name
                        break
                if not selected_airline_name:
                    selected_airline_name = f"{selected_airline_code} Airlines"  # Fallback
                
                # Create history entry
                history_entry = {
                    "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "flight_number": flight_number,
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
                history_key = f"{flight_number}_{origin}_{dest}_{flight_date}_{dep_hour}"
                existing = [h for h in st.session_state.search_history 
                           if h.get("history_key") == history_key]
                
                if not existing:
                    history_entry["history_key"] = history_key
                    st.session_state.search_history.append(history_entry)
                    st.success("✅ Prediction saved to search history!")
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with tab2:
        st.subheader("Model Analytics & EDA")
        
        if predictor is None:
            st.warning("Load model to see analytics")
        else:
            # Model Performance Overview
            st.subheader("Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Model Type", metadata.get('best_model', 'N/A'))
            with col2:
                st.metric("Training Samples", f"{metadata.get('n_train', 0):,}")
            with col3:
                st.metric("Test Samples", f"{metadata.get('n_test', 0):,}")
            with col4:
                st.metric("MAE Performance", f"{metadata.get('selected_mae', 0):.2f} min")
            
            st.info("Model trained on BTS historical delay patterns")
            
            # Summary statistics
            st.subheader("Dataset Summary")
            try:
                if os.path.exists("../OUTPUTS/processing/processed_bts_data.csv"):
                    bts_summary = pd.read_csv("../OUTPUTS/processing/processed_bts_data.csv")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Records", f"{len(bts_summary):,}")
                    with col2:
                        st.metric("Unique Carriers", f"{bts_summary['carrier'].nunique()}")
                    with col3:
                        st.metric("Unique Airports", f"{bts_summary['airport'].nunique()}")
                    with col4:
                        avg_delay = bts_summary['avg_delay_minutes'].mean()
                        st.metric("Avg Delay", f"{avg_delay:.1f} min")
            except Exception:
                pass
            
            # Load and display EDA data
            try:
                # Load BTS processed data
                bts_data_path = "../OUTPUTS/processing/processed_bts_data.csv"
                carrier_summary_path = "../OUTPUTS/processing/carrier_summary.csv"
                airport_summary_path = "../OUTPUTS/processing/airport_summary.csv"
                monthly_trends_path = "../OUTPUTS/processing/monthly_trends.csv"
                
                if os.path.exists(bts_data_path):
                    bts_data = pd.read_csv(bts_data_path)
                    
                    # Convert date column
                    bts_data['date'] = pd.to_datetime(bts_data['date'])
                    
                    # Create tabs for different EDA views
                    eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
                        "📈 Time Series Analysis", 
                        "🏢 Carrier Performance", 
                        "✈️ Airport Analysis", 
                        "🔍 PCA Analysis",
                        "🤖 Model Evaluation"
                    ])
                    
                    with eda_tab1:
                        st.subheader("📈 Monthly Delay Trends")
                        
                        # Interactive filters
                        col1, col2 = st.columns(2)
                        with col1:
                            show_volume = st.checkbox("Show Flight Volume", value=False)
                        with col2:
                            show_causes = st.checkbox("Show Delay Causes Breakdown", value=True)
                        
                        if os.path.exists(monthly_trends_path):
                            monthly_data = pd.read_csv(monthly_trends_path)
                            monthly_data['date'] = pd.to_datetime(monthly_data['date'])
                            
                            # Monthly delay rate trend
                            fig_delay_rate = px.line(
                                monthly_data, 
                                x='date', 
                                y='total_delay_rate',
                                title='Monthly Delay Rate Trend',
                                labels={'total_delay_rate': 'Delay Rate', 'date': 'Month'}
                            )
                            fig_delay_rate.update_layout(height=400)
                            st.plotly_chart(fig_delay_rate, use_container_width=True)
                            
                            # Monthly average delay minutes
                            fig_avg_delay = px.line(
                                monthly_data, 
                                x='date', 
                                y='avg_delay_minutes',
                                title='Monthly Average Delay (Minutes)',
                                labels={'avg_delay_minutes': 'Average Delay (min)', 'date': 'Month'}
                            )
                            fig_avg_delay.update_layout(height=400)
                            st.plotly_chart(fig_avg_delay, use_container_width=True)
                            
                            # Delay cause breakdown over time (conditional)
                            if show_causes:
                                delay_causes = ['carrier_delay_pct', 'weather_delay_pct', 'nas_delay_pct']
                                fig_causes = px.area(
                                    monthly_data, 
                                    x='date', 
                                    y=delay_causes,
                                    title='Delay Causes Over Time',
                                    labels={'value': 'Percentage', 'date': 'Month'}
                                )
                                fig_causes.update_layout(height=400)
                                st.plotly_chart(fig_causes, use_container_width=True)
                            
                            # Flight volume trend (conditional)
                            if show_volume:
                                fig_volume_trend = px.line(
                                    monthly_data, 
                                    x='date', 
                                    y='arr_flights',
                                    title='Monthly Flight Volume',
                                    labels={'arr_flights': 'Arriving Flights', 'date': 'Month'}
                                )
                                fig_volume_trend.update_layout(height=400)
                                st.plotly_chart(fig_volume_trend, use_container_width=True)
                        
                        # Flight volume vs delays
                        st.subheader("Flight Volume vs Delays")
                        fig_volume = px.scatter(
                            bts_data, 
                            x='arr_flights', 
                            y='total_delay_rate',
                            color='carrier',
                            title='Flight Volume vs Delay Rate by Carrier',
                            labels={'arr_flights': 'Arriving Flights', 'total_delay_rate': 'Delay Rate'}
                        )
                        fig_volume.update_layout(height=500)
                        st.plotly_chart(fig_volume, use_container_width=True)
                    
                    with eda_tab2:
                        st.subheader("🏢 Carrier Performance Analysis")
                        
                        if os.path.exists(carrier_summary_path):
                            carrier_data = pd.read_csv(carrier_summary_path)
                            
                            # Top carriers by delay rate
                            carrier_avg = carrier_data.groupby('carrier').agg({
                                'total_delay_rate': 'mean',
                                'avg_delay_minutes': 'mean',
                                'arr_flights': 'sum'
                            }).reset_index()
                            
                            # Delay rate by carrier
                            fig_carrier_delay = px.bar(
                                carrier_avg.sort_values('total_delay_rate', ascending=False).head(15),
                                x='carrier',
                                y='total_delay_rate',
                                title='Average Delay Rate by Carrier (Top 15)',
                                labels={'total_delay_rate': 'Average Delay Rate', 'carrier': 'Carrier Code'}
                            )
                            fig_carrier_delay.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_carrier_delay, use_container_width=True)
                            
                            # Average delay minutes by carrier
                            fig_carrier_minutes = px.bar(
                                carrier_avg.sort_values('avg_delay_minutes', ascending=False).head(15),
                                x='carrier',
                                y='avg_delay_minutes',
                                title='Average Delay Minutes by Carrier (Top 15)',
                                labels={'avg_delay_minutes': 'Average Delay (min)', 'carrier': 'Carrier Code'}
                            )
                            fig_carrier_minutes.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_carrier_minutes, use_container_width=True)
                            
                            # Delay causes by carrier
                            carrier_causes = carrier_data.groupby('carrier').agg({
                                'carrier_delay_pct': 'mean',
                                'weather_delay_pct': 'mean',
                                'nas_delay_pct': 'mean'
                            }).reset_index()
                            
                            fig_causes = px.bar(
                                carrier_causes.sort_values('carrier_delay_pct', ascending=False).head(10),
                                x='carrier',
                                y=['carrier_delay_pct', 'weather_delay_pct', 'nas_delay_pct'],
                                title='Delay Causes by Carrier (Top 10)',
                                labels={'value': 'Percentage', 'carrier': 'Carrier Code'}
                            )
                            fig_causes.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_causes, use_container_width=True)
                    
                    with eda_tab3:
                        st.subheader("Airport Performance Analysis")
                        
                        if os.path.exists(airport_summary_path):
                            airport_data = pd.read_csv(airport_summary_path)
                            
                            # Top airports by delay rate
                            airport_avg = airport_data.groupby('airport').agg({
                                'total_delay_rate': 'mean',
                                'avg_delay_minutes': 'mean',
                                'arr_flights': 'sum'
                            }).reset_index()
                            
                            # Delay rate by airport
                            fig_airport_delay = px.bar(
                                airport_avg.sort_values('total_delay_rate', ascending=False).head(15),
                                x='airport',
                                y='total_delay_rate',
                                title='Average Delay Rate by Airport (Top 15)',
                                labels={'total_delay_rate': 'Average Delay Rate', 'airport': 'Airport Code'}
                            )
                            fig_airport_delay.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_airport_delay, use_container_width=True)
                            
                            # Flight volume vs delay rate scatter
                            fig_airport_volume = px.scatter(
                                airport_avg,
                                x='arr_flights',
                                y='total_delay_rate',
                                size='avg_delay_minutes',
                                hover_name='airport',
                                title='Airport Performance: Volume vs Delay Rate',
                                labels={'arr_flights': 'Total Flights', 'total_delay_rate': 'Delay Rate'}
                            )
                            fig_airport_volume.update_layout(height=500)
                            st.plotly_chart(fig_airport_volume, use_container_width=True)
                            
                            # Delay causes by airport
                            airport_causes = airport_data.groupby('airport').agg({
                                'carrier_delay_pct': 'mean',
                                'weather_delay_pct': 'mean',
                                'nas_delay_pct': 'mean'
                            }).reset_index()
                            
                            fig_airport_causes = px.bar(
                                airport_causes.sort_values('carrier_delay_pct', ascending=False).head(10),
                                x='airport',
                                y=['carrier_delay_pct', 'weather_delay_pct', 'nas_delay_pct'],
                                title='Delay Causes by Airport (Top 10)',
                                labels={'value': 'Percentage', 'airport': 'Airport Code'}
                            )
                            fig_airport_causes.update_layout(height=500, xaxis_tickangle=-45)
                            st.plotly_chart(fig_airport_causes, use_container_width=True)
                    
                    with eda_tab4:
                        st.subheader("PCA Analysis Results")
                        
                        # Load PCA analysis data
                        pca_plots_dir = "../OUTPUTS/pca_analysis/plots"
                        pca_data_dir = "../OUTPUTS/pca_analysis"
                        
                        if os.path.exists(pca_data_dir):
                            # Explained variance
                            if os.path.exists(f"{pca_plots_dir}/explained_variance.png"):
                                st.subheader("Explained Variance")
                                st.image(f"{pca_plots_dir}/explained_variance.png", use_container_width=True)
                            
                            # Scree plot
                            if os.path.exists(f"{pca_data_dir}/scree_plot.png"):
                                st.subheader("📈 Scree Plot")
                                st.image(f"{pca_data_dir}/scree_plot.png", use_container_width=True)
                            
                            # Biplot
                            if os.path.exists(f"{pca_plots_dir}/biplot_pc1_pc2.png"):
                                st.subheader("Principal Components Biplot")
                                st.image(f"{pca_plots_dir}/biplot_pc1_pc2.png", use_container_width=True)
                            
                            # Feature importance heatmap
                            if os.path.exists(f"{pca_plots_dir}/feature_importance_heatmap.png"):
                                st.subheader("🔥 Feature Importance Heatmap")
                                st.image(f"{pca_plots_dir}/feature_importance_heatmap.png", use_container_width=True)
                            
                            # Loadings heatmap
                            if os.path.exists(f"{pca_data_dir}/loadings_heatmap.png"):
                                st.subheader("Component Loadings Heatmap")
                                st.image(f"{pca_data_dir}/loadings_heatmap.png", use_container_width=True)
                            
                            # Component loadings table
                            if os.path.exists(f"{pca_data_dir}/component_loadings.csv"):
                                st.subheader("Component Loadings")
                                loadings_df = pd.read_csv(f"{pca_data_dir}/component_loadings.csv", index_col=0)
                                
                                # Show top features for first few components
                                st.write("**Top Features by Principal Component:**")
                                for i in range(min(5, loadings_df.shape[1])):
                                    pc_name = f"PC{i+1}"
                                    if pc_name in loadings_df.columns:
                                        top_features = loadings_df[pc_name].abs().nlargest(10)
                                        st.write(f"**{pc_name}:** {', '.join(top_features.index.tolist())}")
                            
                            # Feature engineering report
                            if os.path.exists(f"{pca_data_dir}/feature_engineering_report.txt"):
                                st.subheader("📝 Feature Engineering Report")
                                with open(f"{pca_data_dir}/feature_engineering_report.txt", 'r') as f:
                                    report_content = f.read()
                                st.text(report_content)
                        else:
                            st.info("PCA analysis data not found. Run the PCA workflow to generate analysis plots.")
                    
                    with eda_tab5:
                        st.subheader("🤖 Model Evaluation & Performance")
                        
                        # Model evaluation images
                        model_eval_dir = "../OUTPUTS/improved_model"
                        
                        if os.path.exists(model_eval_dir):
                            # Random Forest evaluation
                            if os.path.exists(f"{model_eval_dir}/random_forest_evaluation.png"):
                                st.subheader("🌲 Random Forest Model Evaluation")
                                st.image(f"{model_eval_dir}/random_forest_evaluation.png", use_container_width=True)
                            
                            # Random Forest feature importance
                            if os.path.exists(f"{model_eval_dir}/random_forest_feature_importance.png"):
                                st.subheader("🌲 Random Forest Feature Importance")
                                st.image(f"{model_eval_dir}/random_forest_feature_importance.png", use_container_width=True)
                            
                            # XGBoost evaluation
                            if os.path.exists(f"{model_eval_dir}/xgboost_evaluation.png"):
                                st.subheader("XGBoost Model Evaluation")
                                st.image(f"{model_eval_dir}/xgboost_evaluation.png", use_container_width=True)
                            
                            # XGBoost feature importance
                            if os.path.exists(f"{model_eval_dir}/xgboost_feature_importance.png"):
                                st.subheader("XGBoost Feature Importance")
                                st.image(f"{model_eval_dir}/xgboost_feature_importance.png", use_container_width=True)
                            
                            # Model comparison metrics
                            st.subheader("Model Performance Comparison")
                            
                            # Create a comparison table if metadata is available
                            if metadata:
                                comparison_data = []
                                
                                # Add current model info
                                comparison_data.append({
                                    "Model": metadata.get('best_model', 'Current Model'),
                                    "MAE": f"{metadata.get('selected_mae', 0):.2f}",
                                    "Training Samples": f"{metadata.get('n_train', 0):,}",
                                    "Test Samples": f"{metadata.get('n_test', 0):,}",
                                    "Status": "✅ Active"
                                })
                                
                                # Add baseline comparison
                                comparison_data.append({
                                    "Model": "Baseline (Mean)",
                                    "MAE": "~15.0",
                                    "Training Samples": "N/A",
                                    "Test Samples": "N/A",
                                    "Status": "📊 Reference"
                                })
                                
                                comparison_df = pd.DataFrame(comparison_data)
                                st.dataframe(comparison_df, use_container_width=True)
                                
                                # Performance improvement
                                if metadata.get('selected_mae', 0) > 0:
                                    improvement = ((15.0 - metadata.get('selected_mae', 0)) / 15.0) * 100
                                    st.metric(
                                        "Performance Improvement vs Baseline", 
                                        f"{improvement:.1f}%",
                                        help="Improvement over baseline mean prediction"
                                    )
                        else:
                            st.info("Model evaluation images not found. Run the training pipeline to generate evaluation plots.")
                
                else:
                    st.warning("BTS processed data not found. Please run the data processing pipeline first.")
                    
            except Exception as e:
                st.error(f"Error loading EDA data: {str(e)}")
                st.info("Make sure the data processing pipeline has been run to generate the required files.")

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

    with tab4:
        st.subheader("SERP API Information")
        
        # API Key status
        if st.session_state.get("SERPAPI_API_KEY"):
            st.success("✅ SERP API Key is configured")
        else:
            st.warning("⚠️ No SERP API Key configured. Enter one in the sidebar to enable flight search.")
        
        # Show last search result if available
        if "last_search_result" in st.session_state and st.session_state["last_search_result"]:
            st.subheader("Last Search Result")
            search_result = st.session_state["last_search_result"]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Route Found", "✅" if search_result.get('route_found') else "❌")
            with col2:
                st.metric("Origin", search_result.get('origin', 'N/A'))
            with col3:
                st.metric("Destination", search_result.get('destination', 'N/A'))
            with col4:
                has_status = "✅" if search_result.get('flight_status') else "❌"
                st.metric("Real-time Status", has_status)
            
            # Detailed breakdown
            st.subheader("🔍 Search Details")
            
            # Flight information
            if search_result.get('origin') and search_result.get('destination'):
                st.info(f"**Flight Route:** {search_result['origin']} → {search_result['destination']}")
            
            # Message/status
            if search_result.get('message'):
                st.write(f"**Status:** {search_result['message']}")
            
            # Flight status details
            if search_result.get('flight_status'):
                st.subheader("✈️ Real-time Flight Status")
                status = search_result['flight_status']
                
                status_cols = st.columns(2)
                with status_cols[0]:
                    st.write("**Departure Information:**")
                    st.write(f"- Status: {status.get('status', 'N/A')}")
                    st.write(f"- Scheduled: {status.get('scheduled_departure', 'N/A')}")
                    st.write(f"- Actual: {status.get('actual_departure', 'N/A')}")
                    st.write(f"- Gate: {status.get('gate', 'N/A')}")
                
                with status_cols[1]:
                    st.write("**Arrival Information:**")
                    st.write(f"- Status: {status.get('arrival_status', 'N/A')}")
                    st.write(f"- Scheduled: {status.get('scheduled_arrival', 'N/A')}")
                    st.write(f"- Actual: {status.get('actual_arrival', 'N/A')}")
                    st.write(f"- Terminal: {status.get('terminal', 'N/A')}")
            
            # Google Flights data
            if search_result.get('google_flights_data'):
                st.subheader("📋 Google Flights Data")
                flights_data = search_result['google_flights_data']
                
                if isinstance(flights_data, list) and flights_data:
                    st.write(f"Found {len(flights_data)} flight options:")
                    df_flights = pd.DataFrame(flights_data)
                    st.dataframe(df_flights, use_container_width=True)
                else:
                    st.json(flights_data)
            
            # Debug information
            if search_result.get('debug_info'):
                st.subheader("🐛 Debug Information")
                debug_info = search_result['debug_info']
                st.write("**Search Query:**", debug_info.get('search_query', 'N/A'))
                st.write("**Suggestion:**", debug_info.get('suggestion', 'N/A'))
            
            # Raw response
            with st.expander("📄 Raw SERP API Response"):
                st.json(search_result)
        
        else:
            st.info("No search results yet. Use the 'Search Route' button in the Predict tab to search for flight information.")
        
        # API request history
        st.subheader("📊 API Request History")
        try:
            from acquisition.getSERP import get_request_log
            reqs = get_request_log()
            if reqs:
                st.write(f"**Total API requests made:** {len(reqs)}")
                
                # Show recent requests
                recent_reqs = reqs[-5:]  # Show last 5 requests
                for i, req in enumerate(recent_reqs, 1):
                    with st.expander(f"Request {len(reqs) - len(recent_reqs) + i} - {req.get('timestamp', 'Unknown time')}"):
                        st.write(f"**URL:** {req.get('url', 'Unknown')}")
                        st.write(f"**Status:** {req.get('response_status', 'Unknown')}")
                        st.write("**Parameters:**")
                        st.json(req.get('params', {}))
                
                if len(reqs) > 5:
                    st.caption(f"Showing last 5 of {len(reqs)} total requests")
            else:
                st.info("No API requests made yet. Use the 'Search Route' button to make your first request.")
        except Exception as e:
            st.warning(f"Could not load request history: {e}")
        
        # API usage information
        st.subheader("ℹ️ About SERP API")
        st.markdown("""
        **SERP API** provides real-time flight information including:
        
        - **Route Detection**: Automatically finds origin and destination airports
        - **Real-time Status**: Current flight status, delays, and gate information
        - **Google Flights Data**: Alternative flight options and pricing
        - **Comprehensive Search**: Searches multiple sources for accurate information
        
        **How it works:**
        1. Enter your flight number and date
        2. Click "Search Route" to query SERP API
        3. View detailed results in the expandable section
        4. Use the found route information for delay prediction
        
        **Note:** SERP API requires a valid API key. Get one at [serpapi.com](https://serpapi.com)
        """)


if __name__ == "__main__":
    main()