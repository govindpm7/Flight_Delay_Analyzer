"""
Optimized flight search module using SERP API and Google Flights API
"""
import re
from datetime import date, datetime
from typing import Optional, Dict, List, Any
import streamlit as st

try:
    from serpapi import GoogleSearch
except Exception:
    GoogleSearch = None

# Global request log for tracking API usage
_request_log = []

def get_request_log():
    """Get the current request log"""
    return _request_log.copy()

def clear_request_log():
    """Clear the request log"""
    global _request_log
    _request_log = []

def _log_request(url: str, params: Dict[str, Any], response_status: int = None):
    """Log an API request"""
    global _request_log
    _request_log.append({
        "timestamp": datetime.now().isoformat(),
        "url": url,
        "params": params,
        "response_status": response_status
    })


# Compiled regex patterns for efficiency
FLIGHT_NUMBER_PATTERN = re.compile(r'^\s*([A-Z]{2,3})\s*-?\s*(\d{1,4})\s*$')
# Improved route pattern - more flexible to catch various formats
ROUTE_PATTERN = re.compile(r'\b([A-Z]{3})\s*(?:to|from|->|→|–|—|-)?\s*([A-Z]{3})\b', re.IGNORECASE)
# Alternative pattern for space-separated codes (e.g., "LAX JFK")
ROUTE_PATTERN_ALT = re.compile(r'\b([A-Z]{3})\s+([A-Z]{3})\b', re.IGNORECASE)
IATA_CODE_PATTERN = re.compile(r'^[A-Z]{3}$')


def validate_airport_code(code: str) -> bool:
    """Validate 3-letter IATA airport code format."""
    return bool(IATA_CODE_PATTERN.match(code.upper().strip()))


def parse_flight_number(flight_input: str) -> Optional[tuple]:
    """
    Parse flight number into carrier and number.
    
    Args:
        flight_input: Flight number string (e.g., "AA1224", "AA 1224")
        
    Returns:
        Tuple of (carrier, number) or None if invalid
    """
    match = FLIGHT_NUMBER_PATTERN.match(flight_input)
    if not match:
        return None
    carrier, number = match.group(1).upper(), match.group(2)
    return carrier, number


def safe_serp_call(params: dict, error_context: str = "") -> Optional[dict]:
    """
    Make SERP API call with error handling.
    
    Args:
        params: API parameters
        error_context: Context for error messages
        
    Returns:
        API response dict or None on error
    """
    if GoogleSearch is None:
        st.error("serpapi package not installed")
        return None
        
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        # Log the request
        _log_request("SERP API", params, 200 if "error" not in results else 400)
        
        if "error" in results:
            st.error(f"SERP API error ({error_context}): {results['error']}")
            return None
            
        return results
        
    except Exception as e:
        # Log failed request
        _log_request("SERP API", params, 500)
        st.error(f"SERP API call failed ({error_context}): {str(e)}")
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def find_flight_route_from_search(flight_number: str, api_key: str) -> Optional[Dict]:
    """
    Find flight route using Google Search (cached for 1 hour).
    
    Args:
        flight_number: Full flight number (e.g., "AA1224")
        api_key: SERP API key
        
    Returns:
        Dict with origin, destination, or None
    """
    params = {
        "engine": "google",
        "q": f"{flight_number} flight route",
        "api_key": api_key,
        "gl": "us",
        "hl": "en",
        "num": 5
    }
    
    results = safe_serp_call(params, "route search")
    if not results or "organic_results" not in results:
        return None
    
    # Extract route from search results
    for result in results["organic_results"][:5]:
        text = f"{result.get('title', '')} {result.get('snippet', '')}".upper()
        
        # Try primary route pattern first
        match = ROUTE_PATTERN.search(text)
        if not match:
            # Try alternative pattern for space-separated codes
            match = ROUTE_PATTERN_ALT.search(text)
        
        if match:
            origin, dest = match.group(1), match.group(2)
            if validate_airport_code(origin) and validate_airport_code(dest):
                return {
                    "origin": origin.upper(),
                    "destination": dest.upper(),
                    "source": "google_search",
                    "title": result.get("title", ""),
                    "link": result.get("link", "")
                }
    
    return None


def search_flights_between_airports(origin: str, dest: str, api_key: str) -> Optional[Dict]:
    """
    Search for flights between two specific airports using SERP API.
    Returns multiple airlines found on the route.
    
    Args:
        origin: Origin airport code (e.g., "ATL")
        dest: Destination airport code (e.g., "LAX")
        api_key: SERP API key
        
    Returns:
        Dict with available airlines and flights or None
    """
    # Validate airport codes
    if not validate_airport_code(origin) or not validate_airport_code(dest):
        return None
    
    params = {
        "engine": "google",
        "q": f"flights from {origin} to {dest} today airlines",
        "api_key": api_key,
        "gl": "us",
        "hl": "en",
        "num": 15
    }
    
    results = safe_serp_call(params, "airport-to-airport search")
    if not results or "organic_results" not in results:
        return None
    
    # Collect all airlines and flights found
    airlines_found = set()
    flights_found = []
    
    # Look for flight information in the results
    for result in results["organic_results"][:10]:
        text = f"{result.get('title', '')} {result.get('snippet', '')}".upper()
        
        # Look for flight numbers in the text
        flight_matches = re.findall(r'\b([A-Z]{2,3})\s*(\d{1,4})\b', text)
        for carrier, number in flight_matches:
            if len(carrier) >= 2:  # Valid airline codes
                airlines_found.add(carrier)
                
                # Extract departure time from the text
                departure_time = None
                departure_hour = None
                
                # Look for time patterns (e.g., "8:30 AM", "14:45", "2:15 PM")
                time_patterns = [
                    r'(\d{1,2}):(\d{2})\s*(AM|PM)',  # 8:30 AM, 2:15 PM
                    r'(\d{1,2}):(\d{2})',            # 14:45, 08:30
                    r'(\d{1,2})\s*(AM|PM)',         # 8 AM, 2 PM
                ]
                
                for pattern in time_patterns:
                    time_match = re.search(pattern, text, re.IGNORECASE)
                    if time_match:
                        if ':' in time_match.group(0):
                            # Format with colon
                            hour_str, minute_str = time_match.groups()[:2]
                            hour = int(hour_str)
                            minute = int(minute_str)
                            
                            # Convert to 24-hour format if AM/PM specified
                            if len(time_match.groups()) > 2 and time_match.group(3):
                                am_pm = time_match.group(3).upper()
                                if am_pm == 'PM' and hour != 12:
                                    hour += 12
                                elif am_pm == 'AM' and hour == 12:
                                    hour = 0
                            
                            departure_time = f"{hour:02d}:{minute:02d}"
                            departure_hour = hour
                        else:
                            # Format without colon
                            hour_str = time_match.group(1)
                            hour = int(hour_str)
                            
                            # Convert to 24-hour format if AM/PM specified
                            if len(time_match.groups()) > 1 and time_match.group(2):
                                am_pm = time_match.group(2).upper()
                                if am_pm == 'PM' and hour != 12:
                                    hour += 12
                                elif am_pm == 'AM' and hour == 12:
                                    hour = 0
                            
                            departure_time = f"{hour:02d}:00"
                            departure_hour = hour
                        break
                
                flights_found.append({
                    "carrier": carrier,
                    "number": number,
                    "flight_number": f"{carrier}{number}",
                    "title": result.get("title", ""),
                    "link": result.get("link", ""),
                    "departure_time": departure_time,
                    "departure_hour": departure_hour
                })
    
    # If we found airlines, return them
    if airlines_found:
        return {
            "origin": origin.upper(),
            "destination": dest.upper(),
            "source": "airport_search",
            "route_found": True,
            "available_airlines": sorted(list(airlines_found)),
            "flights_found": flights_found[:10],  # Limit to 10 flights
            "airline_count": len(airlines_found)
        }
    
    # If no specific flights found, return route info with common airlines
    common_airlines = ["AA", "DL", "UA", "WN", "B6", "AS", "F9", "NK"]
    return {
        "origin": origin.upper(),
        "destination": dest.upper(),
        "source": "airport_search",
        "route_found": True,
        "available_airlines": common_airlines,
        "flights_found": [],
        "airline_count": len(common_airlines),
        "message": f"Route {origin}-{dest} found, showing common airlines"
    }


def get_google_flights_data(origin: str, dest: str, flight_date: str, 
                            api_key: str, carrier: str = None) -> Optional[Dict]:
    """
    Get flight details from Google Flights API.
    
    Args:
        origin: 3-letter IATA code
        dest: 3-letter IATA code
        flight_date: Date in YYYY-MM-DD format
        api_key: SERP API key
        carrier: Optional airline code to filter
        
    Returns:
        Dict with flight details or None
    """
    params = {
        "engine": "google_flights",
        "departure_id": origin.upper(),
        "arrival_id": dest.upper(),
        "outbound_date": flight_date,
        "type": "2",  # One way
        "currency": "USD",
        "hl": "en",
        "gl": "us",
        "api_key": api_key
    }
    
    # Add carrier filter if provided
    if carrier:
        params["include_airlines"] = carrier.upper()
    
    results = safe_serp_call(params, "Google Flights")
    if not results:
        return None
    
    # Parse flight results
    flights = results.get("best_flights", []) or results.get("other_flights", [])
    
    if not flights:
        return {
            "origin": origin.upper(),
            "destination": dest.upper(),
            "source": "google_flights_no_results",
            "message": "No flights found for this route on this date"
        }
    
    # Get first flight (most relevant)
    flight = flights[0]
    flight_legs = flight.get("flights", [])
    
    if not flight_legs:
        return None
    
    # Parse flight details
    first_leg = flight_legs[0]
    
    return {
        "origin": origin.upper(),
        "destination": dest.upper(),
        "source": "google_flights",
        "departure_time": first_leg.get("departure_airport", {}).get("time"),
        "arrival_time": first_leg.get("arrival_airport", {}).get("time"),
        "duration_minutes": first_leg.get("duration"),
        "airline": first_leg.get("airline"),
        "flight_number": first_leg.get("flight_number"),
        "airplane": first_leg.get("airplane"),
        "travel_class": first_leg.get("travel_class"),
        "price": flight.get("price"),
        "layovers": flight.get("layovers", []),
        "total_duration": flight.get("total_duration"),
        "carbon_emissions": flight.get("carbon_emissions", {}),
        "extensions": first_leg.get("extensions", []),
        "raw_data": flight
    }


def get_flight_status(flight_number: str, flight_date: str, api_key: str) -> Optional[Dict]:
    """
    Get real-time flight status including delays using Google Search.
    
    Args:
        flight_number: Flight number (e.g., "AA1224")
        flight_date: Date string
        api_key: SERP API key
        
    Returns:
        Dict with status information or None
    """
    # Format query for current flight status
    params = {
        "engine": "google",
        "q": f"{flight_number} flight status {flight_date}",
        "api_key": api_key,
        "gl": "us",
        "hl": "en"
    }
    
    results = safe_serp_call(params, "flight status")
    if not results:
        return None
    
    status_info = {
        "flight_number": flight_number,
        "search_date": flight_date,
        "source": "google_search_status"
    }
    
    # Check for knowledge graph (Google flight card)
    if "knowledge_graph" in results:
        kg = results["knowledge_graph"]
        status_info.update({
            "title": kg.get("title", ""),
            "description": kg.get("description", ""),
            "status": kg.get("status", ""),
            "has_knowledge_graph": True
        })
    
    # Parse organic results for status keywords
    if "organic_results" in results:
        for result in results["organic_results"][:3]:
            snippet = result.get("snippet", "").upper()
            title = result.get("title", "").upper()
            combined = f"{title} {snippet}"
            
            # Look for status indicators
            if "DELAYED" in combined:
                status_info["status"] = "DELAYED"
                status_info["status_link"] = result.get("link")
            elif "CANCELLED" in combined or "CANCELED" in combined:
                status_info["status"] = "CANCELLED"
                status_info["status_link"] = result.get("link")
            elif "ON TIME" in combined:
                status_info["status"] = "ON TIME"
                status_info["status_link"] = result.get("link")
            elif "ARRIVED" in combined:
                status_info["status"] = "ARRIVED"
                status_info["status_link"] = result.get("link")
            elif "LANDED" in combined:
                status_info["status"] = "LANDED"
                status_info["status_link"] = result.get("link")
            
            # Extract times if present
            time_pattern = r'(\d{1,2}:\d{2}\s*(?:AM|PM)?)'
            times = re.findall(time_pattern, combined)
            if times and "times" not in status_info:
                status_info["times_found"] = times
    
    return status_info if len(status_info) > 3 else None


def search_flight_comprehensive(flight_number: str, flight_date: date, 
                                api_key: str, include_status: bool = True) -> Dict:
    """
    Comprehensive flight search combining route, Google Flights, and status.
    
    Args:
        flight_number: Flight number (e.g., "AA1224")
        flight_date: Date object
        api_key: SERP API key
        include_status: Whether to fetch real-time status
        
    Returns:
        Dict with all available flight information
    """
    if not api_key:
        return {"error": "No API key provided"}
    
    # Parse flight number
    parsed = parse_flight_number(flight_number)
    if not parsed:
        return {"error": "Invalid flight number format"}
    
    carrier, number = parsed
    date_str = flight_date.isoformat()
    
    result = {
        "flight_number": flight_number,
        "carrier": carrier,
        "number": number,
        "search_date": date_str,
        "timestamp": datetime.now().isoformat()
    }
    
    # Step 1: Find route
    route_info = find_flight_route_from_search(flight_number, api_key)
    if route_info:
        result.update(route_info)
        result["route_found"] = True
        
        # Step 2: Get Google Flights data for this route
        flights_data = get_google_flights_data(
            route_info["origin"],
            route_info["destination"],
            date_str,
            api_key,
            carrier
        )
        
        if flights_data:
            result["google_flights_data"] = flights_data
    else:
        result["route_found"] = False
        result["message"] = "Could not determine flight route from search results"
        result["debug_info"] = {
            "search_query": f"{flight_number} flight route",
            "suggestion": "Try providing origin and destination manually"
        }
    
    # Step 3: Get real-time status
    if include_status:
        status_info = get_flight_status(flight_number, date_str, api_key)
        if status_info:
            result["status_info"] = status_info
    
    return result


def extract_delay_minutes(status_info: Dict) -> Optional[int]:
    """
    Extract delay in minutes from status information.
    
    Args:
        status_info: Status dict from get_flight_status
        
    Returns:
        Delay in minutes (positive for late, negative for early) or None
    """
    if not status_info or "times_found" not in status_info:
        return None
    
    times = status_info["times_found"]
    if len(times) >= 2:
        # Assume first is scheduled, second is actual
        # This is simplified - would need more sophisticated parsing
        # for production use
        return None  # Placeholder for actual time difference calculation
    
    return None


def format_flight_display(flight_data: Dict) -> Dict:
    """
    Format flight data for display in Streamlit app.
    
    Args:
        flight_data: Result from search_flight_comprehensive
        
    Returns:
        Formatted dict for display
    """
    display = {
        "Flight": flight_data.get("flight_number", "N/A"),
        "Route": f"{flight_data.get('origin', '?')} → {flight_data.get('destination', '?')}",
        "Date": flight_data.get("search_date", "N/A")
    }
    
    # Add Google Flights info if available
    if "google_flights_data" in flight_data:
        gf = flight_data["google_flights_data"]
        display.update({
            "Departure": gf.get("departure_time", "N/A"),
            "Arrival": gf.get("arrival_time", "N/A"),
            "Duration": f"{gf.get('duration_minutes', 0)} min",
            "Airline": gf.get("airline", "N/A"),
            "Aircraft": gf.get("airplane", "N/A")
        })
    
    # Add status if available
    if "status_info" in flight_data:
        status = flight_data["status_info"]
        display["Status"] = status.get("status", "Unknown")
    
    return display
