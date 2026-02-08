#!/usr/bin/env python3
"""
Domestic Flight Price Scraper for Indian Routes
================================================
This script fetches flight prices for the top 20 domestic routes in India
using a rolling window approach (30 days ahead) for price prediction model training.

Based on Top 20 Domestic Routes from India (2024-2025):
1. Delhi (DEL) → Mumbai (BOM)
2. Bengaluru (BLR) → Delhi (DEL)
3. Bengaluru (BLR) → Mumbai (BOM)
4. Hyderabad (HYD) → Delhi (DEL)
5. Hyderabad (HYD) → Mumbai (BOM)
6. Chennai (MAA) → Delhi (DEL)
7. Chennai (MAA) → Mumbai (BOM)
8. Kolkata (CCU) → Delhi (DEL)
9. Kolkata (CCU) → Mumbai (BOM)
10. Pune (PNQ) → Delhi (DEL)
11. Pune (PNQ) → Bengaluru (BLR)
12. Ahmedabad (AMD) → Mumbai (BOM)
13. Ahmedabad (AMD) → Delhi (DEL)
14. Goa (GOI) → Mumbai (BOM)
15. Goa (GOI) → Bengaluru (BLR)
16. Kochi (COK) → Bengaluru (BLR)
17. Jaipur (JAI) → Delhi (DEL)
18. Lucknow (LKO) → Delhi (DEL)
19. Patna (PAT) → Delhi (DEL)
20. Srinagar (SXR) → Delhi (DEL)
"""
from datetime import date


import requests
import json
import csv
import os
from datetime import datetime, timedelta
import random
import time
import logging
from typing import List, Dict, Tuple, Optional
import argparse
from pathlib import Path
Path("data").mkdir(exist_ok=True)



# API Configuration
API_URL = os.getenv("FLIGHT_API_URL")
if not API_URL:
    raise ValueError("FLIGHT_API_URL not set")

API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Content-Type": "application/json"
}


REQUEST_DELAY = 5

ROLLING_WINDOW_DAYS = 1


DOMESTIC_ROUTES = [
    
    ("DEL", "BOM"),  
    ("BLR", "DEL"),  
    ("BLR", "BOM"),  
    ("HYD", "DEL"),  
    ("HYD", "BOM"),  
    ("MAA", "DEL"),  
    ("MAA", "BOM"),  
    ("CCU", "DEL"),  
    ("CCU", "BOM"),  
    ("PNQ", "DEL"),  
    ("PNQ", "BLR"),  
    ("AMD", "BOM"),  
    ("AMD", "DEL"),  
    ("GOI", "BOM"), 
    ("GOI", "BLR"),  
    ("COK", "BLR"),  
    ("JAI", "DEL"),  
    ("LKO", "DEL"),  
    ("PAT", "DEL"),  
    ("SXR", "DEL"),  
    
   
    ("BOM", "DEL"),  # Mumbai → Delhi
    ("DEL", "BLR"),  # Delhi → Bengaluru
    ("BOM", "BLR"),  # Mumbai → Bengaluru
    ("DEL", "HYD"),  # Delhi → Hyderabad
    ("BOM", "HYD"),  # Mumbai → Hyderabad
    ("DEL", "MAA"),  # Delhi → Chennai
    ("BOM", "MAA"),  # Mumbai → Chennai
    ("DEL", "CCU"),  # Delhi → Kolkata
    ("BOM", "CCU"),  # Mumbai → Kolkata
    ("DEL", "PNQ"),  # Delhi → Pune
    ("BLR", "PNQ"),  # Bengaluru → Pune
    ("BOM", "AMD"),  # Mumbai → Ahmedabad
    ("DEL", "AMD"),  # Delhi → Ahmedabad
    ("BOM", "GOI"),  # Mumbai → Goa
    ("BLR", "GOI"),  # Bengaluru → Goa
    ("BLR", "COK"),  # Bengaluru → Kochi
    ("DEL", "JAI"),  # Delhi → Jaipur
    ("DEL", "LKO"),  # Delhi → Lucknow
    ("DEL", "PAT"),  # Delhi → Patna
    ("DEL", "SXR"),  # Delhi → Srinagar
]

# Airline type mapping (Budget vs Premium)
AIRLINE_TYPES = {
    "6E": "Budget",      # IndiGo
    "SG": "Budget",      # SpiceJet
    "G8": "Budget",      # GoAir/GoFirst
    "I5": "Budget",      # AirAsia India
    "QP": "Budget",      # Akasa Air
    "AI": "Premium",     # Air India
    "UK": "Premium",     # Vistara
    "IX": "Premium",     # Air India Express
    "S5": "Premium",     # Star Air
}



def setup_logging(log_file: str = 'domestic_flight_scraper.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()


def fetch_flight_data(origin: str, destination: str, departure_date: datetime) -> Optional[Dict]:
    """
    Fetch flight data for a specific route and date from the API.
    
    Args:
        origin: Origin airport code (e.g., 'DEL')
        destination: Destination airport code (e.g., 'BOM')
        departure_date: Date of departure
        
    Returns:
        API response as dictionary or None if request fails
    """
    payload = {
        "UserIp": "127.0.0.1",
        "Adult": 1,
        "Child": 0,
        "Infant": 0,
        "JourneyType": 1,
        "CabinClass": 1,
        "AirSegments": [
            {
                "Origin": origin,
                "Destination": destination,
                "PreferredTime": departure_date.strftime("%Y-%m-%dT00:00:00")
            }
        ],
        "DirectFlight": True,
        "PreferredCarriers": []
    }
    
    try:
        response = requests.post(API_URL, headers=API_HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for {origin}-{destination}: {e}")
        return None


def parse_flight_data(data: Dict, origin: str, destination: str, 
                      search_date: datetime, departure_date: datetime) -> List[Dict]:
    """
    Parse flight data from API response and extract relevant information.
    
    Args:
        data: Raw API response
        origin: Origin airport code
        destination: Destination airport code
        search_date: Date when search was performed
        departure_date: Flight departure date
        
    Returns:
        List of parsed flight dictionaries
    """
    flights = []
    
    if not data or 'Result' not in data or not data['Result']:
        return flights
    
    # Calculate days before departure
    days_before_departure = (departure_date.date() - search_date.date()).days
    
    try:
        for flight_group in data['Result']:
            for flight in flight_group:
                # Extract segment information
                if 'Segments' not in flight or not flight['Segments']:
                    continue
                
                segment = flight['Segments'][0][0]  # First segment (direct flight)
                
                # Get airline code
                airline_code = segment['Airline']['AirlineCode']
                
                # Determine airline type
                airline_type = AIRLINE_TYPES.get(airline_code, 'Unknown')
                
                # Basic flight information
                flight_info = {
                    'search_date': search_date.strftime("%Y-%m-%d"),
                    'search_timestamp': search_date.strftime("%Y-%m-%d %H:%M:%S"),
                    'departure_date': departure_date.strftime("%Y-%m-%d"),
                    'days_before_departure': days_before_departure,
                    'origin': origin,
                    'destination': destination,
                    'route': f"{origin}-{destination}",
                    'departure_time': segment['Origin']['DepartTime'],
                    'arrival_time': segment['Destination']['ArrivalTime'],
                    'duration_minutes': segment['Duration'],
                    'airline_code': airline_code,
                    'airline_name': segment['Airline']['AirlineName'],
                    'airline_type': airline_type,
                    'flight_number': segment['Airline']['FlightNumber'],
                    'aircraft': segment['Craft'],
                    'origin_terminal': segment['Origin'].get('Terminal', ''),
                    'destination_terminal': segment['Destination'].get('Terminal', ''),
                    # Time-based features
                    'departure_hour': int(segment['Origin']['DepartTime'].split('T')[1].split(':')[0]),
                    'is_weekend': departure_date.weekday() >= 5,
                    'day_of_week': departure_date.weekday(),
                    'month': departure_date.month,
                }
                
                # Extract fare information
                if 'FareList' in flight and flight['FareList']:
                    prices = [fare['PublishedPrice'] for fare in flight['FareList']]
                    flight_info['min_price'] = min(prices)
                    flight_info['max_price'] = max(prices)
                    flight_info['avg_price'] = sum(prices) / len(prices)
                    flight_info['price_range'] = max(prices) - min(prices)
                    flight_info['min_published_price'] = flight.get('MinPublishedPrice', min(prices))
                    
                    # Get details from the cheapest fare
                    cheapest_fare = min(flight['FareList'], key=lambda x: x['PublishedPrice'])
                    flight_info['cabin_class'] = cheapest_fare['CabinClass']
                    flight_info['is_refundable'] = cheapest_fare['IsRefundable']
                    flight_info['fare_type'] = cheapest_fare['FareType']
                    
                    # Baggage information
                    if cheapest_fare.get('SeatBaggage') and cheapest_fare['SeatBaggage'][0]:
                        baggage = cheapest_fare['SeatBaggage'][0][0]
                        flight_info['checkin_baggage'] = baggage.get('CheckIn', '')
                        flight_info['cabin_baggage'] = baggage.get('Cabin', '')
                        flight_info['seats_available'] = baggage.get('NoOfSeatAvailable', 0)
                    else:
                        flight_info['checkin_baggage'] = ''
                        flight_info['cabin_baggage'] = ''
                        flight_info['seats_available'] = 0
                else:
                    # Set default values if no fare information
                    flight_info.update({
                        'min_price': None,
                        'max_price': None,
                        'avg_price': None,
                        'price_range': None,
                        'min_published_price': None,
                        'cabin_class': '',
                        'is_refundable': None,
                        'fare_type': '',
                        'checkin_baggage': '',
                        'cabin_baggage': '',
                        'seats_available': 0
                    })
                
                flights.append(flight_info)
                
    except Exception as e:
        logger.error(f"Error parsing flight data for {origin}-{destination}: {e}")
    
    return flights



def save_to_csv(flights: List[Dict], filename: str = 'domestic_flights_data.csv'):
    """
    Save flight data to CSV file (append mode).
    
    Args:
        flights: List of flight dictionaries
        filename: Output CSV filename
    """
    if not flights:
        return
    
    file_exists = os.path.isfile(filename)
    
    # Define CSV columns
    fieldnames = [
        'search_date', 'search_timestamp', 'departure_date', 'days_before_departure',
        'origin', 'destination', 'route', 'departure_time', 'arrival_time',
        'duration_minutes', 'airline_code', 'airline_name', 'airline_type',
        'flight_number', 'aircraft', 'origin_terminal', 'destination_terminal',
        'departure_hour', 'is_weekend', 'day_of_week', 'month',
        'min_price', 'max_price', 'avg_price', 'price_range', 'min_published_price',
        'cabin_class', 'is_refundable', 'fare_type',
        'checkin_baggage', 'cabin_baggage', 'seats_available'
    ]
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(flights)


def save_aggregated_data(flights: List[Dict], filename: str = 'domestic_flights_aggregated.csv'):
    """
    Save aggregated flight data (route-level statistics) for time series modeling.
    
    Args:
        flights: List of flight dictionaries
        filename: Output CSV filename
    """
    if not flights:
        return
    
    from collections import defaultdict
    
    # Aggregate by route, departure_date, and search_date
    aggregated = defaultdict(lambda: {
        'prices': [],
        'airlines': set(),
        'flights_count': 0
    })
    
    for flight in flights:
        if flight['min_price'] is not None:
            key = (flight['route'], flight['search_date'], flight['departure_date'])
            aggregated[key]['prices'].append(flight['min_price'])
            aggregated[key]['airlines'].add(flight['airline_code'])
            aggregated[key]['flights_count'] += 1
            aggregated[key]['days_before_departure'] = flight['days_before_departure']
            aggregated[key]['is_weekend'] = flight['is_weekend']
            aggregated[key]['day_of_week'] = flight['day_of_week']
            aggregated[key]['month'] = flight['month']
    
    # Convert to list of records
    records = []
    for (route, search_date, departure_date), data in aggregated.items():
        if data['prices']:
            records.append({
                'route': route,
                'search_date': search_date,
                'departure_date': departure_date,
                'days_before_departure': data['days_before_departure'],
                'min_price': min(data['prices']),
                'max_price': max(data['prices']),
                'avg_price': sum(data['prices']) / len(data['prices']),
                'median_price': sorted(data['prices'])[len(data['prices']) // 2],
                'price_std': (sum((p - sum(data['prices']) / len(data['prices']))**2 for p in data['prices']) / len(data['prices'])) ** 0.5 if len(data['prices']) > 1 else 0,
                'num_airlines': len(data['airlines']),
                'airlines': ','.join(sorted(data['airlines'])),
                'flights_count': data['flights_count'],
                'is_weekend': data['is_weekend'],
                'day_of_week': data['day_of_week'],
                'month': data['month']
            })
    
    file_exists = os.path.isfile(filename)
    
    fieldnames = [
        'route', 'search_date', 'departure_date', 'days_before_departure',
        'min_price', 'max_price', 'avg_price', 'median_price', 'price_std',
        'num_airlines', 'airlines', 'flights_count',
        'is_weekend', 'day_of_week', 'month'
    ]
    
    with open(filename, 'a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerows(records)




def scrape_and_save_all_routes(departure_date: datetime, output_file: str = 'domestic_flights_data.csv',
                                aggregated_file: str = 'domestic_flights_aggregated.csv',
                                delay: int = REQUEST_DELAY) -> Tuple[int, int, int]:
    """
    Scrape flight data for all domestic routes for a specific departure date and save to CSV.
    
    Args:
        departure_date: Target departure date
        output_file: Output CSV filename for detailed data
        aggregated_file: Output CSV filename for aggregated data
        delay: Delay between API requests in seconds
        
    Returns:
        Tuple of (successful_routes, failed_routes, total_flights_saved)
    """
    search_date = datetime.now()
    
    logger.info(f"Starting flight data collection")
    logger.info(f"Search date: {search_date.strftime('%Y-%m-%d')}")
    logger.info(f"Departure date: {departure_date.strftime('%Y-%m-%d')}")
    logger.info(f"Days before departure: {(departure_date.date() - search_date.date()).days}")
    logger.info(f"Total routes to process: {len(DOMESTIC_ROUTES)}")
    
    all_flights = []
    successful_routes = 0
    failed_routes = 0
    total_flights_saved = 0
    
    for i, (origin, destination) in enumerate(DOMESTIC_ROUTES, 1):
        logger.info(f"Processing route {i}/{len(DOMESTIC_ROUTES)}: {origin} -> {destination}")
        
        # Fetch data from API
        data = fetch_flight_data(origin, destination, departure_date)
        
        if data:
            # Parse the response
            flights = parse_flight_data(data, origin, destination, search_date, departure_date)
            if flights:
                all_flights.extend(flights)
                successful_routes += 1
                logger.info(f"Found {len(flights)} flights for {origin}-{destination}")
            else:
                logger.warning(f"No flights found for {origin}-{destination}")
                failed_routes += 1
        else:
            failed_routes += 1
        
        # Add delay to avoid overwhelming the API
        time.sleep(delay)
        
        # Save data periodically (every 10 routes) to prevent data loss
        if i % 10 == 0 and all_flights:
            save_to_csv(all_flights, output_file)
            save_aggregated_data(all_flights, aggregated_file)
            total_flights_saved += len(all_flights)
            logger.info(f"[SAVED] {len(all_flights)} flights saved to {output_file}")
            all_flights = []
    
    # Save any remaining flights
    if all_flights:
        save_to_csv(all_flights, output_file)
        save_aggregated_data(all_flights, aggregated_file)
        total_flights_saved += len(all_flights)
        logger.info(f"[SAVED] Final {len(all_flights)} flights saved to {output_file}")
    
    logger.info(f"Data collection for {departure_date.strftime('%Y-%m-%d')} complete!")
    logger.info(f"Successful routes: {successful_routes}")
    logger.info(f"Failed routes: {failed_routes}")
    logger.info(f"Total flights saved: {total_flights_saved}")
    
    return successful_routes, failed_routes, total_flights_saved


def scrape_all_routes(departure_date: datetime, output_file: str = 'domestic_flights_data.csv',
                      delay: int = REQUEST_DELAY) -> Tuple[int, int]:
    """
    Scrape flight data for all domestic routes for a specific departure date.
    (Legacy function - calls scrape_and_save_all_routes internally)
    
    Args:
        departure_date: Target departure date
        output_file: Output CSV filename
        delay: Delay between API requests in seconds
        
    Returns:
        Tuple of (successful_routes, failed_routes)
    """
    today_str = datetime.now().strftime("%Y_%m_%d")
    aggregated_file = f"data/domestic_flights_aggregated_{today_str}.csv"


    successful, failed, _ = scrape_and_save_all_routes(departure_date, output_file, aggregated_file, delay)
    return successful, failed


def scrape_rolling_window(start_date: datetime = None,
                          days_ahead: int = ROLLING_WINDOW_DAYS, 
                          output_file: str = 'domestic_flights_data.csv',
                          delay: int = REQUEST_DELAY):
    """
    Scrape flight data using rolling window approach.
    This fetches data for all dates from start_date up to 'days_ahead' days in the future.
    Saves data to CSV after processing each departure date.
    
    Args:
        start_date: Starting date for scraping (default: today)
        days_ahead: Number of days ahead to fetch data for (default: 30)
        output_file: Output CSV filename
        delay: Delay between API requests in seconds
    """
    # Use today if no start_date provided
    if start_date is None:
        start_date = datetime.now().date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()
        
    today_str = datetime.now().strftime("%Y_%m_%d")
    aggregated_file = f"data/domestic_flights_aggregated_{today_str}.csv"


    
    # Calculate end date
    end_date = start_date + timedelta(days=days_ahead - 1)
    
    logger.info(f"=" * 60)
    logger.info(f"ROLLING WINDOW FLIGHT DATA COLLECTION")
    logger.info(f"=" * 60)
    logger.info(f"Start date: {start_date.strftime('%Y-%m-%d')}")
    logger.info(f"End date: {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Total days to scrape: {days_ahead}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    logger.info(f"Total routes per day: {len(DOMESTIC_ROUTES)}")
    logger.info(f"Total API calls expected: {len(DOMESTIC_ROUTES) * days_ahead}")
    logger.info(f"Estimated time: {(len(DOMESTIC_ROUTES) * days_ahead * delay) / 3600:.1f} hours (with {delay}s delay)")
    logger.info(f"Output files: {output_file}, {aggregated_file}")
    logger.info(f"=" * 60)
    
    total_successful = 0
    total_failed = 0
    total_flights_saved = 0
    
    for day_offset in range(days_ahead):
        departure_date = datetime.combine(start_date + timedelta(days=day_offset), datetime.min.time())
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing departure date: {departure_date.strftime('%Y-%m-%d')} (Day {day_offset + 1}/{days_ahead})")
        logger.info(f"{'='*50}")
        
        successful, failed, flights_count = scrape_and_save_all_routes(
            departure_date, output_file, aggregated_file, delay
        )
        total_successful += successful
        total_failed += failed
        total_flights_saved += flights_count
        
        # Log progress summary after each day
        logger.info(f"Day {day_offset + 1} complete - Success: {successful}, Failed: {failed}, Flights saved: {flights_count}")
        logger.info(f"Running total - Success: {total_successful}, Failed: {total_failed}, Total flights: {total_flights_saved}")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"ROLLING WINDOW COLLECTION COMPLETE!")
    logger.info(f"{'='*60}")
    logger.info(f"Total successful route fetches: {total_successful}")
    logger.info(f"Total failed route fetches: {total_failed}")
    if (total_successful + total_failed) > 0:
        logger.info(f"Success rate: {total_successful / (total_successful + total_failed) * 100:.1f}%")
    logger.info(f"Total flights saved: {total_flights_saved}")
    logger.info(f"Output files:")
    logger.info(f"  - Detailed data: {output_file}")
    logger.info(f"  - Aggregated data: {aggregated_file}")
    logger.info(f"{'='*60}")


def scrape_single_date(departure_date_str: str, output_file: str = 'domestic_flights_data.csv',
                       delay: int = REQUEST_DELAY):
    """
    Scrape flight data for a single departure date.
    
    Args:
        departure_date_str: Departure date string in YYYY-MM-DD format
        output_file: Output CSV filename
        delay: Delay between API requests in seconds
    """
    try:
        departure_date = datetime.strptime(departure_date_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid date format: {departure_date_str}. Use YYYY-MM-DD format.")
        return
    
    scrape_all_routes(departure_date, output_file, delay)



def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(
        description='Domestic Flight Price Scraper for Indian Routes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape data for next 30 days starting from today (default)
  python domestic_flight_scraper.py
  
  # Scrape data for 30 days starting from a specific date
  python domestic_flight_scraper.py --start-date 2025-12-15
  
  # Scrape with custom number of days
  python domestic_flight_scraper.py --start-date 2025-12-15 --days 15
  
  # Scrape with custom delay between requests
  python domestic_flight_scraper.py --start-date 2025-12-15 --delay 20
  
  # Specify custom output file
  python domestic_flight_scraper.py --start-date 2025-12-15 --output my_flights.csv
        """
    )
    
    parser.add_argument('--start-date', type=str, default=None,
                        help='Starting date for scraping (YYYY-MM-DD format). Default: today')
    parser.add_argument('--days', type=int, default=ROLLING_WINDOW_DAYS,
                        help=f'Number of days to scrape starting from start-date (default: {ROLLING_WINDOW_DAYS})')
    parser.add_argument('--output', type=str, default='domestic_flights_data.csv',
                        help='Output CSV filename (default: domestic_flights_data.csv)')
    parser.add_argument('--delay', type=int, default=REQUEST_DELAY,
                        help=f'Delay between API requests in seconds (default: {REQUEST_DELAY})')
    
    args = parser.parse_args()
    
    
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
            logger.info(f"Using start date: {start_date.strftime('%Y-%m-%d')}")
        except ValueError:
            logger.error(f"Invalid date format: {args.start_date}. Use YYYY-MM-DD format.")
            return
    else:
        start_date = datetime.now()
        logger.info(f"No start date specified. Using today: {start_date.strftime('%Y-%m-%d')}")
    

    from datetime import date
    from pathlib import Path
    
    # ensure data folder exists
    Path("data").mkdir(exist_ok=True)
    
    today_str = date.today().strftime("%Y_%m_%d")
    
    output_file = f"data/domestic_flights_{today_str}.csv"
    
    logger.info(f"Starting {args.days}-day scrape from {start_date.strftime('%Y-%m-%d')}")
    scrape_rolling_window(start_date, args.days, output_file, args.delay)
    

if __name__ == "__main__":
    main()
