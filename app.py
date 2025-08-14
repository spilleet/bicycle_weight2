from flask import Flask, render_template, jsonify
from bike_reallocation_system import (
    fetch_realtime_bike_data,
    classify_stations,
    calculate_priority_scores,
    generate_reallocation_route
)
import pandas as pd

app = Flask(__name__)

def get_all_stations():
    """Fetch all available stations by making multiple API calls."""
    all_stations = []
    batch_size = 1000
    
    # Try multiple batches to get all stations
    for batch in range(10):  # Try up to 10 batches
        start_idx = batch * batch_size + 1
        end_idx = start_idx + batch_size - 1
        
        df = fetch_realtime_bike_data(start_idx, end_idx)
        
        if df.empty:
            break
            
        all_stations.append(df)
        
        # If we got less than batch_size, we've reached the end
        if len(df) < batch_size:
            break
    
    if all_stations:
        return pd.concat(all_stations, ignore_index=True)
    else:
        return pd.DataFrame()

@app.route('/')
def index():
    """Display the main page with bike station data and reallocation route."""
    try:
        # Fetch all real-time data
        stations_df = get_all_stations()
        
        if stations_df.empty:
            return render_template('error.html', message="No data available from API")
        
        # Filter out stations with 0 capacity
        stations_df = stations_df[stations_df['capacity'] > 0]
        
        # Classify and calculate priority scores
        classified_stations = classify_stations(stations_df)
        stations_with_scores = calculate_priority_scores(classified_stations)
        
        # Generate reallocation route with more stops
        depot_coords = (37.5650, 126.9770)
        route = generate_reallocation_route(stations_df, depot_coords, truck_capacity=20, max_stops=30)
        
        # Prepare station data for display
        station_data = []
        for _, station in stations_with_scores.iterrows():
            station_data.append({
                'id': station['station_id'],
                'name': station['station_name'],
                'current_bikes': int(station['current_bikes']),
                'capacity': int(station['capacity']),
                'optimal_stock': int(station['optimal_stock']),
                'type': station['station_type'],
                'priority_score': round(station['priority_score'], 3),
                'latitude': station['latitude'],
                'longitude': station['longitude']
            })
        
        # Sort by priority score
        station_data.sort(key=lambda x: x['priority_score'], reverse=True)
        
        # Prepare route data for display
        route_data = []
        total_time = 0
        total_bikes = 0
        
        for stop in route:
            total_time += stop['travel_time']
            if stop['action'] != 'return':
                total_bikes += stop['bikes_transferred']
            
            route_data.append({
                'stop_number': stop['stop_number'],
                'station_name': stop['station_name'],
                'action': stop['action'],
                'bikes_transferred': stop['bikes_transferred'],
                'truck_load_after': stop['truck_load_after'],
                'travel_time': round(stop['travel_time'], 2),
                'efficiency_score': round(stop['efficiency_score'], 3) if stop['efficiency_score'] > 0 else '-',
                'priority_score': round(stop['priority_score'], 3) if stop['priority_score'] > 0 else '-'
            })
        
        # Calculate summary statistics
        summary = {
            'total_stations': len(station_data),
            'pickup_stations': sum(1 for s in station_data if s['type'] == 'Pickup'),
            'dropoff_stations': sum(1 for s in station_data if s['type'] == 'Drop-off'),
            'balanced_stations': sum(1 for s in station_data if s['type'] == 'Balanced'),
            'total_stops': len(route) - 1,  # Exclude depot return
            'total_time': round(total_time, 2),
            'total_bikes_redistributed': total_bikes,
            'pickup_stops': sum(1 for s in route if s['action'] == 'pickup'),
            'dropoff_stops': sum(1 for s in route if s['action'] == 'dropoff')
        }
        
        return render_template('index.html', 
                             stations=station_data, 
                             route=route_data,
                             summary=summary)
    
    except Exception as e:
        return f"Error: {str(e)}", 500

@app.route('/api/stations')
def api_stations():
    """API endpoint to get station data in JSON format."""
    try:
        stations_df = get_all_stations()
        
        if stations_df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        stations_df = stations_df[stations_df['capacity'] > 0]
        classified_stations = classify_stations(stations_df)
        stations_with_scores = calculate_priority_scores(classified_stations)
        
        station_list = []
        for _, station in stations_with_scores.iterrows():
            station_list.append({
                'id': station['station_id'],
                'name': station['station_name'],
                'current_bikes': int(station['current_bikes']),
                'capacity': int(station['capacity']),
                'optimal_stock': int(station['optimal_stock']),
                'type': station['station_type'],
                'priority_score': round(station['priority_score'], 3),
                'coordinates': [station['latitude'], station['longitude']]
            })
        
        return jsonify(station_list)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/route')
def api_route():
    """API endpoint to get reallocation route in JSON format."""
    try:
        stations_df = get_all_stations()
        
        if stations_df.empty:
            return jsonify({'error': 'No data available'}), 404
        
        stations_df = stations_df[stations_df['capacity'] > 0]
        depot_coords = (37.5650, 126.9770)
        route = generate_reallocation_route(stations_df, depot_coords, truck_capacity=20, max_stops=30)
        
        route_list = []
        for stop in route:
            route_list.append({
                'stop_number': stop['stop_number'],
                'station_name': stop['station_name'],
                'action': stop['action'],
                'bikes_transferred': stop['bikes_transferred'],
                'truck_load_after': stop['truck_load_after'],
                'travel_time': round(stop['travel_time'], 2)
            })
        
        return jsonify(route_list)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002, host='0.0.0.0')