import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import math
import requests
import json

def get_travel_cost(start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> float:
    """
    모의 이동 비용 API 함수.
    유클리드 거리 기반으로 이동 시간을 시뮬레이션.
    
    Args:
        start_coords: 시작점의 (위도, 경도) 튜플
        end_coords: 끝점의 (위도, 경도) 튜플
    
    Returns:
        시뮬레이션된 이동 시간 (유클리드 거리 * 10)
    """
    lat1, lon1 = start_coords
    lat2, lon2 = end_coords
    
    euclidean_distance = math.sqrt((lat2 - lat1) ** 2 + (lon2 - lon1) ** 2)
    
    return euclidean_distance * 10

def classify_stations(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    최적 재고 수준을 사전 계산하고 스테이션을 분류.
    
    Args:
        stations_df: 스테이션 정보가 담긴 DataFrame
    
    Returns:
        분류 정보가 추가된 DataFrame
    """
    df = stations_df.copy()
    
    df['optimal_stock'] = df['capacity'] * 0.5
    
    df['deviation'] = df['current_bikes'] - df['optimal_stock']
    
    conditions = [
        df['deviation'] > 0,
        df['deviation'] < 0,
        df['deviation'] == 0
    ]
    choices = ['Pickup', 'Drop-off', 'Balanced']
    df['station_type'] = np.select(conditions, choices, default='Balanced')
    
    return df

def calculate_priority_scores(stations_df: pd.DataFrame, 
                             pickup_weights: Tuple[float, float] = (0.7, 0.3),
                             dropoff_weights: Tuple[float, float] = (0.8, 0.2)) -> pd.DataFrame:
    """
    모든 스테이션의 우선순위 점수 계산.
    
    Args:
        stations_df: 분류된 스테이션 정보가 담긴 DataFrame
        pickup_weights: 수거 우선순위 계산을 위한 (w1, w2) 가중치 튜플
        dropoff_weights: 배치 우선순위 계산을 위한 (w3, w4) 가중치 튜플
    
    Returns:
        우선순위 점수가 추가된 DataFrame
    """
    df = stations_df.copy()
    w1, w2 = pickup_weights
    w3, w4 = dropoff_weights
    
    pickup_mask = df['station_type'] == 'Pickup'
    df.loc[pickup_mask, 'saturation_score'] = (
        df.loc[pickup_mask, 'current_bikes'] / df.loc[pickup_mask, 'capacity']
    )
    df.loc[pickup_mask, 'surplus_bikes'] = df.loc[pickup_mask, 'deviation']
    
    dropoff_mask = df['station_type'] == 'Drop-off'
    df.loc[dropoff_mask, 'depletion_score'] = (
        1 - (df.loc[dropoff_mask, 'current_bikes'] / df.loc[dropoff_mask, 'capacity'])
    )
    df.loc[dropoff_mask, 'demand_bikes'] = -df.loc[dropoff_mask, 'deviation']
    
    if pickup_mask.any():
        surplus_95th = df.loc[pickup_mask, 'surplus_bikes'].quantile(0.95)
        df.loc[pickup_mask, 'surplus_bikes_capped'] = df.loc[pickup_mask, 'surplus_bikes'].clip(upper=surplus_95th)
        
        min_surplus = df.loc[pickup_mask, 'surplus_bikes_capped'].min()
        max_surplus = df.loc[pickup_mask, 'surplus_bikes_capped'].max()
        if max_surplus > min_surplus:
            df.loc[pickup_mask, 'normalized_surplus_score'] = (
                (df.loc[pickup_mask, 'surplus_bikes_capped'] - min_surplus) / 
                (max_surplus - min_surplus)
            )
        else:
            df.loc[pickup_mask, 'normalized_surplus_score'] = 0
        
        df.loc[pickup_mask, 'priority_score'] = (
            w1 * df.loc[pickup_mask, 'saturation_score'] + 
            w2 * df.loc[pickup_mask, 'normalized_surplus_score']
        )
    
    if dropoff_mask.any():
        demand_95th = df.loc[dropoff_mask, 'demand_bikes'].quantile(0.95)
        df.loc[dropoff_mask, 'demand_bikes_capped'] = df.loc[dropoff_mask, 'demand_bikes'].clip(upper=demand_95th)
        
        min_demand = df.loc[dropoff_mask, 'demand_bikes_capped'].min()
        max_demand = df.loc[dropoff_mask, 'demand_bikes_capped'].max()
        if max_demand > min_demand:
            df.loc[dropoff_mask, 'normalized_demand_score'] = (
                (df.loc[dropoff_mask, 'demand_bikes_capped'] - min_demand) / 
                (max_demand - min_demand)
            )
        else:
            df.loc[dropoff_mask, 'normalized_demand_score'] = 0
        
        df.loc[dropoff_mask, 'priority_score'] = (
            w3 * df.loc[dropoff_mask, 'depletion_score'] + 
            w4 * df.loc[dropoff_mask, 'normalized_demand_score']
        )
    
    df['priority_score'] = df['priority_score'].fillna(0)
    
    return df

def calculate_efficiency_score(station_row: pd.Series, 
                              current_coords: Tuple[float, float],
                              alpha: float = 0.6) -> float:
    """
    우선순위와 이동 비용을 결합한 효율성 점수 계산.
    
    Args:
        station_row: 스테이션 데이터가 담긴 Series
        current_coords: 현재 위치 좌표
        alpha: 우선순위 대 거리 가중치 (0-1)
    
    Returns:
        효율성 점수
    """
    station_coords = (station_row['latitude'], station_row['longitude'])
    travel_cost = get_travel_cost(current_coords, station_coords)
    
    if travel_cost == 0:
        travel_cost = 0.001
    
    distance_factor = 1 / travel_cost
    
    min_distance = 0.001
    max_distance = 100
    normalized_distance = (distance_factor - (1/max_distance)) / ((1/min_distance) - (1/max_distance))
    normalized_distance = np.clip(normalized_distance, 0, 1)
    
    efficiency_score = alpha * station_row['priority_score'] + (1 - alpha) * normalized_distance
    
    return efficiency_score

def generate_reallocation_route(stations_df: pd.DataFrame, 
                               depot_coords: Tuple[float, float],
                               truck_capacity: int = 20,
                               max_stops: int = 10) -> List[Dict]:
    """
    탐욕적 휴리스틱을 사용하여 최적화된 재배치 경로 생성.
    
    Args:
        stations_df: 스테이션 정보가 담긴 DataFrame
        depot_coords: 차고의 좌표 (위도, 경도)
        truck_capacity: 트럭이 운송할 수 있는 최대 자전거 수
        max_stops: 방문할 최대 스테이션 수
    
    Returns:
        경로 정보가 담긴 딕셔너리 리스트
    """
    df = classify_stations(stations_df)
    df = calculate_priority_scores(df)
    
    current_location = depot_coords
    truck_load = 0
    route = []
    visited_stations = set()
    
    for stop_num in range(max_stops):
        best_station = None
        best_efficiency = -1
        best_action = None
        
        for _, station in df.iterrows():
            if station['station_id'] in visited_stations:
                continue
            
            if station['station_type'] == 'Balanced':
                continue
            
            if station['station_type'] == 'Pickup':
                bikes_to_pickup = min(
                    int(station['surplus_bikes']) if 'surplus_bikes' in station else 0,
                    truck_capacity - truck_load
                )
                if bikes_to_pickup <= 0:
                    continue
                action = 'pickup'
                
            elif station['station_type'] == 'Drop-off':
                bikes_to_dropoff = min(
                    int(station['demand_bikes']) if 'demand_bikes' in station else 0,
                    truck_load
                )
                if bikes_to_dropoff <= 0:
                    continue
                action = 'dropoff'
            
            efficiency = calculate_efficiency_score(station, current_location)
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_station = station
                best_action = action
        
        if best_station is None:
            break
        
        station_coords = (best_station['latitude'], best_station['longitude'])
        travel_time = get_travel_cost(current_location, station_coords)
        
        if best_action == 'pickup':
            bikes_transferred = min(
                int(best_station['surplus_bikes']) if 'surplus_bikes' in best_station else 0,
                truck_capacity - truck_load
            )
            truck_load += bikes_transferred
            df.loc[df['station_id'] == best_station['station_id'], 'current_bikes'] -= bikes_transferred
        else:
            bikes_transferred = min(
                int(best_station['demand_bikes']) if 'demand_bikes' in best_station else 0,
                truck_load
            )
            truck_load -= bikes_transferred
            df.loc[df['station_id'] == best_station['station_id'], 'current_bikes'] += bikes_transferred
        
        route.append({
            'stop_number': stop_num + 1,
            'station_id': best_station['station_id'],
            'station_name': best_station['station_name'],
            'action': best_action,
            'bikes_transferred': bikes_transferred,
            'truck_load_after': truck_load,
            'travel_time': travel_time,
            'efficiency_score': best_efficiency,
            'priority_score': best_station['priority_score']
        })
        
        visited_stations.add(best_station['station_id'])
        current_location = station_coords
        
        df = classify_stations(df)
        df = calculate_priority_scores(df)
    
    return_time = get_travel_cost(current_location, depot_coords)
    route.append({
        'stop_number': len(route) + 1,
        'station_id': 'DEPOT',
        'station_name': 'Return to Depot',
        'action': 'return',
        'bikes_transferred': 0,
        'truck_load_after': truck_load,
        'travel_time': return_time,
        'efficiency_score': 0,
        'priority_score': 0
    })
    
    return route

def fetch_realtime_bike_data(start_idx: int = 1, end_idx: int = 100) -> pd.DataFrame:
    """
    서울시 공공자전거 API에서 실시간 자전거 스테이션 데이터 조회.
    
    Args:
        start_idx: API 페이지네이션 시작 인덱스
        end_idx: API 페이지네이션 끝 인덱스
    
    Returns:
        스테이션 정보가 담긴 DataFrame
    """
    # 실제 API 키 사용
    api_key = "6464716442737069363863566b466c"
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start_idx}/{end_idx}/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        if not response.text:
            print("Empty response from API")
            return pd.DataFrame()
            
        data = response.json()
        
        if 'rentBikeStatus' in data and 'row' in data['rentBikeStatus']:
            bike_data = data['rentBikeStatus']['row']
            
            stations_list = []
            for station in bike_data:
                stations_list.append({
                    'station_id': station.get('stationId', ''),
                    'station_name': station.get('stationName', ''),
                    'latitude': float(station.get('stationLatitude', 0)),
                    'longitude': float(station.get('stationLongitude', 0)),
                    'capacity': int(station.get('rackTotCnt', 0)),
                    'current_bikes': int(station.get('parkingBikeTotCnt', 0))
                })
            
            return pd.DataFrame(stations_list)
        else:
            print("API response does not contain expected data structure")
            print(f"Response: {data}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error fetching data from API: {e}")
        print(f"URL: {url}")
        return pd.DataFrame()

def main():
    """
    실시간 데이터를 이용한 자전거 재배치 시스템 시연용 메인 함수.
    """
    print("=" * 80)
    print("자전거 재배치 시스템 - 실시간 데이터")
    print("=" * 80)
    print("\n실시간 자전거 스테이션 데이터 조회 중...")
    
    stations_df = fetch_realtime_bike_data(1, 5)
    
    if stations_df.empty:
        print("데이터가 없습니다. 종료합니다...")
        return
    
    stations_df = stations_df[stations_df['capacity'] > 0]
    
    depot_coords = (37.5650, 126.9770)
    
    print(f"\n조회된 총 스테이션 수: {len(stations_df)}")
    print(f"샘플 스테이션 데이터:")
    print(stations_df.head())
    
    classified_stations = classify_stations(stations_df)
    stations_with_scores = calculate_priority_scores(classified_stations)
    
    print("\n스테이션 분석 (상위 10개 우선순위 스테이션):")
    print("-" * 80)
    
    priority_stations = stations_with_scores[stations_with_scores['station_type'] != 'Balanced'].nlargest(10, 'priority_score')
    
    for _, station in priority_stations.iterrows():
        print(f"\n{station['station_name']} (ID: {station['station_id']})")
        print(f"  현재: {station['current_bikes']}/{station['capacity']} 대")
        print(f"  최적: {station['optimal_stock']:.0f} 대")
        print(f"  유형: {station['station_type']}")
        print(f"  우선순위 점수: {station['priority_score']:.3f}")
    
    route = generate_reallocation_route(stations_df, depot_coords, truck_capacity=20, max_stops=8)
    
    print("\n" + "=" * 80)
    print("최적화된 재배치 경로")
    print("=" * 80)
    
    total_time = 0
    for stop in route:
        print(f"\n정차 {stop['stop_number']}: {stop['station_name']}")
        print(f"  작업: {stop['action'].upper()}")
        if stop['action'] not in ['return']:
            print(f"  이동 자전거: {stop['bikes_transferred']}대")
        print(f"  작업 후 트럭 적재량: {stop['truck_load_after']}대")
        print(f"  이동 시간: {stop['travel_time']:.2f}분")
        if stop['action'] not in ['return']:
            print(f"  효율성 점수: {stop['efficiency_score']:.3f}")
            print(f"  우선순위 점수: {stop['priority_score']:.3f}")
        total_time += stop['travel_time']
    
    print("\n" + "=" * 80)
    print(f"경로 요약")
    print("=" * 80)
    print(f"총 정차지: {len(route) - 1}개 스테이션 + 차고 복귀")
    print(f"총 이동 시간: {total_time:.2f}분")
    print(f"총 재배치 자전거: {sum(s['bikes_transferred'] for s in route[:-1])}대")
    
    pickup_stops = [s for s in route if s['action'] == 'pickup']
    dropoff_stops = [s for s in route if s['action'] == 'dropoff']
    print(f"수거 정차지: {len(pickup_stops)}개")
    print(f"배치 정차지: {len(dropoff_stops)}개")

if __name__ == "__main__":
    main()