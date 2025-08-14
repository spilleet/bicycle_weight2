import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import math
import requests
import json
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# --- 기존 헬퍼 함수들 (일부 개선) ---

def get_travel_cost(start_coords: Tuple[float, float], end_coords: Tuple[float, float]) -> float:
    """
    두 위도/경도 지점 사이의 Haversine 거리를 계산합니다. (단위: km)
    실제 경로와 유사한 '비용'을 위해 상수를 곱합니다. (예: 이동 시간 분으로 변환)
    """
    R = 6371  # 지구 반지름 (km)
    lat1, lon1 = math.radians(start_coords[0]), math.radians(start_coords[1])
    lat2, lon2 = math.radians(end_coords[0]), math.radians(end_coords[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance_km = R * c
    # 이동 시간(분)으로 대략적인 변환 (평균 속도 20km/h 가정)
    travel_time_minutes = distance_km / 20 * 60 
    return travel_time_minutes

def classify_stations(stations_df: pd.DataFrame) -> pd.DataFrame:
    df = stations_df.copy()
    df['optimal_stock'] = df['capacity'] * 0.5
    df['deviation'] = df['current_bikes'] - df['optimal_stock']
    conditions = [
        df['deviation'] > 0,
        df['deviation'] < 0,
    ]
    choices = ['Pickup', 'Drop-off']
    df['station_type'] = np.select(conditions, choices, default='Balanced')
    return df

def calculate_priority_scores(stations_df: pd.DataFrame, 
                             pickup_weights: Tuple[float, float] = (0.7, 0.3),
                             dropoff_weights: Tuple[float, float] = (0.8, 0.2)) -> pd.DataFrame:
    df = stations_df.copy()
    w1, w2 = pickup_weights
    w3, w4 = dropoff_weights
    
    pickup_mask = df['station_type'] == 'Pickup'
    df.loc[pickup_mask, 'saturation_score'] = df.loc[pickup_mask, 'current_bikes'] / df.loc[pickup_mask, 'capacity']
    df.loc[pickup_mask, 'surplus_bikes'] = df.loc[pickup_mask, 'deviation']
    
    dropoff_mask = df['station_type'] == 'Drop-off'
    df.loc[dropoff_mask, 'depletion_score'] = 1 - (df.loc[dropoff_mask, 'current_bikes'] / df.loc[dropoff_mask, 'capacity'])
    df.loc[dropoff_mask, 'demand_bikes'] = -df.loc[dropoff_mask, 'deviation']
    
    # 수거 정류소 우선순위 계산
    if pickup_mask.any():
        surplus_95th = df.loc[pickup_mask, 'surplus_bikes'].quantile(0.95)
        df.loc[pickup_mask, 'surplus_bikes_capped'] = df.loc[pickup_mask, 'surplus_bikes'].clip(upper=surplus_95th)
        min_surplus, max_surplus = df.loc[pickup_mask, 'surplus_bikes_capped'].min(), df.loc[pickup_mask, 'surplus_bikes_capped'].max()
        if max_surplus > min_surplus:
            df.loc[pickup_mask, 'normalized_surplus_score'] = (df.loc[pickup_mask, 'surplus_bikes_capped'] - min_surplus) / (max_surplus - min_surplus)
        else:
            df.loc[pickup_mask, 'normalized_surplus_score'] = 0
        df.loc[pickup_mask, 'priority_score'] = w1 * df.loc[pickup_mask, 'saturation_score'] + w2 * df.loc[pickup_mask, 'normalized_surplus_score']
    
    # 배치 정류소 우선순위 계산
    if dropoff_mask.any():
        demand_95th = df.loc[dropoff_mask, 'demand_bikes'].quantile(0.95)
        df.loc[dropoff_mask, 'demand_bikes_capped'] = df.loc[dropoff_mask, 'demand_bikes'].clip(upper=demand_95th)
        min_demand, max_demand = df.loc[dropoff_mask, 'demand_bikes_capped'].min(), df.loc[dropoff_mask, 'demand_bikes_capped'].max()
        if max_demand > min_demand:
            df.loc[dropoff_mask, 'normalized_demand_score'] = (df.loc[dropoff_mask, 'demand_bikes_capped'] - min_demand) / (max_demand - min_demand)
        else:
            df.loc[dropoff_mask, 'normalized_demand_score'] = 0
        df.loc[dropoff_mask, 'priority_score'] = w3 * df.loc[dropoff_mask, 'depletion_score'] + w4 * df.loc[dropoff_mask, 'normalized_demand_score']
    
    df['priority_score'] = df['priority_score'].fillna(0)
    return df

# --- Google OR-Tools를 사용한 새로운 경로 생성 함수 ---

def create_cvrp_data_model(problem_stations: pd.DataFrame, depot_coords: Tuple[float, float], truck_capacity: int) -> Dict:
    """CVRP 솔버를 위한 데이터 모델을 생성합니다."""
    # 차고지(depot)를 포함한 전체 위치 리스트 생성
    locations = [depot_coords] + list(zip(problem_stations['latitude'], problem_stations['longitude']))
    
    # 거리 행렬 계산
    num_locations = len(locations)
    distance_matrix = np.zeros((num_locations, num_locations), dtype=int)
    for i in range(num_locations):
        for j in range(num_locations):
            # 이동 비용(시간)을 정수형으로 변환하여 사용
            distance_matrix[i, j] = int(get_travel_cost(locations[i], locations[j]))

    # 요구량(Demands) 계산
    # 수거(Pickup): 양수, 배치(Drop-off): 음수
    demands = [0] # 차고지의 요구량은 0
    for _, station in problem_stations.iterrows():
        if station['station_type'] == 'Pickup':
            # 트럭 용량을 초과하지 않는 범위에서 수거할 자전거 수
            demand = min(int(station['surplus_bikes']), truck_capacity)
            demands.append(demand)
        else: # Drop-off
            # 배치할 자전거 수 (음수로 표현)
            demand = -min(int(station['demand_bikes']), truck_capacity)
            demands.append(demand)
            
    data = {
        'distance_matrix': distance_matrix.tolist(),
        'demands': demands,
        'vehicle_capacities': [truck_capacity],
        'num_vehicles': 1,
        'depot': 0
    }
    return data

def generate_optimized_route_with_ortools(stations_df: pd.DataFrame, 
                                          depot_coords: Tuple[float, float],
                                          truck_capacity: int = 20,
                                          max_stations_in_problem: int = 15) -> List[Dict]:
    """OR-Tools CVRP 솔버를 사용하여 최적의 재배치 경로를 생성합니다."""
    # 1. 문제에 포함할 정류소 선택
    df = classify_stations(stations_df)
    df = calculate_priority_scores(df)
    
    priority_stations = df[df['station_type'] != 'Balanced'].nlargest(max_stations_in_problem, 'priority_score')
    
    if priority_stations.empty:
        print("재배치가 필요한 정류소가 없습니다.")
        return []

    # 2. CVRP 데이터 모델 생성
    data = create_cvrp_data_model(priority_stations, depot_coords, truck_capacity)

    # 3. 라우팅 모델 설정
    manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']), data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    # 거리 콜백 등록
    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 요구량(용량) 콜백 등록 및 제약조건 추가
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],
        True,  # start cumul to zero
        'Capacity'
    )
    
    # 4. 검색 파라미터 설정 및 해결
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.FromSeconds(5)

    solution = routing.SolveWithParameters(search_parameters)
    
    # 5. 결과 해석 및 반환
    if not solution:
        print('최적 경로를 찾을 수 없습니다!')
        return []
        
    route = []
    index = routing.Start(0)
    truck_load = 0
    stop_num = 1
    
    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        
        if node_index != data['depot']:
            station_info = priority_stations.iloc[node_index - 1] # depot(0) 제외
            demand = data['demands'][node_index]
            
            truck_load += demand
            
            route.append({
                'stop_number': stop_num,
                'station_id': station_info['station_id'],
                'station_name': station_info['station_name'],
                'action': 'Pickup' if demand > 0 else 'Drop-off',
                'bikes_transferred': abs(demand),
                'truck_load_after': truck_load,
                'travel_time': routing.GetArcCostForVehicle(index, solution.Value(routing.NextVar(index)), 0),
                'priority_score': station_info['priority_score']
            })
            stop_num += 1
            
        index = solution.Value(routing.NextVar(index))

    # 차고지로 복귀 정보 추가
    # 마지막 정류소에서 차고지로 돌아오는 시간
    last_node_index = manager.IndexToNode(solution.Value(routing.NextVar(routing.End(0))))
    return_time = data['distance_matrix'][last_node_index][data['depot']]
    
    route.append({
        'stop_number': stop_num,
        'station_id': 'DEPOT',
        'station_name': 'Return to Depot',
        'action': 'return',
        'bikes_transferred': 0,
        'truck_load_after': truck_load,
        'travel_time': return_time,
        'priority_score': 0
    })

    return route


def fetch_realtime_bike_data(api_key: str, start_idx: int = 1, end_idx: int = 1000) -> pd.DataFrame:
    # 데이터 요청할 웹 서버의 주소를 지정
    url = f"http://openapi.seoul.go.kr:8088/{api_key}/json/bikeList/{start_idx}/{end_idx}/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'rentBikeStatus' not in data or 'row' not in data['rentBikeStatus']:
            print("API 응답에 'rentBikeStatus' 또는 'row' 필드가 없습니다. 응답 확인:", data)
            return pd.DataFrame()
            
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
    except requests.exceptions.RequestException as e:
        print(f"API 데이터 요청 중 오류 발생: {e}")
        return pd.DataFrame()
    except KeyError:
        print("데이터 구조가 예상과 다릅니다. API 응답:", data)
        return pd.DataFrame()

def main():
    # 중요: 발급받은 실제 따릉이 API 키를 입력하세요.
    SEOUL_API_KEY = "6464716442737069363863566b466c"  # 사용자의 키로 교체됨
    
    print("=" * 80)
    print("BIKE REALLOCATION SYSTEM - OR-TOOLS OPTIMIZATION")
    print("=" * 80)
    print("\n실시간 따릉이 정류소 데이터 요청 중...")
    
    stations_df = fetch_realtime_bike_data(SEOUL_API_KEY, 1, 1000)
    if stations_df.empty:
        print("데이터를 가져오지 못했습니다. 프로그램을 종료합니다.")
        return

    stations_df = stations_df[stations_df['capacity'] > 0].dropna()
    depot_coords = (37.5385095, 127.1251279) # 차고지 좌표 (예: 천호 센터)

    print(f"\n분석 대상 정류소 수: {len(stations_df)}")
    print(f"샘플 데이터:")
    print(stations_df.head())
    
    # 최적화된 경로 생성
    route = generate_optimized_route_with_ortools(stations_df, depot_coords, truck_capacity=20, max_stations_in_problem=15)

    if not route:
        return
        
    print("\n" + "=" * 80)
    print("OPTIMIZED REALLOCATION ROUTE (via Google OR-Tools)")
    print("=" * 80)
    
    total_time = 0
    for stop in route:
        print(f"\nStop {stop['stop_number']}: {stop['station_name']} (ID: {stop['station_id']})")
        print(f"  Action: {stop['action'].upper()}")
        if stop['action'] not in ['return']:
            print(f"  Bikes Transferred: {stop['bikes_transferred']}")
        print(f"  Truck Load After: {stop['truck_load_after']} bikes")
        print(f"  Travel Time to here: {stop['travel_time']:.2f} minutes")
        if stop['action'] not in ['return']:
            print(f"  Original Priority Score: {stop['priority_score']:.3f}")
        total_time += stop['travel_time']

    print("\n" + "=" * 80)
    print(f"ROUTE SUMMARY")
    print("=" * 80)
    print(f"Total Stops: {len(route) - 1} stations + depot return")
    print(f"Total Estimated Travel Time: {total_time:.2f} minutes")
    
    bikes_picked_up = sum(s['bikes_transferred'] for s in route if s['action'] == 'Pickup')
    print(f"Total Bikes Picked Up: {bikes_picked_up}")
    
    pickup_stops = [s for s in route if s['action'] == 'Pickup']
    dropoff_stops = [s for s in route if s['action'] == 'Drop-off']
    print(f"Pickup Stops: {len(pickup_stops)}")
    print(f"Drop-off Stops: {len(dropoff_stops)}")

if __name__ == "__main__":
    main()