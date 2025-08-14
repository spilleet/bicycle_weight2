import requests
import json

def test_api():
    url = "http://openapi.seoul.go.kr:8088/sample/json/bikeList/1/5/"
    print(f"Testing API: {url}")
    
    try:
        response = requests.get(url, timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Headers: {response.headers}")
        print(f"Content Length: {len(response.text)}")
        print(f"First 200 chars: {response.text[:200]}")
        
        if response.text:
            data = response.json()
            print("\nJSON parsed successfully!")
            print(f"Keys in response: {data.keys()}")
            
            if 'rentBikeStatus' in data:
                print(f"Keys in rentBikeStatus: {data['rentBikeStatus'].keys()}")
                if 'row' in data['rentBikeStatus']:
                    print(f"Number of stations: {len(data['rentBikeStatus']['row'])}")
                    print("\nFirst station data:")
                    print(json.dumps(data['rentBikeStatus']['row'][0], indent=2, ensure_ascii=False))
        else:
            print("Empty response!")
            
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_api()