import json
import requests

def get_nasa_parameters(community: str, temporal: str) -> list:
    BASE = "https://power.larc.nasa.gov/api/system/manager/parameters"
    params = {"community": community, "temporal": temporal}
    
    try:
        r = requests.get(BASE, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        
        if isinstance(data, dict):
            return list(data.keys())
        else:
            return []
    except requests.RequestException as e:
        print(f"Error fetching parameters: {e}")
        return []

# Example usage
if __name__ == "__main__":
    keys = get_nasa_parameters("SB", "daily")
    print(json.dumps(keys, indent=2))
    print(f"len: {len(keys)}")
