import time
import json
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from sklearn.datasets import fetch_california_housing

CACHE_FILE = "california_city_cache.json"
RADIUS_KM = 2.0
LIMIT = 2000


data = fetch_california_housing()
lats = data.data[:, -2]
lons = data.data[:, -1]
coordinates = list(zip(lats, lons))

try:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        city_cache = json.load(f)
    city_cache = {tuple(map(float, k.split(","))): v for k, v in city_cache.items()}
except FileNotFoundError:
    city_cache = {}


geolocator = Nominatim(user_agent="california_housing_eda")

def find_in_cache(lat, lon, radius_km=RADIUS_KM):
    """Return cached entry if a point within radius exists."""
    for (clat, clon), entry in city_cache.items():
        if geodesic((lat, lon), (clat, clon)).km <= radius_km:
            return entry
    return None

def get_location(lat, lon):
    entry = find_in_cache(lat, lon)
    if entry:
        return entry

    try:
        location = geolocator.reverse((lat, lon), language="en", addressdetails=True)
        time.sleep(1.1)  # Respect Nominatim usage policy
        if location:
            entry = location.raw.get("address", {})
            city_cache[(lat, lon)] = entry
            return entry
    except Exception as e:
        print(f"Error for {lat},{lon}: {e}")
    return None

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump({f"{lat},{lon}": entry for (lat, lon), entry in city_cache.items()},
                  f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    try:
        for i, (lat, lon) in enumerate(coordinates[LIMIT:], start=2001):
            addr = get_location(lat, lon)
            city = None
            if addr:
                city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("hamlet")
            print(f"[{i}] {lat},{lon} -> {city}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        save_cache()
        print(f"Saved {len(city_cache)} unique lookups to {CACHE_FILE}")
