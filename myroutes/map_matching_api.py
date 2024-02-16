import os
import requests

class Map_matching:
    def __init__(self):
        pass
    
    
    async def getTurnRouteDataByGPS(self,gps_tracker_data):
        valhalla_base_url = os.getenv("API_URL")
        api_params = {
                "id": "group-4",
                "shape": [{"lat": lat, "lon": lon} for lat, lon in gps_tracker_data],
                "costing": "pedestrian",
                "language": "en",
                "units": "kilometers",
                "shape_match":"map_snap",
            }
        response = requests.post(f"{valhalla_base_url}/trace_route", json=api_params)
        trace_data = response.json()   
        return trace_data

        
