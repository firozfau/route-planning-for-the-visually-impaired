import os
import requests
import json


class Transit:
    def __init__(self):
        pass
    
    def setter(self,data):
        
        self.from_lat=str(data["from"]["lat"])
        self.from_lon=str(data["from"]["lon"])
        self.from_radius=str(data["from"]["radius"])
        
        
        self.destination_lat=str(data["destination"]["lat"])
        self.destination_lon=str(data["destination"]["lon"])
        self.destination_radius=str(data["destination"]["radius"])
        
        

    async def getTransitData(self):
        api_base_url = os.getenv("API_URL") + "/TBD/transit_available?json="

        api_params = {
                    "locations": [
                        {
                        "lat": self.from_lat, 
                        "lon": self.from_lon, 
                        "radius": self.from_radius
                        },
                        {
                        "lat":self.destination_lat,
                        "lon": self.destination_lon,  
                        "radius": self.destination_radius,  
                        }
                    ]
        }

        api_full_url = api_base_url + json.dumps(api_params)
        response = requests.get(api_full_url)
        if response.status_code == 200:
            return response.json()
        else:
            return False    