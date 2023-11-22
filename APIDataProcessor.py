import requests
from xml.etree import ElementTree as ET
from datetime import datetime, timedelta
import os
import json
import math
from geopy.geocoders import Nominatim
from pprint import pprint

class APIDataProcessor:
    def __init__(self):
        pass

    def getStreetName(self, latitude, longitude):
        geolocator = Nominatim(
            user_agent="my_geocoder"
        )  # You can set your own user agent
        location = geolocator.reverse((latitude, longitude), language="en")

        if location and "address" in location.raw:
            street_name = location.raw["address"].get("road", "Street name not found")
            return street_name
        else:
            return False

    def getDataFromValhallaAPI(self, position_data):

        pre_street_name=position_data["pre_position"]["street"]
        cur_street_name=position_data["cur_position"]["street"]
        
        api_base_url = os.getenv("VALHALLA_API_BASE_URL") + "/route?json="

        if (pre_street_name==False or pre_street_name==""):
            pre_street_name = self.getStreetName(
                position_data["pre_position"]["lat"], position_data["pre_position"]["lon"]
            )
        
        if(cur_street_name==False or cur_street_name==""):    
            cur_street_name = self.getStreetName(
                position_data["cur_position"]["lat"], position_data["cur_position"]["lat"]
            )


        api_params = {
            "id": "group-4",
            "costing": "pedestrian",
            "use_ferry": 1,
            "use_hills": 0.1,
            "walking_speed": 5.1,
            "walkway_factor": 35,
            "sidewalk_factor": 40,
            "alley_factor": 2,
            "driveway_factor": 5,
            "step_penalty": 0,
            "language": "en",
            "units": "kilometers",
            "locations": [
                {
                    "lat": position_data["pre_position"]["lat"],
                    "lon": position_data["pre_position"]["lon"],
                    "street": pre_street_name,
                },
                {
                    "lat": position_data["cur_position"]["lat"],
                    "lon": position_data["cur_position"]["lon"],
                    "street": cur_street_name,
                },
            ],
        }

        api_full_url = api_base_url + json.dumps(api_params)
        # Make a request using the params parameter
        response = requests.get(api_full_url)

        # Check the response
        if response.status_code == 200:
            return response.json()
        else:
            return False
            # print(f"Error: {response.status_code}, {response.text}")




    def getKilometerTOmeter(self,kilometer):
        meter = kilometer * 1000
        return meter


    def getEarthRadius(self,latitude, longitude): 
        latitude_radians = math.radians(latitude)
        longitude_radians = math.radians(longitude)

        # WGS84 ellipsoid parameters
        a = 6378137.0  # Semi-major axis
        f = 1 / 298.257223563  # Flattening factor

        # Calculate the semi-minor axis
        b = a * (1 - f)

        # Calculate the radius of curvature in the prime meridian
        N = a / math.sqrt(1 - f * math.sin(latitude_radians) ** 2)

        # Calculate the radius of curvature in the meridian
        R = N / (1 - f * math.sin(latitude_radians) ** 2)

        return R
    
    
    
    
    def getNewLatLon(self,current_lat,current_lon,kilometer):
        
        earth_radius = self.getEarthRadius(current_lat,current_lon)
        distance_meters = self.getKilometerTOmeter(kilometer)

        latitude_radians = math.radians(current_lat)
        longitude_radians = math.radians(current_lon)
        
        change_in_latitude = distance_meters / earth_radius * math.cos(latitude_radians)
        new_starting_point_lat = current_lat + change_in_latitude
        
        change_in_longitude = distance_meters / earth_radius * math.sin(longitude_radians)
        new_starting_point_lon = current_lon + change_in_longitude
        
        data={
            "new_latitude":new_starting_point_lat,
            "new_longitude":new_starting_point_lon,
        }
    
        return data
    
    
    def getCustomizeAPIdata(self,position):
        super_data=[]
        all_api_data= self.getDataFromValhallaAPI(position)
        
        #pprint(all_api_data)
        #exit()
        
        if(all_api_data):
            if(all_api_data["trip"]):
                if(all_api_data["trip"]["legs"]):
                    if(all_api_data["trip"]["legs"][0]):
                        
                        
                        starting_point_lat = position["pre_position"]["lat"]
                        starting_point_lon = position["pre_position"]["lon"]
                        #pprint(all_api_data)
                        #exit()
                        api_data=all_api_data["trip"]["legs"][0]["maneuvers"]
                        store_api_data=api_data
                        
                
                        for iterationNumber, data in enumerate(api_data):
                        
                            if(data['type']!=1): 
                            
                                new_lat_lon = self.getNewLatLon(store_api_data[iterationNumber-1]['lat'],store_api_data[iterationNumber-1]['lon'],store_api_data[iterationNumber-1]['length'])
                                starting_point_lat=new_lat_lon['new_latitude']
                                starting_point_lon=new_lat_lon['new_longitude']
                        
                            data['lat']=starting_point_lat
                            data['lon']=starting_point_lon 
                                
                            super_data.append(data)
        
        return super_data
    
    def getHaversineDistance(self,from_lat,from_lon,to_lat,to_lon):
        from_lat_radians = math.radians(from_lat)
        from_lon_radians = math.radians(from_lon)
        to_lat_radians = math.radians(to_lat)
        to_lon_radians = math.radians(to_lon)
        
        earth_radius =self.getEarthRadius(from_lat,from_lon)
        
        # Calculate the difference in latitude and longitude
        delta_lat = to_lat_radians - from_lat_radians
        delta_lon = to_lon_radians - from_lon_radians
        
        # Calculate the haversine
        a = math.sin(delta_lat / 2) ** 2 + math.cos(from_lat_radians) * math.cos(to_lat_radians) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = earth_radius * c
        return distance   # meter
    
    def getDataFromGPXFile(self, gpx_file_location):
        tree = ET.parse(gpx_file_location)
        root = tree.getroot()

        gps_data = []
        for trkpt in root.findall(os.getenv("GPX_FILE_PARSER_TRKPT_URL")):
            lat = float(trkpt.attrib["lat"])
            lon = float(trkpt.attrib["lon"])
            #ele = float(trkpt.find(os.getenv("GPX_FILE_PARSER_ELE_URL")).text)
            #time_str = trkpt.find(os.getenv("GPX_FILE_PARSER_TIME_URL")).text
            #time = datetime.strptime(time_str, "%Y-%m-%dT%H:%M:%SZ")

            gps_data.append({"lat": lat, "lon": lon})
        return gps_data

    def getNavigationPosition(type):
        # Define the dictionary with index as key and value as text
        navigation_dict = {
            "0": "kNone",
            "1": "kStart",
            "2": "kStartRight",
            "3": "kStartLeft",
            "4": "kDestination",
            "5": "kDestinationRight",
            "6": "kDestinationLeft",
            "7": "kBecomes",
            "8": "kContinue",
            "9": "kSlightRight",
            "10": "kRight",
            "11": "kSharpRight",
            "12": "kUturnRight",
            "13": "kUturnLeft",
            "14": "kSharpLeft",
            "15": "kLeft",
            "16": "kSlightLeft",
            "17": "kRampStraight",
            "18": "kRampRight",
            "19": "kRampLeft",
            "20": "kExitRight",
            "21": "kExitLeft",
            "22": "kStayStraight",
            "23": "kStayRight",
            "24": "kStayLeft",
            "25": "kMerge",
            "26": "kRoundaboutEnter",
            "27": "kRoundaboutExit",
            "28": "kFerryEnter",
            "29": "kFerryExit",
            "30": "kTransit",
            "31": "kTransitTransfer",
            "32": "kTransitRemainOn",
            "33": "kTransitConnectionStart",
            "34": "kTransitConnectionTransfer",
            "35": "kTransitConnectionDestination",
            "36": "kPostTransitConnectionDestination",
            "37": "kMergeRight",
            "38": "kMergeLeft"
        }
        if(navigation_dict[navigation_dict]):
            return navigation_dict[navigation_dict]
