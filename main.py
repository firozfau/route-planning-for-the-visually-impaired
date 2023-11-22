from pprint import pprint
from dotenv import load_dotenv

load_dotenv()
from APIDataProcessor import APIDataProcessor
from BVIPSimulator import BVIPSimulator

api_object = APIDataProcessor()
BVIP_object = BVIPSimulator()
gpx_file_path = "GPS_tracker.gpx"
position = {
    "pre_position": {
        "lat": 49.59595,
        "lon": 11.002837,
        "street":"Calvinstraße 1, 91054 Erlangen",
        "direction": "",
        "moved": "",
    },
    "cur_position": {
        "lat": 49.596594,
        "lon": 11.007049,
        "street":"Untere Karlstraße, 91054 Erlangen",
        "direction": "",
        "moved": "",
    },
}

# Create an instance of APIDataProcessor
try:

    BVIP_object.getWalkingInformationDetails(api_object,position,gpx_file_path)

    
    
    exit()


except Exception as e:
    print(f"Error: {e}")
