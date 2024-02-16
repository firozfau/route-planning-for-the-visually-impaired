import dependency

from pprint import pprint

import asyncio
from dotenv import load_dotenv

load_dotenv()
from lib.customize_lib import Customize_lib
from myroutes.map_matching_api import Map_matching

from myroutes.matrix_service_api import Matrix
from myroutes.transit_available_api import Transit

from tasks.task_4 import Task_4

customize_instance = Customize_lib()
map_matching_instance = Map_matching()
task4_instant = Task_4()

#matrix_instance = Matrix()
#transit_instance = Transit()


from_destination_data = {
    "from": {
        "lat": 49.59618809570076,
        "lon": 11.002669602387051,
        "street":"H2W3+F3 Erlangen", 
        "radius": 20
    },
    "destination": {
        "lat": 49.596428891431195,
        "lon": 11.006161716329569,
        "street":"Untere Karlstraße, 91054 Erlangen", 
        "radius": 10
    },
}

async def main():
    try:
        
        data= await task4_instant.runWalkingSimulator(await map_matching_instance.getTurnRouteDataByGPS(await customize_instance.getGPSCoordinateData()))
    
        
    

    except Exception as e:
        print(f"Error: {e}")










if __name__ == "__main__":
    asyncio.run(main())