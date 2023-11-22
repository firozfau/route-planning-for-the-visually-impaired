from pprint import pprint
import math  
import time

class BVIPSimulator:
    def __init__(self):

        self.position = {
            "pre_position": {"lat": 0.0, "lon": 0.0, "direction": "", "moved": "","street":""},
            "cur_position": {"lat": 0.0, "lon": 0.0, "direction": "", "moved": "","street":""},
        }


    def getWalkingInformationDetails(self,api_object,position,gpx_file_path):
        
        all_api_data= api_object.getCustomizeAPIdata(position)  
        gpx_data = api_object.getDataFromGPXFile(gpx_file_path)
        #pprint(all_api_data)
        
        #pprint(all_api_data)
        #exit()
    
        first_message="\n Start walking: 🚶‍♀\n\n"
        print(first_message) 
    
    
        for coordinates in gpx_data: 
            result=self.checkAPIdata(api_object,all_api_data,coordinates) 
            print(result)
            time.sleep(2)  # Sleep 

    def checkAPIdata(self,api_object,all_api_data,gpx_data):
        
        message="↓"
        
        from_lat=gpx_data['lat'] 
        from_lon=gpx_data['lon'] 
        
        for data in all_api_data:
            to_lat=data['lat'] 
            to_lon=data['lon']

            distance_data=api_object.getHaversineDistance(from_lat,from_lon,to_lat,to_lon)
            
            if((distance_data==0.0 or distance_data==0 )or (from_lat==to_lat and from_lon==to_lon) ):
                
                
                if(data['type']==4):
                    message="🚶‍♀"+data['verbal_pre_transition_instruction']+"\n"
                else:
                    message=data['verbal_pre_transition_instruction']+" and "+data['verbal_post_transition_instruction']+"\n" 
                break
        
        
        return message
