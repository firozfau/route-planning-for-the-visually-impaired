
import time
import polyline

class Task_4:
    def __init__(self):
        pass

    async def runWalkingSimulator(self,api_data):
    
        walking_start_time=time.time()

        shape_data=api_data['trip']['legs'][0]['shape']
        maneuvers_data=api_data['trip']['legs'][0]['maneuvers']
        
        
        start_message=" Now start walking :";
        
        print(f"\n\n🧍{start_message}\n") 
        
        for obj in maneuvers_data:
            try:
                
                
                if obj['type']==4: 
                    print(f"\n🧍{obj['verbal_pre_transition_instruction']}")  
                    print("\n\n")
                    break
                
                await self.walkingOnStreet(obj['type'],obj['verbal_post_transition_instruction'],obj['time'])

                print("\n\n")
            except Exception as e:
                print(f"An Exception occurred: {e}")
    
    
    async def walkingOnStreet(self,type,instruction_text,time_duration):
        total_walking_time=0
        sleep_time=2
        instruction_code={
            "1":"WALK : ",
            "10":"GO TO RIGHT : ",
            "15":"GO TO LEFT : ",
            "8":"GOT TO CENTER : ", 
        }
        
        
        print(f"{instruction_code[str(type)]} {instruction_text} [ Approximately {time_duration} seconds ]")  
        while total_walking_time<=time_duration:
            print(f"\n\t🚶Go straight for {round(time_duration-total_walking_time)}s")
            time.sleep(sleep_time)
            total_walking_time+=sleep_time+5 