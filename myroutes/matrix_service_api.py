import os
import aiofiles
import requests
import json

class Matrix:
    def __init__(self):
        pass

    async def getSourcesTargetJsonData(self):
        if not os.path.exists('files'):
            os.makedirs('files')

        async with aiofiles.open('files/sources_to_target.json', 'r') as file:
            content = await file.read()
            json_object = json.loads(content)

        return json_object


    async def getOneToMany(self):
        api_base_url = os.getenv("API_URL") + "/sources_to_targets?json="

        json_data = await self.getSourcesTargetJsonData() 

        api_params = json_data

        api_full_url = api_base_url + json.dumps(api_params)
        response = requests.get(api_full_url)
        if response.status_code == 200:
            return response.json()
        else:
            return False