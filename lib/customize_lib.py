import gpxpy

from pprint import pprint

class Customize_lib:
    def __init__(self):
        pass

    
    
    async def getGPSCoordinateData(self):
        with open('files/gps_coordinates.gpx') as f:
            gpx_data = gpxpy.parse(f)

        gpx_coordinates = []
        for track in gpx_data.tracks:
            for segment in track.segments:
                for point in segment.points:
                    gpx_coordinates.append((point.latitude, point.longitude))
        return gpx_coordinates
    
