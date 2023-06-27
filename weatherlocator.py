import meteostat as mtst
from geopy.geocoders import Nominatim
import pandas as pd
import plotly.express as px
from datetime import datetime
import os


class WeatherLocator:
    """
    This class gets an input string (location) from the user
    And outputs the weather data of the nearest station from
    the specified station
    """
    def __init__(self,
                 start=datetime(2022, 9, 1),
                 end=datetime(2022, 11, 29, 23, 59)):
        self.location: str
        self.latitude: float
        self.longitude: float
        self.geolocator = Nominatim(user_agent=os.environ["user_weather"])
        self.location_point = None
        self.latlong_dict = {}
        self.latlong_df = pd.DataFrame()
        self.station_id = None
        self.station_name = None
        self.start = start
        self.end = end
        self.weather_graph = None
        self.weather_data = None

    def plot_points(self, location):
        """
        This method returns a plot of the geolocation of
        the specified input location in html format
        :return:
        """
        self.location = location
        self.location_point = self.geolocator.geocode(self.location)
        self.latitude = self.location_point.latitude
        self.longitude = self.location_point.longitude
        self.latlong_dict = {"lat": [self.latitude],
                             "lon": [self.longitude],
                             "loc": self.location,
                             }
        self.latlong_df = pd.DataFrame(self.latlong_dict)
        geofig = px.scatter_geo(data_frame=self.latlong_df,
                                lat="lat",
                                lon="lon",
                                scope="asia",
                                hover_name="loc")
        return geofig.to_html(full_html=True,
                              include_plotlyjs=True)

    def get_weather(self):
        """
        This method returns the weather data (resampled into 5 minute periods)
        of the nearest station in the specified location
        :return: unprocessed and uncleaned weather data
        """
        self.plot_points(location=self.location)
        stations_nearby = mtst.Stations()
        stations_nearby = stations_nearby.nearby(lat=self.latitude,
                                                 lon=self.longitude)
        stations = stations_nearby.fetch()
        station_id = None
        station_name = None
        weather_data = None
        for index, row in stations.iterrows():
            station_id = index
            station_name = row["name"]
            weather_data = mtst.Hourly(str(station_id),
                                       start=self.start,
                                       end=self.end)
            weather_data = weather_data.fetch()
            if weather_data.empty:
                continue
            else:
                break
        self.station_id = station_id
        self.station_name = station_name
        self.weather_graph = px.line(weather_data,
                                x=weather_data.index,
                                y=weather_data.columns[:],
                                title=f"Weather in {self.station_name} Weather Station",)
        self.weather_data = weather_data.resample("300S").pad()
        return self.weather_graph.to_html(), self.weather_data
