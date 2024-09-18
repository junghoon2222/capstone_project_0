import requests

class InfoFetcher:
    def fetch(self):
        raise NotImplementedError("Subclasses should implement this method")

class WeatherFetcher(InfoFetcher):
    def __init__(self, api_key, location):
        self.api_key = api_key
        self.location = location

    def fetch(self):
        url = f"http://api.openweathermap.org/data/2.5/weather?q={self.location}&appid={self.api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return f"Weather in {self.location}: {data['weather'][0]['description']}, {data['main']['temp']}°C"
        else:
            return "Failed to fetch weather data"
