import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# Replace with your own API key from OpenWeatherMap
API_KEY = '68aafb1d6e221680d668b4989237a3c5'
BASE_URL = "http://api.openweathermap.org/data/2.5/forecast"

# Load world map data using the correct path to the shapefile
shapefile_path = r'C:\Users\pulapa raja\Desktop\ne_110m_admin_0_countries\ne_110m_admin_0_countries.shp'
world = gpd.read_file(shapefile_path)

# Function to fetch weather data for a given city
def get_weather_data(city_name):
    complete_api_link = f"{BASE_URL}?q={city_name}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(complete_api_link)
        response.raise_for_status()  # Check for HTTP errors
        data = response.json()
        
        # Extract relevant data
        weather_data = []
        for entry in data['list']:
            weather_info = {
                'date_time': entry['dt_txt'],
                'temp': entry['main']['temp'],
                'pressure': entry['main']['pressure'],
                'humidity': entry['main']['humidity'],
                'wind_speed': entry['wind']['speed'],
                'rain': entry['rain'].get('3h', 0) if 'rain' in entry else 0
            }
            weather_data.append(weather_info)
        
        return pd.DataFrame(weather_data)
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()  # Return empty DataFrame in case of error

# Preprocess the data
def preprocess_data(df):
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df = df.drop(columns=['date_time'])
    
    # One-hot encoding for day and hour
    df = pd.get_dummies(df, columns=['day', 'hour'], drop_first=True)
    return df

# Function to train the Linear Regression model and predict
def train_and_predict(city_name, location):
    # Fetch and preprocess data
    df = get_weather_data(city_name)
    if df.empty:
        print("No data available for the given city.")
        return
    
    df = preprocess_data(df)
    
    # Features and target
    X = df.drop(columns=['rain'])
    y = df['rain']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    
    # Plot actual vs predicted values
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Rainfall")
    plt.ylabel("Predicted Rainfall")
    plt.title(f"Actual vs Predicted Rainfall for {city_name}")
    plt.show()
    
    # Geographical Visualization
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy([location['lon']] * len(df), [location['lat']] * len(df)))
    
    # Plotting the map with points
    fig, ax = plt.subplots(figsize=(10, 10))
    world.plot(ax=ax, color='lightblue')
    gdf.plot(ax=ax, marker='o', color='red', markersize=y_pred*10, alpha=0.6)
    
    plt.title(f"Geographical Distribution of Predicted Rainfall in {city_name}")
    plt.show()

# Example usage: Train, predict, and visualize for Hyderabad
location_hyderabad = {'lat': 17.3850, 'lon': 78.4867}
train_and_predict('Hyderabad', location_hyderabad)
