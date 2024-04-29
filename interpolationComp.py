import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt

# Load the dataset from a CSV file
data_path = 'Data/GolfSwingData.csv'
data = pd.read_csv(data_path)

# Ensure the data is sorted by 'seconds_elapsed'
data = data.sort_values(by='seconds_elapsed')

# Define the columns to interpolate
columns_for_interpolation = [
    'rotationRateX', 'rotationRateY', 'rotationRateZ', 
    'gravityX', 'gravityY', 'gravityZ', 
    'accelerationX', 'accelerationY', 'accelerationZ', 
    'quaternionW', 'quaternionX', 'quaternionY', 'quaternionZ'
]

# Choose a degree for polynomial interpolation
poly_degree = 3  # Degree 3 for a cubic polynomial fit

# Interpolation and plotting for each sensor data
for column in columns_for_interpolation:
    # Extract the complete time series and corresponding values
    time_points = data['seconds_elapsed'].values
    sensor_values = data[column].values
    
    # Create a Cubic Spline interpolation
    cs = CubicSpline(time_points, sensor_values)
    print(cs)
    # Create a Linear interpolation
    li = interp1d(time_points, sensor_values, kind='linear')
    print(li)

    # Create a Polynomial interpolation
    poly_coeffs = np.polyfit(time_points, sensor_values, poly_degree)
    poly = np.poly1d(poly_coeffs)
    
    # Generate dense time points for a smooth plot
    dense_time_points = np.linspace(min(time_points), max(time_points), 300)
    interpolated_values_cubic = cs(dense_time_points)
    interpolated_values_linear = li(dense_time_points)
    interpolated_values_poly = poly(dense_time_points)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, sensor_values, 'o', label='Actual Data', markersize=5)
    plt.plot(dense_time_points, interpolated_values_cubic, '-', label='Cubic Spline Interpolation', linewidth=2)
    plt.plot(dense_time_points, interpolated_values_linear, '--', label='Linear Interpolation', color='red', linewidth=1.5)
    plt.plot(dense_time_points, interpolated_values_poly, ':', label='Polynomial Interpolation', color='green', linewidth=2)
    plt.title(f'Interpolation Comparison for {column}')
    plt.xlabel('Time (seconds_elapsed)')
    plt.ylabel('Sensor Value')
    plt.legend()
    plt.grid(True)
    plt.show()

