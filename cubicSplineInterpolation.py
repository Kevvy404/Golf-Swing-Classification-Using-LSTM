import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

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

# Get time points from user input
print("Enter the first time point:")
time_point1 = float(input())
print("Enter the second time point:")
time_point2 = float(input())
print("Enter the first indice point:")
indice_point1 = float(input())
print("Enter the second indice point:")
indice_point2 = float(input())

# Calculate the interpolation time points
correct_time_points = [time_point1, time_point2]
interpolation_time_points = np.arange(min(correct_time_points), max(correct_time_points) + 0.01, 0.01)

# Extract values at specified indices
specified_indices = [indice_point1, indice_point2]  # Adjust these indices based on actual data
data_at_indices = data.loc[specified_indices]

# Perform cubic spline interpolation for each column at these new time points
detailed_interpolated_values = {column: [] for column in columns_for_interpolation}
for column in columns_for_interpolation:
    values = data_at_indices[column]
    cs = CubicSpline(correct_time_points, values)
    detailed_interpolated_values[column] = cs(interpolation_time_points)

# Create a DataFrame for easier viewing
interpolated_df = pd.DataFrame(detailed_interpolated_values, index=interpolation_time_points)
interpolated_df.index.name = 'seconds_elapsed'

interpolated_df.to_csv('Data/interpolated_data.csv')
