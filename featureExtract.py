import pandas as pd
import numpy as np
import unittest
import os

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        # Path to the CSV file
        self.file_path = 'extracted_features.csv'

    def test_file_exists(self):
        """Test if the CSV file exists."""
        self.assertTrue(os.path.exists(self.file_path))

    def test_file_not_empty(self):
        """Test that the CSV file is not empty."""
        self.assertTrue(os.stat(self.file_path).st_size > 0)

    def test_features_updated(self):
        """Test that the features in the CSV file have been updated correctly."""
        # Load the data from the CSV file
        data = pd.read_csv(self.file_path)
        
        # Check if specific columns are present
        expected_columns = {'mean_accelerationX', 'max_accelerationX', 'min_accelerationX',
                            'mean_rotationRateX', 'max_rotationRateX', 'min_rotationRateX'}
        self.assertTrue(expected_columns.issubset(data.columns))

        # Example to check if the data is not empty (no zero rows)
        self.assertTrue(not data.empty)

        # Optionally, check for more specific conditions, like statistical properties
        mean_values = data.filter(like='mean_').mean()
        self.assertFalse(mean_values.isnull().any(), "Mean values should not contain NaN")

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def normalize_data(df):
    """Normalize the data to have zero mean and unit variance for each column."""
    return (df - df.mean(numeric_only=True)) / df.std(numeric_only=True)

def extract_features(data):
    """Extract features from acceleration, rotation rate, and quaternion data for each row."""
    # Initialize a list to hold feature dictionaries for each row (observation)
    features_list = []
    
    # Iterate over each row in the DataFrame
    for index, row in data.iterrows():
        # Initialize a dictionary to store features for the current row
        features = {'observation_index': index}
        
        # Extract features for acceleration and rotation rate sensors
        for sensor in ['acceleration', 'rotationRate']:
            for axis in ['X', 'Y', 'Z']:
                # Construct the full column name for the current sensor and axis
                column_name = f'{sensor}{axis}'
                # Calculate and store the statistical features for the current row
                features[f'mean_{column_name}'] = np.mean(row[column_name])
                features[f'max_{column_name}'] = np.max(row[column_name])
                features[f'min_{column_name}'] = np.min(row[column_name])

        # Extract features for quaternion components
        for component in ['W', 'X', 'Y', 'Z']:
            column_name = f'quaternion{component}'
            features[f'mean_{column_name}'] = np.mean(row[column_name])
            features[f'max_{column_name}'] = np.max(row[column_name])
            features[f'min_{column_name}'] = np.min(row[column_name])
        
        # Append the features of the current row to the list
        features_list.append(features)
    
    # Convert the list of feature dictionaries to a DataFrame and return it
    return pd.DataFrame(features_list)

def main():
    # Load the data
    data = load_data('Data/GolfSwingData.csv')
    
    # Reset the index to ensure it's unique and sequential
    data.reset_index(drop=True, inplace=True)

    # Check if data is loaded correctly
    # print("Data Loaded: ", data.head())

    # Normalize the data
    normalized_data = normalize_data(data)
    
    # Extract features using the DataFrame index as identifiers
    features = extract_features(normalized_data)
    
    # Check extracted features
    # print("Extracted Features: ", features)

    # Save or return features for further use
    features.to_csv('Data/extracted_features.csv', index=False)

if __name__ == '__main__':
    main()
    unittest.main()


