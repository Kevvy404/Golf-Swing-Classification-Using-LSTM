# test_feature_extraction.py
import unittest
import pandas as pd
import os
from featureExtract import load_data, normalize_data, extract_features

class TestFeatureExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_csv = 'test.csv'
        test_data = {
            'accelerationX': [0.1, 0.2, 0.3],
            'accelerationY': [0.1, 0.2, 0.3],
            'accelerationZ': [0.1, 0.2, 0.3],
            'rotationRateX': [0.1, 0.2, 0.3],
            'rotationRateY': [0.1, 0.2, 0.3],
            'rotationRateZ': [0.1, 0.2, 0.3],
            'quaternionW': [0.1, 0.2, 0.3],
            'quaternionX': [0.1, 0.2, 0.3],
            'quaternionY': [0.1, 0.2, 0.3],
            'quaternionZ': [0.1, 0.2, 0.3],
            'label': [1, 0, 1]
        }
        pd.DataFrame(test_data).to_csv(cls.test_csv, index=False)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_csv):
            os.remove(cls.test_csv)

    def test_load_data(self):
        """Test that the CSV file loads correctly."""
        data = load_data(self.test_csv)
        self.assertFalse(data.empty)
        self.assertTrue('accelerationX' in data.columns)

    def test_normalize_data(self):
        """Test that the data normalizes correctly."""
        data = pd.read_csv(self.test_csv)
        normalized_data = normalize_data(data)
        expected_mean = pd.Series(0, index=normalized_data.columns).astype(float)  # Cast to float
        expected_std = pd.Series(1, index=normalized_data.columns).astype(float)   # Cast to float
        pd.testing.assert_series_equal(normalized_data.mean(numeric_only=True), expected_mean, rtol=1e-5)
        pd.testing.assert_series_equal(normalized_data.std(numeric_only=True), expected_std, rtol=1e-5)


    def test_extract_features(self):
        """Test that features are extracted correctly."""
        data = pd.read_csv(self.test_csv)
        features = extract_features(data)
        # Make sure the output is a DataFrame and has the right number of columns
        self.assertIsInstance(features, pd.DataFrame)
        expected_columns = ['observation_index', 'mean_accelerationX', 'max_accelerationX', 
                            'min_accelerationX', 'mean_accelerationY', 'max_accelerationY', 
                            'min_accelerationY', 'mean_accelerationZ', 'max_accelerationZ', 
                            'min_accelerationZ', 'mean_rotationRateX', 'max_rotationRateX', 
                            'min_rotationRateX', 'mean_rotationRateY', 'max_rotationRateY', 
                            'min_rotationRateY', 'mean_rotationRateZ', 'max_rotationRateZ', 
                            'min_rotationRateZ', 'mean_quaternionW', 'max_quaternionW', 
                            'min_quaternionW', 'mean_quaternionX', 'max_quaternionX', 
                            'min_quaternionX', 'mean_quaternionY', 'max_quaternionY', 
                            'min_quaternionY', 'mean_quaternionZ', 'max_quaternionZ', 
                            'min_quaternionZ']
        self.assertListEqual(list(features.columns), expected_columns)
        # And it shouldn't be empty
        self.assertFalse(features.empty)

if __name__ == '__main__':
    unittest.main()
