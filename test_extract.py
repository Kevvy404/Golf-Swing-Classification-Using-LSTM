import unittest
from unittest.mock import patch
import pandas as pd
import os
from extract import extract_and_save_features 

class TestExtractAndSaveFeatures(unittest.TestCase):

    def setUp(self):
        # Setup a temporary CSV file to act as our test input
        self.test_input_csv = 'test_input.csv'
        self.test_output_csv = 'test_output.csv'
        self.columns = ['accelerationX', 'rotationRateZ', 'rotationRateY', 'quaternionW', 'label']
        test_data = {
            'accelerationX': [0.1, 0.2, 0.3],
            'rotationRateZ': [0.4, 0.5, 0.6],
            'rotationRateY': [0.7, 0.8, 0.9],
            'quaternionW': [1, 0.9, 0.8],
            'label': ['swing', 'swing', 'miss']
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.test_input_csv, index=False)

    def tearDown(self):
        # Clean up the test CSV files after the tests run
        os.remove(self.test_input_csv)
        if os.path.exists(self.test_output_csv):
            os.remove(self.test_output_csv)

    def test_extract_and_save_features_valid(self):
        # Test that the function works with valid input
        extract_and_save_features(self.test_input_csv, self.test_output_csv)
        # Check that the output file was created
        self.assertTrue(os.path.exists(self.test_output_csv))
        # Check the contents of the output file
        output_data = pd.read_csv(self.test_output_csv)
        self.assertTrue((output_data.columns == self.columns).all())

    def test_extract_and_save_features_missing_columns(self):
        # Alter the input data to have a missing column
        df = pd.read_csv(self.test_input_csv)
        df.drop(columns=['quaternionW'], inplace=True)
        df.to_csv(self.test_input_csv, index=False)
        # Test that the function raises an error with invalid input
        with self.assertRaises(ValueError) as context:
            extract_and_save_features(self.test_input_csv, self.test_output_csv)
        self.assertIn('Missing columns in the CSV file', str(context.exception))

    @patch('builtins.print')
    def test_extract_and_save_features_prints_success(self, mock_print):
        # Test that the function prints a success message with valid input
        extract_and_save_features(self.test_input_csv, self.test_output_csv)
        mock_print.assert_called_with(f"Data successfully saved to {self.test_output_csv}")

if __name__ == '__main__':
    unittest.main()
