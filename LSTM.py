import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TestPredictModel(unittest.TestCase):

    def setUp(self):
        # Mock data setup for all tests
        self.features = ['accelerationX', 'rotationRateZ', 'rotationRateY', 'quaternionW']
        self.data = pd.DataFrame({
            'accelerationX': np.random.randn(100),
            'rotationRateZ': np.random.randn(100),
            'rotationRateY': np.random.randn(100),
            'quaternionW': np.random.randn(100),
            'label': ['swing']*50 + ['miss']*50
        })
        self.sequence_length = 10

    def test_preprocess_data(self):
        # Ensure data is scaled correctly
        scaled_data = preprocess_data(self.data, self.features)
        self.assertEqual(scaled_data.shape, (100, 4))  # Check the shape is maintained
        self.assertNotEqual(np.sum(self.data['accelerationX']), np.sum(scaled_data[:, 0]))  # Ensure values are scaled

    def test_create_sequences(self):
        # Test sequence generation
        labels = np.array([1]*100)
        processed_data = np.random.randn(100, 4)
        sequences, seq_labels = create_sequences(processed_data, labels, self.sequence_length)
        self.assertEqual(len(sequences), 91)  # 100 - 10 + 1
        self.assertEqual(len(seq_labels), 91)
        self.assertEqual(sequences.shape[1:], (self.sequence_length, 4))  # Shape of each sequence

    @patch('tensorflow.keras.models.load_model')
    @patch('pandas.read_csv')
    def test_load_and_predict(self, mock_read_csv, mock_load_model):
        # Mock the dependencies
        mock_read_csv.return_value = self.data
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.25, 0.25, 0.25, 0.25]] * 91)
        mock_load_model.return_value = mock_model

        # Call the function
        result = load_and_predict('Model/trainedLSTM.keras', 'Data/GolfSwingData-interpolate.csv', self.features, self.sequence_length, 'label')
        self.assertIn('Accuracy', result)
        self.assertIn('Prediction', result)
        self.assertIn('Precision', result)
        self.assertIn('Recall', result)
        self.assertIn('F1 Score', result)

    def test_map_predictions(self):
        # Test prediction mapping
        predictions = np.array([0, 1, 2, 3])
        mapped = map_predictions(predictions)
        self.assertTrue(np.array_equal(mapped, np.array(['Drive', 'Chip', 'Putt', 'No Shot'])))

def preprocess_data(data, features):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled

def create_sequences(data, labels, sequence_length):
    xs = []
    ys = []
    for i in range(len(data) - sequence_length + 1):
        xs.append(data[i:(i + sequence_length)])
        ys.append(labels[i + sequence_length - 1])  # Ensure labels align with the end of the sequence
    return np.array(xs), np.array(ys)

def load_and_predict(model_path, data_path, features, sequence_length, label_col):
    try:
        expected_feature_count = 4  # Adjust this to match the actual number of features your model was trained with
        if len(features) != expected_feature_count:
            error_message = (
                f"Model expects {expected_feature_count} features, but {len(features)} were provided. "
                f"Provided features: {features}"
            )
            raise ValueError(error_message)

        model = load_model(model_path)
        new_data = pd.read_csv(data_path)
        
        labels = new_data[label_col].values  # Extract labels
        new_data = new_data.drop(columns=[label_col])  # Drop the label column from data to be processed

        if isinstance(labels[0], str):  # Check if the labels are strings
            label_encoder = LabelEncoder()
            labels = label_encoder.fit_transform(labels)
            # print("Labels converted to integers.")

        # Preprocess only the feature data
        feature_data_processed = preprocess_data(new_data, features)

        # Generate sequences and align labels
        data_sequences, aligned_labels = create_sequences(feature_data_processed, labels, sequence_length)

        # print(f"Generated {len(data_sequences)} sequences and {len(aligned_labels)} labels.")

        predictions = model.predict(data_sequences)
        predicted_labels = np.argmax(predictions, axis=1)

        if len(predicted_labels) > 0:
            average_prediction = np.round(np.mean(predicted_labels)).astype(int)
            average_shot = map_predictions(average_prediction)
        else:
            average_shot = "No predictions available"
        
        accuracy = accuracy_score(aligned_labels, predicted_labels) * 100
        precision = precision_score(aligned_labels, predicted_labels, average='macro', zero_division=1)
        recall = recall_score(aligned_labels, predicted_labels, average='macro', zero_division=1)
        f1 = f1_score(aligned_labels, predicted_labels, average='macro', zero_division=1)

        return {"Prediction": average_shot,"Accuracy": f"{accuracy: .2f}%" ,"Precision": precision, "Recall": recall, "F1 Score": f1}
    except Exception as e:
        # print(f"An error occurred: {e}")
        return {"Error": str(e)}

def map_predictions(predictions):
    # Define your mapping
    label_mapping = {
        0: 'Drive',
        1: 'Chip',
        2: 'Putt',
        3: 'No Shot'
    }
    # Apply mapping to the predictions
    mapped_predictions = np.vectorize(label_mapping.get)(predictions)
    return mapped_predictions

if __name__ == "__main__":
    model_path = 'Model/trainedLSTM.keras'
    # Path to your new data
    data_path = 'Data/GolfSwingData-interpolate.csv' 
    features = ['accelerationX', 'rotationRateZ', 'rotationRateY', 'quaternionW']
    # Replace 'true_labels' with the actual column name for true labels
    label_col = 'label' 
    # Must match the sequence length used during training 
    sequence_length = 50

    results = load_and_predict(model_path, data_path, features, sequence_length, label_col)
    print("Prediction: ", results["Prediction"])
    print("Accuracy:", results["Accuracy"])
    print("Precision:", results["Precision"])
    print("Recall:", results["Recall"])
    print("F1 Score:", results["F1 Score"])
    unittest.main()