import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from LSTMTrain import create_sequences, build_model

class TestModelTraining(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create some mock data
        cls.mock_data = pd.DataFrame({
            'accelerationX': np.random.randn(100),
            'rotationRateZ': np.random.randn(100),
            'rotationRateY': np.random.randn(100),
            'quaternionW': np.random.randn(100),
            'label': np.random.choice(['swing', 'hit', 'miss', 'idle'], size=100)
        })
        cls.features = ['accelerationX', 'rotationRateZ', 'rotationRateY', 'quaternionW']
        cls.label = 'label'

    def test_label_encoding(self):
        # Test label encoding
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.mock_data[self.label])
        self.assertEqual(len(np.unique(encoded_labels)), len(np.unique(self.mock_data[self.label])))

    def test_data_normalization(self):
        # Test data normalization
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.mock_data[self.features])
        self.assertAlmostEqual(X_scaled.mean(), 0, places=1)
        self.assertAlmostEqual(X_scaled.std(), 1, places=1)

    def test_create_sequences(self):
        # Test sequence creation
        sequence_length = 10
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.mock_data[self.features])
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(self.mock_data[self.label])
        X, y = create_sequences(X_scaled, encoded_labels, sequence_length)
        self.assertEqual(X.shape[0], y.shape[0])
        self.assertEqual(X.shape[1], sequence_length)
        self.assertEqual(X.shape[2], len(self.features))

    def test_model_compilation(self):
        # Test model compilation
        input_shape = (10, len(self.features))  # 10 time steps, number of features
        model = build_model(input_shape, units=50, dropout_rate=0.2, num_classes=len(np.unique(self.mock_data[self.label])))
        # Model should compile successfully if build_model is correct
        self.assertIsNotNone(model)

    def test_model_prediction_shape(self):
        # Test that model prediction output has the correct shape
        input_shape = (10, len(self.features))
        model = build_model(input_shape, units=50, dropout_rate=0.2, num_classes=len(np.unique(self.mock_data[self.label])))
        # Generate a batch of "sequences"
        sequences = np.random.randn(5, *input_shape)
        predictions = model.predict(sequences)
        self.assertEqual(predictions.shape, (5, len(np.unique(self.mock_data[self.label]))))

if __name__ == '__main__':
    unittest.main()
