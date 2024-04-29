import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('Data/trainingData.csv')
features = ['accelerationX', 'rotationRateZ', 'rotationRateY', 'quaternionW']
label = 'label'

# Encode labels
label_encoder = LabelEncoder()
data[label] = label_encoder.fit_transform(data[label])

# Select features and normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])

# Define sequence length
sequence_length = 10

# Generate sequences
def create_sequences(X, y, seq_length):
    xs, ys = [], []
    for i in range(len(X) - seq_length + 1):
        xs.append(X[i:(i + seq_length)])
        ys.append(y[i + seq_length - 1])
    return np.array(xs), np.array(ys)

X, y = create_sequences(X_scaled, data[label].values, sequence_length)

# Define model building function for multi-class classification
def build_model(input_shape, units=200, dropout_rate=0.2, num_classes=4):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')  # Output layer with softmax for multi-class classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Define KFold cross-validation
num_folds = 5
kf = KFold(n_splits=num_folds)

# Initialize variables to track the best model and its performance
best_model = None
best_model_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}

# Train and evaluate model across folds
for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    model = build_model((sequence_length, len(features)), units=100, dropout_rate=0.2, num_classes=4)
    model.fit(X_train, y_train, epochs=10, batch_size=64, verbose=1)
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Update best metrics and best model if current values are higher
    if accuracy > best_model_metrics['accuracy']:
        best_model_metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
        best_model = model

    print(f"Fold {fold} - Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1: {f1}")

# Check if a best model has been saved
if best_model is not None:
    best_model.save('Model/trainedLSTM.keras')
    print("Save Success: Model was successfully saved!")
    print("====================================================")
    print(f"Highest Accuracy: {best_model_metrics['accuracy']}")
    print(f"Highest Precision: {best_model_metrics['precision']}")
    print(f"Highest Recall: {best_model_metrics['recall']}")
    print(f"Highest F1 Score: {best_model_metrics['f1']}")
else:
    raise Exception("Save Failed: No model was found!")

