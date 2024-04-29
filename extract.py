import pandas as pd

def extract_and_save_features(csv_file_path, output_file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(csv_file_path)
    
    # Specify the columns to extract
    columns_to_extract = ['accelerationX','rotationRateZ', 'rotationRateY', 'quaternionW' , 'label']
    
    # Check if all required columns are in the DataFrame
    missing_columns = [col for col in columns_to_extract if col not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the CSV file: {missing_columns}")
    
    # Extract the necessary columns
    extracted_data = data[columns_to_extract]
    
    # Save the extracted data to a new CSV file
    extracted_data.to_csv(output_file_path, index=False)
    # print(f"Data successfully saved to {output_file_path}")

# Specify the input and output file paths
input_csv_path = 'Data/GolfSwingData-interpolate.csv'  # Replace with your input file path
output_csv_path = 'Data/trainingData.csv'  # Define your desired output file path

# Call the function with the specified paths
extract_and_save_features(input_csv_path, output_csv_path)
