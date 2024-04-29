import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Function to plot acceleration graph
def plot_acceleration(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['accelerationX'], df['accelerationY'], df['accelerationZ'], c='r', marker='o')
    ax.set_xlabel('Acceleration X')
    ax.set_ylabel('Acceleration Y')
    ax.set_zlabel('Acceleration Z')
    ax.set_title('3D Scatter Plot of Acceleration')
    plt.show()

# Function to plot rotation rate graph
def plot_rotation_rate(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['rotationRateX'], df['rotationRateY'], df['rotationRateZ'], c='b', marker='^')
    ax.set_xlabel('Rotation Rate X')
    ax.set_ylabel('Rotation Rate Y')
    ax.set_zlabel('Rotation Rate Z')
    ax.set_title('3D Scatter Plot of Rotation Rate')
    plt.show()

def plot_gravity(df):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['gravityX'], df['gravityY'], df['gravityZ'], c='b', marker='^')
    ax.set_xlabel('Gravity X')
    ax.set_ylabel('Gravity Y')
    ax.set_zlabel('Gravity Z')
    ax.set_title('3D Scatter Plot of Gravity')
    plt.show()

def main():
    file_path = 'Data/WristMotion.csv' 
    
    try:
        # Load the data
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
    # Ask the user which graph to display
    graph_choice = input("Which graph to display? (1 for Acceleration, 2 for Rotation Rate, 3 for Gravity): ").strip()
    
    # Display the chosen graph
    if graph_choice == '1':
        plot_acceleration(df)
    elif graph_choice == '2':
        plot_rotation_rate(df)
    elif graph_choice == '3':
        plot_gravity(df)  # Assuming you have a function to plot gravity data
    else:
        print("Invalid input. Please enter 1 for Acceleration, 2 for Rotation Rate, or 3 for Gravity.")

if __name__ == "__main__":
    main()

