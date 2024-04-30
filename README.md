# Golf Classification Using an LSTM Network
---
This program provides a Graphical User Interface (GUI) which can be used to classify 3 different golf swings: Drive, Chip and Putt. The GUI allows the user to import their dataset in the format of a CSV file and run the classification. The classification executes the LSTM with the unseed dataset and displays the prediction on the interface. Once the classification has been ran, metrics such as accuracy, precision, recall and F1 Score are also displayed. A 3D scatter graph is also displayed on the interface which plots the dataset provided by the user, allowing them to analyse their golf swing.

## Features
---

- Import a CSV file for classification
- Change the appearance of the graphical user interface to either 'Light' or 'Dark' mode
- Displays a scatter plot of the dataset
- Displays the type of shot the model predicted
- Displays the different metrics for the performance of the model

## Installation and Usage
---
This section specifies the steps required to run the program.

### Installing the required Python Libraries
```sh
pip install customTkinter
pip install pandas
pip install matplotlib 
```

### Running The Program
Go to the directory of where the code is saved on your local machine through the terminal. This can be done by using the commands:

#### For MacOS and Linux
```sh
cd someDirectory # Changes the directory to the specified one
cd .. # Moves the directory back to the previous directory
ls # Provides a list of files and directories in the current directory
```
Then run this command to start the program:
```sh
python3 main.py 
```

#### For Windows
```sh
cd someDirectory # Changes the directory to the specified one
cd .. # Moves the directory back to the previous directory
dir # Provides a list of files and directories in the current directory
```
Then run this command to start the program:
```sh
python main.py 
```

## Running The Unit Tests
---
Some of the unit tests are in their own separate files and some are within the function files. To run the unit tests, enter these commands into the terminal:

### For MacOS and Linux 

```sh
python3 -m unittest test_LSTMTrain.py # Example of the tests within their own file
python3 -m unittest LSTM.py # Example of the tests within the function file
```

### For Windows
```sh
python -m unittest test_LSTMTrain.py # Example of the tests within their own file
python -m unittest LSTM.py # Example of the tests within the function file
```

## MIT License
