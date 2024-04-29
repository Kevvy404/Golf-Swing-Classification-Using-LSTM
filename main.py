import customtkinter
from CTkMessagebox import CTkMessagebox
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from LSTM import load_and_predict

customtkinter.set_appearance_mode("Light")
customtkinter.set_default_color_theme("green")

root = customtkinter.CTk()
root.geometry("1080x920")
root.title("")
frame = customtkinter.CTkFrame(master=root)
frame.pack(pady=40, padx=40, fill="both", expand=True)

label = customtkinter.CTkLabel(master=frame, text="Golf Classification using an LSTM Network", font=("Roboto", 24))
label.pack(pady=12, padx=10)

theme_colors = {
    "Light": "#dbdbdb",
    "Dark": "#2b2b2b"
}

# Function to change appearance
def change_appearance(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)
    # Set the background color of the plot based on the new appearance mode
    bg_color = theme_colors.get(new_appearance_mode, "white")
    ax.set_facecolor(bg_color)
    fig.patch.set_facecolor(bg_color)

    # Set the text color of labels based on the theme
    text_color = 'white' if new_appearance_mode != "Light" else 'black'
    ax.title.set_color(text_color)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.zaxis.label.set_color(text_color)
    for line in ax.xaxis.get_ticklines() + ax.yaxis.get_ticklines() + ax.zaxis.get_ticklines():
        line.set_color(text_color)
    for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
        label.set_color(text_color)

    canvas.draw()

theme = customtkinter.CTkOptionMenu(master=frame, values=["Light", "Dark"], command=change_appearance)
theme.pack(pady=14, padx=10)

# Function to import data
def import_data():
    file_path = filedialog.askopenfilename(title="Select a file for classification",
                                           filetypes=(("CSV Files", "*.csv"),("All Files", "*.*")))
    if file_path:
        # Check if the selected file is a CSV
        if not file_path.lower().endswith('.csv'):
            CTkMessagebox(title="Error", message="Invalid File Type: Please select a CSV file.")
        else:
            print("File selected:", file_path)
            selected_file_label.configure(text=f"Selected File: {file_path}")


import_button = customtkinter.CTkButton(master=frame, text="Import Data", command=import_data)
import_button.pack(pady=0, padx=10)

selected_file_label = customtkinter.CTkLabel(master=frame, text="No file selected")
selected_file_label.pack(pady=0, padx=10)

def run_model():
    selected_file = selected_file_label.cget("text").replace("Selected File: ", "").strip()
    if selected_file == "No file selected":
        CTkMessagebox(title="Error", message="Please Select A File")
        return
    
    try:
        # Load data to be used in the plot update and prediction
        data = pd.read_csv(selected_file)
        update_plot(data)  # Update the plot with new data

        # Configure for prediction
        features = ['accelerationX', 'rotationRateZ', 'rotationRateY', 'quaternionW']
        sequence_length = 10
        model_path = 'Model/trainedLSTM.keras'
        label_col = 'label'

        # Make predictions
        results= load_and_predict(model_path, selected_file, features, sequence_length, label_col)

        if "Error" not in results:
            shot_type_var.set(f"Shot Type: {results.get('Prediction')}")
            # print(f"Prediction: {results.get('Prediction')}")
            metrics_vars[0].set(f"{results.get('Accuracy')}")
            metrics_vars[1].set(f"{results.get('Precision')}")
            metrics_vars[2].set(f"{results.get('Recall')}")
            metrics_vars[3].set(f"{results.get('F1 Score')}")
        else:
            return None
            # print(f"Error in prediction: {results.get('Error')}")

        # print(prediction_label)
        # Notify user of successful analysis
        CTkMessagebox(title="Success", message="Analysis complete.")
        return True
    except Exception as e:
        # print(f"Exception occurred: {str(e)}")  # Output the error to the console for debugging
        return False
# Button to run the machine learning model
run_button = customtkinter.CTkButton(master=frame, text="Run", command=run_model)
run_button.pack(pady=10, padx=10)

metrics_frame = customtkinter.CTkFrame(master=frame)
metrics_frame.pack(side="left",pady=5, padx=10, fill='both', expand=False)

# Metrics display sub-frame to use grid layout
metrics_sub_frame = customtkinter.CTkFrame(master=metrics_frame)
metrics_sub_frame.pack(fill='both', expand=True)

# Initialize the shot_type_var with a default message
shot_type_var = customtkinter.StringVar(value="Shot Type: N/A")

# Displaying the shot type using grid in the sub-frame, now with bold font
shot_type_label = customtkinter.CTkLabel(master=metrics_sub_frame, 
                                         textvariable=shot_type_var, 
                                         font=("Roboto", 18, "bold"),  # Making the font bold
                                         anchor="w")
shot_type_label.grid(row=0, column=0, columnspan=2, pady=20, padx=10)

# Placeholder for dynamically updating metrics
metrics_vars = [customtkinter.StringVar(value="0.0") for _ in range(4)]
metrics_labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

for i, metric in enumerate(metrics_labels):
    metric_label = customtkinter.CTkLabel(master=metrics_sub_frame, text=f"{metric}:", anchor="w",
                                          font=("Roboto", 14))
    metric_label.grid(row=i+1, column=0, sticky="w", pady=2, padx=5)
    
    value_label = customtkinter.CTkLabel(master=metrics_sub_frame, textvariable=metrics_vars[i], anchor="w",
                                         font=("Roboto", 14))
    value_label.grid(row=i+1, column=1, sticky="w", pady=2, padx=5)

# Plotting frame
plot_frame = customtkinter.CTkFrame(master=frame)
plot_frame.pack(side="right",pady=25, padx=60, fill="both", expand=True)

inital_bg_color_plot = "#dbdbdb"

# Generating a 3D scatter plot
fig = plt.Figure(figsize=(8,6), dpi=100, facecolor=inital_bg_color_plot)
ax = fig.add_subplot(111, projection='3d', facecolor=inital_bg_color_plot)

ax.set_title('3D Scatter Plot')
ax.set_title('3D Scatter Plot of Golf Swing')
ax.set_xlabel('Gravity X')
ax.set_ylabel('Gravity Y')
ax.set_zlabel('Gravity Z')

# Function to update plot
def update_plot(data):
    ax.clear()
    ax.scatter(data['gravityX'], data['gravityY'], data['gravityZ'], c='r', marker='o')
    ax.set_title('3D Scatter Plot of Golf Swing')
    ax.set_xlabel('Gravity X')
    ax.set_ylabel('Gravity Y')
    ax.set_zlabel('Gravity Z')
    canvas.draw()

# Function to handle zooming
def on_scroll(event):
    delta = event.delta
    if delta > 0:
        zoom_factor = 0.9
    else:
        zoom_factor = 1.1

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    ax.set_xlim(xlim[0] * zoom_factor, xlim[1] * zoom_factor)
    ax.set_ylim(ylim[0] * zoom_factor, ylim[1] * zoom_factor)
    ax.set_zlim(zlim[0] * zoom_factor, zlim[1] * zoom_factor)

    canvas.draw()

canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=customtkinter.TOP, fill=customtkinter.BOTH, expand=True)
canvas_widget.bind("<MouseWheel>", on_scroll)

root.mainloop()
