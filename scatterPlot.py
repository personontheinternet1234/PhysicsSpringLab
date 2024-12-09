import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "springLabData.csv")
data = pd.read_csv(file_path)
print(data.head())

def extract_columns(data):
    distance = data['Distance (cm)'].tolist()
    forceBig = data['Force (Big) (N)'].tolist()
    forceSmall = data['Force (Small) (N)'].tolist()
    return {
        'distance': distance,
        'forceBig': forceBig,
        'forceSmall': forceSmall
    }

def create_plot(distance, force):
    plt.scatter(distance, force)
    plt.title("Hooke's Law: Force (N) vs Displacement (m)")
    plt.xlabel("Displacement (m)")
    plt.ylabel("Force (N)")
    plt.show()

data_columns = extract_columns(data)
create_plot(data_columns['distance'], data_columns['forceBig'])
# print(data_columns['distance'])
# print(data_columns['forceBig'])
# print(data_columns['forceSmall'])