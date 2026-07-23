#!/usr/bin/env python3
import os
import random
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import shutil #ne_smeshno


def read_csv_from_gz(file_path):
    """
    param file_path: Path to the .gz file containing the CSV
    return: A pandas DataFrame with the CSV data
    """
    try:
        with gzip.open(file_path, 'rt', encoding='utf-8') as gz_file:
            df = pd.read_csv(gz_file, skiprows=[0, 2])
            for col in ['Fx', 'Fy', 'Fz', 'time']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=['Fx', 'Fy', 'Fz', 'time'])
        return df
    except Exception as e:
        print(f"Error opening file: {e}")
        return None


def plot_graph(df, graph_name, label_y, label_x1, label_x2, label_x3):
    # Convert the time column to numeric, coercing errors
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    
    # # Filter the DataFrame for a specific interval (if needed)
    # start_time = df['time'].iloc[0]
    # end_time = start_time + 2 * 10 ** 9  # 2 seconds in nanoseconds
    # df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]

    # Subtract the first time value to normalize time to start from 0
    df['time'] -= df['time'].iloc[0]
    start_time = 0
    end_time = df['time'].values[-1]#7 * 10 ** 9  # 7 seconds in nanoseconds
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    # Plot the graph
    plt.figure(figsize=(20, 12))
    # plt.scatter(df[label_y], df[label_x1], color='blue', alpha=0.5, label='Data points')
    
    plt.plot(df[label_y], df[label_x1], label=label_x1, color='red')
    plt.plot(df[label_y], df[label_x2], label=label_x2, color='blue')
    plt.plot(df[label_y], df[label_x3], label=label_x3, color='green')
    plt.title(graph_name)
    plt.xlabel(label_y +", s")
    plt.ylabel(f'Force, N: {label_x1}, {label_x2}, {label_x3}')
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('white')
    
    # Show the plot for 2 seconds, then close it
    plt.show(block=False)  # Show the plot without blocking code execution
    # plt.pause(2)  # Pause for 2 seconds
    input("Press Enter to close the plot...")
    plt.close()  # Close the plot after 2 seconds



if __name__ == '__main__':
    root = "/home/rustam/ur_modern_driver_ros1_eggs/src/ati_daq_pack/include/pyForceDAQ/data/16.csv.gz"
    df = read_csv_from_gz(root)
    if df is not None:
        normalized_path = os.path.normpath(root)
        path_parts = normalized_path.split(os.sep)
        desired_portion = "/".join(path_parts[-4:])  # Last 5th to last 2nd parts
        try:
            plot_graph(df, desired_portion, 'time', 'Fx', 'Fy', 'Fz')
        except Exception as e:
            print(f"Error drwaing graph: {e}")