import os
import random
import gzip
import pandas as pd
import matplotlib.pyplot as plt
import shutil #ne_smeshno

i = 0
# with gzip.open('file.txt.gz', 'rb') as f_in:
#     with open('file.txt', 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)

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
    end_time = 7 * 10 ** 9  # 7 seconds in nanoseconds
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
    plt.pause(2)  # Pause for 2 seconds
    plt.close()  # Close the plot after 2 seconds

def open_dir(root):
    folders = [folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))]
    # if len(folders) > 3:
    #     selected_folders = random.sample(folders, 3)
    # else:
    #     selected_folders = folders
        
    for folder in os.listdir(root):
        # file_count = sum(1 for item in os.listdir(root) if os.path.isfile(os.path.join(root, item)))
        # print(file_count)
        
        new_path = os.path.join(root, folder)
        # print(new_path)
        if os.path.isdir(new_path):
            open_dir(new_path)
        
        items = [item for item in os.listdir(root) if os.path.isfile(os.path.join(root, item))]
        if len(items) > 3:
            selected_items = random.sample(items, 3)
        else:
            selected_items = items

        if len(selected_items) > 0:
            for item in selected_items:
                new_path = os.path.join(root, item)
                df = read_csv_from_gz(new_path)
                if df is not None:
                    # print(f"Processing file: {new_path}")
                    # print(f"Force graph for {os.path.basename(os.path.normpath(root))}")

                    # Normalize the path
                    normalized_path = os.path.normpath(new_path)
                    # Split the path into parts
                    path_parts = normalized_path.split(os.sep)
                    # Extract the desired portion (set1/sfc1/1/sh3)
                    desired_portion = "/".join(path_parts[-4:])  # Last 5th to last 2nd parts
                    try:
                        plot_graph(df, desired_portion, 'time', 'Fx', 'Fy', 'Fz')
                    except Exception as e:
                        print(f"Error drwaing graph: {e}")
            return 
        else:
            continue


if __name__ == '__main__':
    root = "/home/rustam/ur_ws/src/ati_daq_pack/include/pyForceDAQ/tactile_glove/set1"
    open_dir(root)