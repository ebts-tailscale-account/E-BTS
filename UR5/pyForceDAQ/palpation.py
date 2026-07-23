#!/usr/bin/env python3
# """
# See COPYING file distributed along with the pyForceDAQ copyright and license terms.
# """

# __author__ = "Oliver Lindemann"

# import rospy
# from std_msgs.msg import String
# from forceDAQ.force import DataRecorder, SensorSettings, SensorProcess
# from forceDAQ._lib.timer import Timer
# import logging
# import os
# import time
# from my_msgs.msg import directory

# def shot_callback(msg):
#     """ ROS callback function to handle messages received on the /shot topic. """
#     print("Received trigger to record data into file: {} {}".format(msg.data1, msg.data2))
#     rospy.loginfo("Received a message")

#     # Determine the full path for the new data file
#     filename = msg.data2 + ".csv"  # Assuming 'data' directory is in the working dir

#     # Start new recording session
#     recorder.open_data_file(filename=filename, subdirectory=msg.data1, zipped=True,
#                            time_stamp_filename=False,   comment_line="")
#     recorder.start_recording()
#     rospy.loginfo("Started recording")
#     try:
#         while 1:
#             time.sleep(0.01)
#     except KeyboardInterrupt:
#         pass        

#     # Record for a fixed period of time or until another condition or message instructs to stop
#     #timer.wait(1000)  # Adjust time as necessary for your requirements
#     recorder.save_daq_event(1000)  # Ensure to adjust according to your recording duration

#     print("Data recording completed for file: {}".format(msg.data1 + '/' + filename))
#     # recorder.quit()  # Close the file and properly shutdown the recording session
#     recorder.pause_recording()
#     rospy.loginfo("Finished recording")
#     recorder.close_data_file()

# if __name__ == "__main__":
#     rospy.init_node('data_recording_node', anonymous=True)

#     timer = Timer()

#     # Create a sensor
#     sensor1 = SensorSettings(device_id="1",
#                              calibration_folder="/home/rustam/ur_ws/src/ati_daq_pack/include/pyForceDAQ/",
#                              sensor_name="FT12877", reverse_parameter_names="Fz")  # FIXME: Handle these parameters properly

#     # Create a data recorder
#     recorder = DataRecorder(force_sensor_settings=[sensor1],
#                             poll_udp_connection=False, polling_priority="normal")

#     print("Setting bias, do not touch the sensor!")
#     recorder.determine_biases(n_samples=100)  # Determine bias once at the beginning

#     # Subscribe to the /shot topic
#     rospy.Subscriber("/shot", directory, shot_callback)

#     print("Ready to record data on trigger from /shot topic.")
#     rospy.spin()  # Keep the node running until shut down.












# #!/usr/bin/env python3
# """
# See COPYING file distributed along with the pyForceDAQ copyright and license terms.
# """

# __author__ = "Oliver Lindemann"

from forceDAQ.force import DataRecorder, SensorSettings, SensorProcess
from forceDAQ._lib.types import ForceData, DAQEvents
from forceDAQ._lib.timer import Timer
import logging
import matplotlib.pyplot as plt
from time import gmtime, strftime
import os
import gzip
import pandas as pd

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

    # Subtract the first time value to normalize time to start from 0
    df['time'] -= df['time'].iloc[0]
    start_time = 0
    end_time = 7 * 10 ** 9  # 7 seconds in nanoseconds
    df = df[(df['time'] >= start_time) & (df['time'] <= end_time)]
    
    # Plot the graph
    plt.figure(figsize=(20, 12))
    
    plt.plot(df[label_y], df[label_x1], label=label_x1, color='red')
    plt.plot(df[label_y], df[label_x2], label=label_x2, color='blue')
    plt.plot(df[label_y], df[label_x3], label=label_x3, color='green')
    plt.title(graph_name)
    plt.xlabel(label_y +", s")
    plt.ylabel(f'Force, N: {label_x1}, {label_x2}, {label_x3}')
    plt.legend()
    plt.grid(True)
    plt.gca().set_facecolor('white')

    plt.show(block=False)  # Show the plot without blocking code execution
    # plt.pause(2)  # Pause for 2 seconds
    input("Press Enter to close the plot...")
    plt.close()




if __name__  == "__main__":
    timer = Timer()

    root_dir = "tactile_glove"  ##  don't change
    experiment_set = "set1"     ##  "set1", "set2", so on...
    mode = "sfc1"                 ##  "bh"  OR "sfc1"
    subject_id = "4"            ##  "1" to "8" participants
    print(f" Root Directory: {root_dir},\n Experiment Set: {experiment_set},\n Mode: {mode},\n Subject ID: {subject_id}")
    prototype = input("Enter your prototype index (ex. sh1): ")
    thr_passed = False
    data = []

    # checks if out dir is okay
    output_dir = root_dir + '/' + experiment_set + '/' + mode + '/' + subject_id + '/' + prototype
    if not os.path.isdir(output_dir):
        print(output_dir)
        raise Exception("directory doesn't exists or incorrect prototype index!")
        

    filename = "rec_" + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + ".csv"

    # sensor settings
    sensor1 = SensorSettings(device_id="1",
                             calibration_folder="/home/rustam/ur_ws/src/ati_daq_pack/include/pyForceDAQ/", sensor_name="FT12877", reverse_parameter_names="Fz") #FIXME

    # create a data recorder
    recorder = DataRecorder(force_sensor_settings= [sensor1],
                            poll_udp_connection=False, polling_priority="normal")
    recorder.open_data_file(filename, subdirectory=output_dir, zipped=True,
                           time_stamp_filename=False,   comment_line="")

    print ("\nSetting bias, do not touch the sensor! please wait just 1 sec\n")
    recorder.determine_biases(n_samples=500)
    
    timer.wait(1_000_000_000)
    # print ("Started polling sensor data.")

    time_axis = 7_000_000_000       # 3 seconds
    recorder.start_recording()
    proc = recorder.force_sensor_processes()
    # while not thr_passed:
    #     for fs in proc:
    #         if fs.get_thr_flag:
    #             thr_passed = True
    print("You can start experiment.\nRecording starts once Fz reading exceeds threshold of +0.2N.\n")
    while not thr_passed:
        if proc[0].get_thr_flag():
            thr_passed = True
            print("Start Recording! Threshold passed")
            break

    timer.wait(time_axis)
    recorder.save_daq_event(time_axis)
    print ("Stop Recording")

    data = recorder.quit()
    print ("quitted")

    # plotting the recorded data
    pre_path = "/home/rustam/ur_ws/src/ati_daq_pack/include/pyForceDAQ/"
    full_path = pre_path + output_dir + '/' + filename + '.gz'
    timer.wait(1_000_000)
    df = read_csv_from_gz(full_path)
    if df is not None:
        normalized_path = os.path.normpath(full_path)
        path_parts = normalized_path.split(os.sep)
        desired_portion = "/".join(path_parts[-4:])  # Last 5th to last 2nd parts
        try:
            plot_graph(df, desired_portion, 'time', 'Fx', 'Fy', 'Fz')
        except Exception as e:
            print(f"Error drawing graph: {e}")
