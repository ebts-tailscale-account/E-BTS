#!/usr/bin/env python3
"""
See COPYING file distributed along with the pyForceDAQ copyright and license terms.
"""

__author__ = "Oliver Lindemann"

from forceDAQ.force import DataRecorder, SensorSettings, SensorProcess
from forceDAQ._lib.types import ForceData, DAQEvents
from forceDAQ._lib.timer import Timer
import logging
import matplotlib.pyplot as plt
import time
import pandas as pd
from pathlib import Path
import os


def fd_count():
    try:
        return len(os.listdir("/proc/self/fd"))
    except Exception:
        return -1


def main():

    # create a sensor
    sensor1 = SensorSettings(device_id="1",
                                calibration_folder="calibration",
                                sensor_name="FT12876",
                                reverse_parameter_names="Fz")
    
    # Create a data recorder
    recorder = DataRecorder(force_sensor_settings=[sensor1],
                            poll_udp_connection=False, polling_priority="normal")

    print("Setting bias, do not touch the sensor!")
    recorder.determine_biases(n_samples=100)
    print("Sensor callibrated")

    # recorder.open_data_file(filename="test", subdirectory="recorded_data", zipped=True,
    #                     time_stamp_filename=False,   comment_line="")

    # recorder.start_recording()
    # time.sleep(0.2)
    # data = recorder.quit()
    # print ("quitted")

    for i in range(5):
        timer = Timer()
        filename = "ft_200samples_avg.csv"
        file_path = Path("recorded_data") / (filename + ".gz")

        # Delete old file if it exists
        if file_path.exists():
            print("yes")
            os.remove(file_path)

        
        # Create a data recorder
        # recorder = DataRecorder(force_sensor_settings=[sensor1],
        #                         poll_udp_connection=False, polling_priority="normal")

        print("Setting bias, do not touch the sensor!")
        recorder.determine_biases_fake(n_samples=100)
        print("Sensor callibrated")

        recorder.open_data_file(filename=filename, subdirectory="recorded_data", zipped=True,
                            time_stamp_filename=False,   comment_line="")

        print ("start recording")
        # x = input('Press any key to start recording... (or press q to stop)')
        # if x=='q':
        #     print('aborted')
        #     return
        start = time.time()
        recorder.start_recording()
        # x = input('Press any key to stop recording... ')
        stop = time.time()
        # recorder.save_daq_event(3_000_000_000) # Set marker code in file time=(stop-start)*1e9, code="code"
        time.sleep(0.2)
        print("Data recording completed for file: {}".format("recorded_data" + '/' + filename))
        # data = recorder.quit()
        buffer = recorder.pause_recording()
        recorder.close_data_file()
        # print(data)
        print ("quitted")


        # Read CSV directly from gzip (skip header comment)
        df = pd.read_csv("recorded_data/"+filename+".gz", comment="#", compression="gzip")

        # Compute average of Fz column
        fz_mean = df["Fz"].mean()

        print(f"Average Fz: {fz_mean:.6f}")
        print(f"[DEBUG] open FDs: {fd_count()}")





if __name__  == "__main__":
    main()

