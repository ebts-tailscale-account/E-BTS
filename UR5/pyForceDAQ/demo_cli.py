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


def main():
    timer = Timer()
    filename = time.strftime("%a_%b_%d_%H-%M-%S_%Y") + ".csv"

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

    recorder.open_data_file(filename=filename, subdirectory="recorded_data", zipped=True,
                           time_stamp_filename=False,   comment_line="")

    #print ("start recording")
    x = input('Press any key to start recording... (or press q to stop)')
    if x=='q':
        print('aborted')
        return
    start = time.time()
    recorder.start_recording()
    x = input('Press any key to stop recording... ')
    stop = time.time()
    recorder.save_daq_event(3_000_000_000) # Set marker code in file time=(stop-start)*1e9, code="code"
    print("Data recording completed for file: {}".format("recorded_data" + '/' + filename))
    data = recorder.quit()
    print ("quitted")

if __name__  == "__main__":
    main()

