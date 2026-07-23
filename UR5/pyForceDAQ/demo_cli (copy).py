#!/usr/bin/env python3
"""
See COPYING file distributed along with the pyForceDAQ copyright and license terms.
"""

__author__ = "Oliver Lindemann"

import rospy
from std_msgs.msg import String
from forceDAQ.force import DataRecorder, SensorSettings, SensorProcess
from forceDAQ._lib.timer import Timer
import logging
import os
import time
from my_msgs.msg import directory

def shot_callback(msg):
    """ ROS callback function to handle messages received on the /shot topic. """
    print("Received trigger to record data into file: {} {}".format(msg.data1, msg.data2))
    rospy.loginfo("Received a message")

    # Determine the full path for the new data file
    filename = msg.data2 + ".csv"  # Assuming 'data' directory is in the working dir

    # Start new recording session
    recorder.open_data_file(filename=filename, subdirectory=msg.data1, zipped=True,
                           time_stamp_filename=False,   comment_line="")
    recorder.start_recording()
    rospy.loginfo("Started recording")
    try:
        while 1:
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass        

    # Record for a fixed period of time or until another condition or message instructs to stop
    #timer.wait(1000)  # Adjust time as necessary for your requirements
    recorder.save_daq_event(1000)  # Ensure to adjust according to your recording duration

    print("Data recording completed for file: {}".format(msg.data1 + '/' + filename))
    # recorder.quit()  # Close the file and properly shutdown the recording session
    recorder.pause_recording()
    rospy.loginfo("Finished recording")
    recorder.close_data_file()

if __name__ == "__main__":
    rospy.init_node('data_recording_node', anonymous=True)

    timer = Timer()

    # Create a sensor
    sensor1 = SensorSettings(device_id="1",
                             calibration_folder="/home/rustam/ur_ws/src/ati_daq_pack/include/pyForceDAQ/",
                             sensor_name="FT12877", reverse_parameter_names="Fz")  # FIXME: Handle these parameters properly

    # Create a data recorder
    recorder = DataRecorder(force_sensor_settings=[sensor1],
                            poll_udp_connection=False, polling_priority="normal")

    print("Setting bias, do not touch the sensor!")
    recorder.determine_biases(n_samples=100)  # Determine bias once at the beginning

    # Subscribe to the /shot topic
    rospy.Subscriber("/shot", directory, shot_callback)

    print("Ready to record data on trigger from /shot topic.")
    rospy.spin()  # Keep the node running until shut down.












# #!/usr/bin/env python3
# """
# See COPYING file distributed along with the pyForceDAQ copyright and license terms.
# """

# __author__ = "Oliver Lindemann"

# from forceDAQ.force import DataRecorder, SensorSettings, SensorProcess
# from forceDAQ._lib.types import ForceData, DAQEvents
# from forceDAQ._lib.timer import Timer
# import logging
# import matplotlib.pyplot as plt

# if __name__  == "__main__":
#     timer = Timer()

#     # create a sensor
#     sensor1 = SensorSettings(device_id="1",
#                              calibration_folder="/home/rustam/ur_ws/src/ati_daq_pack/include/pyForceDAQ/", sensor_name="FT12877", reverse_parameter_names="Fz") #FIXME

#     # create a data recorder
#     recorder = DataRecorder(force_sensor_settings= [sensor1],
#                             poll_udp_connection=False, polling_priority="normal")
#     recorder.open_data_file("outdata", subdirectory="data", zipped=True,
#                            time_stamp_filename=False,   comment_line="")

#     print ("setting bias, not touch the sensor!")
#     recorder.determine_biases(n_samples=100)
#     data = []
#     timer.wait(1000)
#     print ("start recording")

#     time_axis = 3000
#     recorder.start_recording()
#     timer.wait(time_axis)
#     recorder.save_daq_event(time_axis)

#     print ("pause recording")

#     data = recorder.quit()
#     print ("quitted")
