__author__ = 'Oliver Lindemann'

import atexit
import ctypes as ct
from multiprocessing import Process, Event, sharedctypes, Pipe
import logging

from .._lib.types import DAQEvents
from .._lib.timer import app_timer
from .._lib.polling_time_profile import PollingTimeProfile
from .._lib.process_priority_manager import get_priority

from .sensor import SensorSettings, Sensor

from collections import deque

class SensorProcess(Process):
    def __init__(self, settings, pipe_buffered_data_after_pause=True,
                  chunk_size=10000):
        """ForceSensorProcess

        return_buffered_data_after_pause: does not write shared data queue continuously and
            writes it the buffer data to queue only after pause (or stop)

        """

        # DOC explain usage

        # type checks
        if not isinstance(settings, SensorSettings):
            raise RuntimeError(
                "settings has to be force_sensor.Settings object")

        super(SensorProcess, self).__init__()
        self.sensor_settings = settings
        self._pipe_buffer_after_pause = pipe_buffered_data_after_pause
        self._chunk_size = chunk_size

        self._pipe_i, self._pipe_o = Pipe()
        self._event_is_polling = Event()
        self._event_sending_data = Event()
        self._event_new_data = Event()
        self.event_bias_is_available = Event()
        self.event_trigger = Event()
        self.thr_passed = Event()

        self._last_Fx = sharedctypes.RawValue(ct.c_float)
        self._last_Fy = sharedctypes.RawValue(ct.c_float)
        self._last_Fz = sharedctypes.RawValue(ct.c_float)
        self._last_Tx = sharedctypes.RawValue(ct.c_float)
        self._last_Ty = sharedctypes.RawValue(ct.c_float)
        self._last_Tz = sharedctypes.RawValue(ct.c_float)
        self._buffer_size = sharedctypes.RawValue(ct.c_uint64)
        self._sample_cnt = sharedctypes.Value(ct.c_uint64)
        self._event_quit_request = Event()
        self._determine_bias_flag = Event()
        self._determine_bias_flag_fake = Event()

        self._bias_n_samples = 200
        atexit.register(self.join)

    @property
    def Fx(self):
        return self._last_Fx.value

    @property
    def Fy(self):
        return self._last_Fy.value

    @property
    def Fz(self):
        return self._last_Fz.value

    @property
    def Tx(self):
        return self._last_Tx.value

    @property
    def Ty(self):
        return self._last_Ty.value

    @property
    def Tz(self):
        return self._last_Tz.value

    def get_force(self, parameter_id):
        if   parameter_id == 0: return self._last_Fx.value
        elif parameter_id == 1: return self._last_Fy.value
        elif parameter_id == 2: return self._last_Fz.value
        elif parameter_id == 3: return self._last_Tx.value
        elif parameter_id == 4: return self._last_Ty.value
        elif parameter_id == 5: return self._last_Tz.value
        else: return None

    def get_Fxyz(self):
        return (self._last_Fx.value, self._last_Fy.value, self._last_Fz.value)

    def Txyz(self):
        return (self._last_Tx.value, self._last_Ty.value, self._last_Tz.value)

    @property
    def sample_cnt(self):
        return self._sample_cnt.value

    def get_sample_cnt(self):
        return int(self._sample_cnt.value)

    def get_buffer_size(self):
        return int(self._buffer_size.value)

    def determine_bias(self, n_samples=100):
        """recording is paused after bias determination

        Bias determination is only possible while pause.
        This process might take a while. Please use "wait_bias_available" to
        ensure that the process is finished and the sensor is again read for
        recording.
        """

        if not self._event_is_polling.is_set():
            self._bias_n_samples = n_samples
            self.event_bias_is_available.clear()
            self._determine_bias_flag.set()

    def determine_bias_fake(self, n_samples=100):
        """recording is paused after bias determination

        Bias determination is only possible while pause.
        This process might take a while. Please use "wait_bias_available" to
        ensure that the process is finished and the sensor is again read for
        recording.
        """

        if not self._event_is_polling.is_set():
            self._bias_n_samples = n_samples
            self.event_bias_is_available.clear()
            self._determine_bias_flag_fake.set()
            
    def get_thr_flag(self):
        return self.thr_passed.wait()

    def start_polling(self):
        self._event_is_polling.set()

    def pause_polling(self):
        self._event_is_polling.clear()

    def get_buffer(self, timeout=1.0):
        """return recorded buffer"""
        rtn = []
        # print("hey")
        # print(self._event_sending_data)
        if self._event_sending_data.is_set() or self._buffer_size.value > 0:
            # print("kkk")
            self._event_sending_data.wait()
            # print(self._buffer_size.value)
            while self._buffer_size.value > 0:  # wait until buffer is empty
                rtn.extend(self._pipe_i.recv())
                # print(self._pipe_i.recv())
            self._event_sending_data.clear() # stop sending
            # print("fetbuff")
            # print(rtn[0])
            logging.debug(f"Received data: {rtn}")
        return rtn

    def join(self, timeout=None):

        if self._event_is_polling.is_set():
            self.pause_polling()
            app_timer.wait(100)
            self.get_buffer() # empty buffer, required to quit process run loop

        self._event_quit_request.set()
        super(SensorProcess, self).join(timeout)

        


    def run(self):
        buffer = []
        self._buffer_size.value = 0
        sensor = Sensor(self.sensor_settings)

        self._event_is_polling.clear()
        self._event_sending_data.clear()
        is_polling = False
        ptp = PollingTimeProfile() #TODO just for testing?
        # Initialize a deque with a maximum size of 10
        window = deque(maxlen=10)
        iter = 0
        while iter < 9:
            window.append(0.55555555)
            iter = iter + 1
        
        iter = 0
        # self.thr_passed.clear()
        self.thr_passed.set()

        while not self._event_quit_request.is_set():
            if self._event_is_polling.is_set():
                # is polling
                if not is_polling:
                    # start NI device and acquire one first dummy sample to
                    # ensure good timing
                    sensor.start_data_acquisition()
                    # buffer.append(DAQEvents(time=sensor.timer.time,
                    #                         code="started:"+repr(sensor.device_id)))
                    logging.info("Sensor start, name {},  pid {}, priority {}".format(
                        self.pid, sensor.name, get_priority(self.pid)))

                    self._buffer_size.value = len(buffer)
                    is_polling = True

                d = sensor.poll_data()
                ptp.update(d.time)

                self._last_Fx.value, self._last_Fy.value, self._last_Fz.value, \
				                     self._last_Tx.value, self._last_Ty.value, \
                                     self._last_Tz.value = d.forces
                self._sample_cnt.value += 1
                if self.event_trigger.is_set():
                    self.event_trigger.clear()
                    d.trigger[0] = 1

                if float(self._last_Fz.value) > (sum(window) / 10 + 0.2) and not self.thr_passed.is_set(): 
                    self.thr_passed.set()
                    buffer.append(DAQEvents(time=sensor.timer.time,
                                            code="Fz_value:"+repr(float(self._last_Fz.value))+",Moving_threshold:" + repr(sum(window) / 10 + 0.2)+","))
                    # print("Threshold passed!")

                if iter == 10:
                    window.append(float(self._last_Fz.value))

                if self.thr_passed.is_set():
                    buffer.append(d)
                    
                # print(d.forces)
                
                self._buffer_size.value = len(buffer)
                # if self._buffer_size.value % 20 == 0:                         #Comment this
                #     window.append(float(self._last_Fz.value))
                #     print("#")
                #     print("Fz: ")
                #     print(float(self._last_Fz.value)) 
                #     print("Average: ")
                #     print(sum(window) / 10)
                iter = iter + 1

            else:
                # pause: not polling
                if is_polling:
                    sensor.stop_data_acquisition()
                    buffer.append(DAQEvents(time=sensor.timer.time,
                                            code="pause:"+repr(sensor.device_id)))
                    self._buffer_size.value = len(buffer)
                    logging.info("Sensor stop, name {}".format(
                        self.pid, sensor.name, get_priority(self.pid)))
                    is_polling = False
                    ptp.stop()

                if self._pipe_buffer_after_pause and self._buffer_size.value>0:
                    # sending data to force
                    self._event_sending_data.set()
                    chks = self._chunk_size
                    # print("ru")
                    # print(self._event_sending_data)
                    while len(buffer)>0:
                        if chks > len(buffer):
                            chks = len(buffer)
                        self._pipe_o.send(buffer[0:chks])
                        # print(buffer[0:chks])
                        buffer[0:chks] = []
                        self._buffer_size.value = len(buffer)
                        # print("in sending" + str(self._buffer_size.value))

                    while self._event_sending_data.is_set():
                        sensor.timer.wait(2)

                if self._determine_bias_flag.is_set():
                    sensor.determine_bias(n_samples=self._bias_n_samples)
                    self._determine_bias_flag.clear()
                    self.event_bias_is_available.set()

                if self._determine_bias_flag_fake.is_set():
                    sensor.determine_bias_fake(n_samples=self._bias_n_samples)
                    self._determine_bias_flag_fake.clear()
                    self.event_bias_is_available.set()

                self._event_is_polling.wait(timeout=0.1)

        
        # stop process
        sensor.stop_data_acquisition()
        self._buffer_size.value = 0

        logging.info("Sensor quit, {}, {}".format(
            sensor.name, ptp.get_profile_str()))
