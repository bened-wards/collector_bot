import serial
from enum import Enum
import time
import math
import multiprocessing
import ctypes
import logging

debug = False

class MsgType(Enum):
    velocity = 1
    state = 2
    fail = 3

class State:
    x = 0.0 # metres
    y = 0.0 # metres
    theta = 0.0 # radians

class VelocityState:
    v = 0.0 # m/s
    w = 0.0 # rad/s

class MotionController:

    def __init__(self, device_str='/dev/ttyS0', baud_rate=115200, timeout=1):
        self._serial = serial.Serial(device_str, baud_rate, timeout=timeout)
        self._device_str = device_str
        self._baud_rate = baud_rate

        # to make these multiprocessing safe
        self._state = multiprocessing.Array(ctypes.c_double, 3)
        self._velocity_state = multiprocessing.Array(ctypes.c_double, 2)

        self._lock = multiprocessing.Lock()

        self._write_queue = multiprocessing.Queue()
        self._stop_event = multiprocessing.Event()

        self._internal_velocity_state: VelocityState = VelocityState()

    ### WRITING VELOCITY TO PI ZERO VIA SERIAL PORT ###
    ###################################################
    def drive_straight(self, vel_m_per_s):
        self._write_velocity(vel_m_per_s, 0.0)

    def drive_forward(self, lin_speed_m_per_s):
        self._write_velocity(lin_speed_m_per_s, 0.0)
    
    def drive_backward(self, lin_speed_m_per_s):
        self._write_velocity(-lin_speed_m_per_s, 0.0)

    def rotate_cw(self, ang_speed_rad_per_s):
        self._write_velocity(0.0, -ang_speed_rad_per_s)

    def rotate_ccw(self, ang_speed_rad_per_s):
        self._write_velocity(0.0, ang_speed_rad_per_s)

    def rotate(self, ang_vel_rad_per_s):
        self._write_velocity(0.0, ang_vel_rad_per_s)

    def stop(self):
        self._write_velocity(0.0, 0.0)

    def really_stop(self):
        for _ in range(5):
            self.stop()

    ## FOR LINE FOLLOWING ##
    def add_rotation(self, w_added):
        """Useful for line following where we want to add small amount of rotational velocity"""
        v, w = self.get_internal_velocity_state()
        self._write_velocity(v, w + w_added)

    def drive_straight_with_rotation(self, w):
        """Useful for line following where want to keep same linear velocity but set a small rotational velocity"""
        v, _ = self.get_internal_velocity_state()
        self._write_velocity(v, w)

    @staticmethod
    def get_velocity_str(v, w):
        """v+0.000w-0.000\n"""
        return f"v{'+' if v >= 0 else '-'}{abs(v):.3f}w{'+' if w >= 0 else '-'}{abs(w):.3f}\n"

    def _write_velocity(self, v, w):
        """Put velocity message into write queue to be picked up be writer process"""
        self.set_internal_velocity_state(v, w)
        self._write_queue.put(self.get_velocity_str(v,w))

    def reset_state_at_box(self, x, y, theta):
        self._write_state(x, y, theta)

    @staticmethod
    def get_state_str(x, y, theta):
        """x+0.000y+0.000t-0.000\n"""
        return f"x{'+' if x >= 0 else '-'}{abs(x):.3f}y{'+' if y >= 0 else '-'}{abs(y):.3f}t{'+' if theta >= 0 else '-'}{abs(theta):.3f}\n"
    
    def _write_state(self, x, y, theta):
        """Put state message into write queue to be picked up by writer process"""
        self._write_queue.put(self.get_state_str(x, y, theta))

    ### GETTERS AND SETTERS ###
    ###########################
    # TODO - may need to update these for threading not multiprocessing
    def get_state(self):
        with self._lock:
            return self._state[0], self._state[1], self._state[2]
    
    def set_state(self, x, y, theta):
        # with self._lock:
        #     self._state.x = x
        #     self._state.y = y
        #     self._state.theta = theta
        with self._lock:
            self._state[0] = x
            self._state[1] = y
            self._state[2] = theta

    def get_velocity_state(self):
        with self._lock:
            return self._velocity_state
    
    def set_velocity_state(self, v, w):
        # with self._lock:
        #     self._velocity_state.v = v
        #     self._velocity_state.w = w
        with self._lock:
            self._velocity_state[0] = v
            self._velocity_state[1] = w

    def get_internal_velocity_state(self):
        return self._internal_velocity_state
    
    def set_internal_velocity_state(self, v, w):
        self._internal_velocity_state.v = v
        self._internal_velocity_state.w = w

    ### READING VELOCITY AND STATE FROM PI ZERO VIA SERIAL PORT ###
    ###############################################################
    def read(self):
        """Reads from serial port (blocks until data available if no timeout is set)"""
        no_decoded = 0
        while self._serial.in_waiting > 0:
            msg = self._serial.readline().decode('utf-8').strip()
            # logging.debug(f"Received msg: {msg}")
            if not(len(msg) > 0):
                return False
            if msg[0] == 'v':
                try:
                    v, w = self.decode_velocity_msg(msg)
                    # logging.debug(f"Decoded velocity: v={v}, w={w}")
                    self.set_velocity_state(v, w)
                    no_decoded += 1
                except ValueError:
                    logging.warning(f"Failed to decode velocity message: {msg}")
            elif msg[0] == 'x':
                try:
                    x, y, theta = self.decode_state_msg(msg)
                    # logging.debug(f"Decoded state: x={x}, y={y}, theta={theta}")
                    self.set_state(x, y, theta)
                    no_decoded += 1
                except ValueError:
                    logging.warning(f"Failed to decode state message: {msg}")
        return no_decoded > 0

    @staticmethod
    def decode_velocity_msg(msg):
        """Expected format: v+0.000w-0.000"""
        v = float(msg[1:7])
        w = float(msg[8:])
        return v, w

    @staticmethod
    def decode_state_msg(msg):
        """Expected format: x+0.000y+0.000t-0.000"""
        x = float(msg[1:7])
        y = float(msg[8:14])
        theta = float(msg[15:])
        return x, y, theta

    ### LOOPS THAT RUN IN SEPARATE PROCESSES ###
    ############################################
    def read_from_serial(self):
        print_counter = 0
        while not self._stop_event.is_set():
            self.read()
            print_counter += 1
            if (print_counter > 200):
                x, y, theta = self.get_state()
                logging.info(f"Current state: x={x}, y={y}, theta={theta*180/math.pi:.4f}")
                print_counter = 0
            time.sleep(0.01) # self.read is blocking so we don't need to sleep for long

    def write_to_serial(self):
        while not self._stop_event.is_set() or not self._write_queue.empty():
            try:
                msg = self._write_queue.get(timeout=0.1)
                self._serial.write(msg.encode('utf-8'))
            except multiprocessing.queues.Empty:
                time.sleep(0.01)

    def set_stop_event(self):
        """Stop read and write processes"""
        self._stop_event.set()



def write_test(mc):
    v = 0.5
    w = -2.0
    start_time = time.time()
    print(f"Sending drive straight: v={v:.3f}\n\n\n")
    while time.time() - start_time < 15:
        mc.drive_straight(v)
        time.sleep(0.1)

    start_time = time.time()
    print(f"\n\n\nSending rotate: w={w:.3f}\n\n\n")
    while time.time() - start_time < 15:
        mc.rotate(w)
        time.sleep(0.1)

def full_control_test(mc: MotionController):
    v = 0.2
    w = -0.5
    start_time = time.time()
    print(f"Sending drive straight: v={v:.3f}\n\n\n")
    print_counter = 0
    while time.time() - start_time < 5:
        mc.drive_forward(v)
        time.sleep(0.1)
        print_counter += 1
        if print_counter > 10:
            x, y, theta = mc.get_state()
            print(f"Current state: x={x}, y={y}, theta={theta}")
            print_counter = 0

    mc.stop()
    start_time = time.time()
    print(f"\n\n\nSending rotate: w={w:.3f}\n\n\n")
    print_counter = 0
    while time.time() - start_time < 5:
        mc.rotate(w)
        time.sleep(0.1)
        print_counter += 1
        if print_counter > 10:
            x, y, theta = mc.get_state()
            print(f"Current state: x={x}, y={y}, theta={theta}")
            print_counter = 0
    mc.stop()

if __name__ == "__main__":
    mc = MotionController()

    reader_proc = multiprocessing.Process(target=mc.read_from_serial, args=())
    writer_proc = multiprocessing.Process(target=mc.write_to_serial, args=())

    reader_proc.start()
    writer_proc.start()

    full_control_test(mc)
    mc.set_stop_event()

    reader_proc.join()
    writer_proc.join()
    

