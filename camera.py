import cv2
import time
import json
import numpy as np
import logging

from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory # to use hardware PWM

class Camera:
    """Wrapper around an OpenCV camera object"""
    def __init__(self, 
                 camera_id, calibration_file_path, camera_height, camera_pivot_length,
                 camera_angles, camera_thresholds, ball_close_to_distance_threshold,
                 camera_servo_angles, servo_gpio_pin,
                 width=None, height=None):
        """Wrapper around OpenCV camera object

        Args:
            camera_id (int): camera's id for opencv object
            calibration_file_path (str): file path to camera calibration json file (from camera_calibration.py) which has intrinsic matrix and distortion coefficients
            camera_height (float): height in metres above the ground to translate to robot frame
            camera_angles (list(float)): 
                                        angle in degrees from horizontal (positive downwards) to translate to robot frame. 
                                        First element is for 5m -> 60cm
                                        Second element is for 60cm -> 30cm
                                        Third element is for 30cm -> 20cm
            camera_thresholds(list(float)): distance thresholds for each camera angle
            ball_close_to_distance_threshold (float): distance threshold for ball being close to the threshold
            camera_servo_angles (list(float)): angles in degrees for the servo (corresponds to above)
            servo_gpio_pin (int): GPIO pin for the servo
            width (float, optional): desired width of camera object. Defaults to None.
            height (float, optional): desired height of camera object. Defaults to None.
        """
        self._camera_id = camera_id
        self._cap = cv2.VideoCapture(camera_id)
        time.sleep(2) # let camera warm up
        if (width is not None) and (height is not None):
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        else:
            width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.set_dimensions(width, height)

        self._intrinsic_matrix, self._distortion_coefs = self._load_intrinsic_matrix_distortion_coefs(calibration_file_path)
        self._optimal_intrinsic_matrix, _ = self._get_optimal_camera_matrix_and_roi(self._intrinsic_matrix, self._distortion_coefs, width, height)

        logging.info(f"Camera intrinsic matrix: {self._optimal_intrinsic_matrix}")

        self._pivot_length = camera_pivot_length
        self._height_above_ground = camera_height
        self._distance_from_centre = 0.01 # 6mm from robot centre to camera centre
        self._angles_horizontal = camera_angles
        self._distance_thresholds = camera_thresholds
        self._ball_close_to_distance_threshold = ball_close_to_distance_threshold
        self._servo_angles = camera_servo_angles

        # TODO: tune these pulse widths?
        hardware_pwm_factory = PiGPIOFactory()
        self._servo = AngularServo(servo_gpio_pin, 
                                   min_pulse_width=0.0006, 
                                   max_pulse_width=0.0023, 
                                   pin_factory=hardware_pwm_factory)
        
        # set the initial angle to the flattest
        self.set_angle(self._angles_horizontal[0])

    def get_dimensions(self):
        """Returns the width and height of the camera"""
        return self.width, self.height

    def set_dimensions(self, width, height):
        self.width = width
        self.height = height
        self.midpoint = width // 2

    def get_midpoint_lims(self):
        """Returns the limits of the midpoint of the camera"""
        return self.midpoint - self.width // 10, self.midpoint + self.width // 10

    def clear_image_buffer(self, num_frames=5):
        for _ in range(num_frames):
            self._cap.grab()

    def grab(self):
        """Grab a frame from the camera"""
        return self._cap.grab()
    
    def read(self):
        """Read a frame from the camera"""
        return self._cap.read()
    
    def release(self):
        self._cap.release()

    def turn_off(self):
        self.release()

    def _load_intrinsic_matrix_distortion_coefs(self, json_file_path):
        with open(json_file_path, 'r') as file: # Read the JSON file
            json_data = json.load(file)

        camera_matrix = np.array(json_data['camera_matrix'])
        distortion_coefs = np.array(json_data['distortion_coefs'])

        logging.debug(f"Loaded camera matrix: {camera_matrix}")
        logging.debug(f"Loaded distortion coefficients: {distortion_coefs}")

        return camera_matrix, distortion_coefs
    
    def _get_optimal_camera_matrix_and_roi(self, camera_matrix, distortion_coefs, width, height):
        # get optimal camera matrix based on image dimensions
        # i.e. based on the free scaling parameter
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefs, (width, height), 1, (width,height))
        return new_camera_matrix, roi
    
    @property
    def intrinsic_matrix(self):
        return self._optimal_intrinsic_matrix
    
    def get_angle(self) -> float:
        """Get the angle of the camera to the horizontal

        Returns:
            float: angle in degrees from horizontal (positive downwards)
        """
        return self._angle_to_horizontal
    
    def set_angle(self, angle_to_horizontal) -> bool:
        """Set the angle of the camera to the horizontal

        Args:
            angle_to_horizontal (float): angle in degrees from horizontal (positive downwards)

        Returns:
            bool: True if successful, False otherwise
        """
        if angle_to_horizontal not in self._angles_horizontal:
            logging.error(f"Invalid angle: {angle_to_horizontal}. Valid angles: {self._angles_horizontal}")
            return False
        
        # set servo angle
        index = self._angles_horizontal.index(angle_to_horizontal)
        self._servo.angle = self._servo_angles[index]

        # set internal angle and distance threshold to horizontal object for transforms
        self._angle_to_horizontal = angle_to_horizontal
        logging.debug(f"Set camera angle to {angle_to_horizontal}, which is servo angle: {self._servo.angle}")
        self._distance_threshold = self._distance_thresholds[index]
        logging.debug(f"Sleeping for a small period to let camera stabilise")
        time.sleep(0.1)
        return True
    
    def set_angle_based_on_ball_distance(self, distance):
        for i, threshold in enumerate(self._distance_thresholds):
            if distance > threshold:
                return self.set_angle(self._angles_horizontal[i])
            
    def get_max_rotation_down_index(self, clearing_path=False):
        """Get the index of the maximum rotation down"""
        return len(self._angles_horizontal) - 1 if clearing_path else len(self._angles_horizontal) - 3
    
    def rotate_down(self, clearing_path=False):
        """Rotate the camera down to the next angle"""
        index = self._angles_horizontal.index(self._angle_to_horizontal)
        max_index = self.get_max_rotation_down_index(clearing_path)
        if index >= max_index:
            logging.warning("Camera is already at lowest angle")
            return False
        
        next_angle = self._angles_horizontal[index + 1]
        return self.set_angle(next_angle)
    
    def rotate_up(self):
        """Rotate the camera up to the next angle"""
        index = self._angles_horizontal.index(self._angle_to_horizontal)
        if index == 0:
            logging.warning("Camera is already at highest angle")
            return False
        
        next_angle = self._angles_horizontal[index - 1]
        return self.set_angle(next_angle)
    
    def get_out_of_way(self):
        """Move the camera out of the way"""
        result = self._servo.max()
        logging.info("SLEEPING FOR 0.5S WHEN MOVING CAMERA OUT OF THE WAY")
        time.sleep(0.5)
        return result
    
    def reset_to_highest_angle(self):
        """Reset the camera to the highest angle"""
        logging.info("SLEEPING FOR 0.3S WHEN RESETING ANGLE TO HIGHEST")
        result = self.set_angle(self._angles_horizontal[0])
        time.sleep(0.3)
        return result
    
    def get_maximum_distance_threshold(self):
        """Get the maximum threshold for the camera where ball goes out of bottom of frame"""
        return self._distance_thresholds[0]
    
    def get_current_distance_threshold(self):
        """Get the current threshold for the camera where ball goes out of bottom of frame"""
        return self._distance_threshold
    
    def get_ball_close_for_rotation_threshold(self):
        """Get the threshold for the camera where ball is close to the distance threshold for rotating slower"""
        return self._distance_thresholds[-3]
    
    def get_ball_close_to_approach_threshold(self):
        """Get the threshold for the camera where ball is close to the distance threshold"""
        index = self.get_max_rotation_down_index(clearing_path=False)
        return self._distance_thresholds[index] + self._ball_close_to_distance_threshold
    
    def ball_is_close_to_distance_threshold(self, distance):
        return distance < self._distance_threshold + self._ball_close_to_distance_threshold
    
    def get_camera_frame_to_robot(self):
        """Translation and rotation (only around y) from camera to robot

        Returns:
            tuple(height, angle): height above ground, angle relative to horizontal (positive downwards)
        """
        return self._height_above_ground, self._angle_to_horizontal
    
    def camera_coordinates_to_robot(self, camera_coords):
        """Convert camera coordinates to robot coordinates"""
        x, y = camera_coords
        height, angle = self.get_camera_frame_to_robot()
        x_robot = (x + self._pivot_length) * np.cos(np.deg2rad(angle))
        y_robot = y + self._distance_from_centre
        return x_robot, y_robot


if __name__ == "__main__":
    # test camera servo angles
    # Load configuration from YAML file
    from config import Config
    import sys
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())

    camera_index = config.camera_index
    camera_calib_path = config.camera_calibration_file_path
    camera_height = config.camera_height
    camera_pivot_length = config.camera_pivot_length
    camera_angles = config.camera_angles_horizontal
    camera_thresholds = config.camera_thresholds
    ball_close_to_threshold = config.camera_ball_close_to_threshold
    camera_servo_angles = config.camera_servo_angles
    camera_servo_gpio_pin = config.camera_servo_gpio_pin
    camera = Camera(
            camera_index, camera_calib_path,
            camera_height, camera_pivot_length, camera_angles, 
            camera_thresholds, ball_close_to_threshold,
            camera_servo_angles, camera_servo_gpio_pin)
    
    print(f"Camera angle: {camera.get_angle()}")

    for angle in camera_angles:
        print(f"Setting angle to {angle}")
        camera.set_angle(angle)
        print(f"Camera angle: {camera.get_angle()}")
        time.sleep(2)
    

    

