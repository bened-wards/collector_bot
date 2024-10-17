import os
import time
import warnings
import logging
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

# from ultralytics import YOLO

from yolo_onnx import YOLO
from camera import Camera
from tennis_types import BallRelativeToRobot, BoxRelativeToRobot
import utils

from config import Config

# Suppress specific warnings
warnings.filterwarnings("ignore")

class Detector:
    def __init__(self, camera: Camera, 
                 model_path,  
                 confidence_thresh, iou_thresh,
                 object_dimensions, distance_threshold,
                 output_directory="pics"):
        """Detect objects using a YOLO model. Gives relative pose of detected objects.

        Args:
            camera (Camera): camera object to capture images
            model_path (str): string file path to the model
            confidence_thresh (float): confidence threshold for yolo model
            iou_thresh (float): confidence threshold for yolo model
            object_dimensions (list): dimensions of object to detect (width, depth, height)
            distance_threshold (float): threshold beyond which object shouldn't be considered
            output_directory (str, optional): output directory to save debug pictures. Defaults to "pics".
        """
        # Initialize the model
        self.model_path = model_path
        self.model = YOLO(model_path, confidence_thresh, iou_thresh)

        self.output_directory = output_directory
        os.makedirs(self.output_directory, exist_ok=True)

        self.camera = camera

        # tennis ball: [0.067, 0.067, 0.067]
        # competition tennis ball: [0.062, 0.062, 0.062]
        # TODO: measure with calipers
        self.object_dimensions = object_dimensions
        self.distance_threshold = distance_threshold

        self.img_width, self.img_height = self.camera.get_dimensions()

    def capture_image(self):
        logging.debug("CLEARING IMAGE BUFFER")
        self.camera.clear_image_buffer(num_frames=5)

        logging.info("TAKING IMAGE")
        ret, image = self.camera.read()
        if not ret:
            logging.error("Failed to capture image")
            return None
        return image

    def capture_and_process_image(self, save_image=False):
        start_time = time.time()
        image = self.capture_image()
        if image is None:
            return [], [], []
        
        logging.debug("PROCESSING IMAGE")
        # Perform inference to get the detection results (filteres by confidence threshold)
        boxes_xywh, confidences, class_ids = self.model.detect(image)

        if save_image:
            annotated_image = image.copy()
            logging.debug("DRAWING BOXES ON IMAGE")
            annotated_image = self.model.draw_boxes(annotated_image)
            logging.debug("SHOWING IMAGE")
            cv2.imshow('Annotated Image', annotated_image)
            cv2.waitKey(1)
            # pass
            
            # # Save the annotated image
            # timestamp = time.strftime("%Y%m%d_%H%M%S")
            # image_filename = os.path.join(self.output_directory, f'annotated_image_{timestamp}.jpg')
            # logging.info(f"Saving image with boxes to filename: f{image_filename}")
            # plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for matplotlib
            # plt.axis('off')
            # plt.savefig(image_filename, format='jpg', bbox_inches='tight', pad_inches=0)
            # plt.close()

        logging.info(f"Time taken to process image: {time.time() - start_time:.2f} seconds")
        return boxes_xywh, confidences, class_ids

    def get_relative_positions(self, save_image=True) -> list[BallRelativeToRobot]:
        boxes_xywh, confidences, _ = self.capture_and_process_image(save_image=save_image)

        balls_relative_to_robot = []
        for box_xywh, confidence in zip(boxes_xywh, confidences):
            x, y, w, h = box_xywh
            if w > self.img_width * 0.985 and h > self.img_height * 0.985:
                logging.warning(f"Skipping ball with dims: x{x},y{y},w{w},h{h} because bounding box full image")
                continue
            relative_position = self.estimate_relative_pose(box_xywh, self.distance_threshold)
            # don't add if the ball is beyond distance threshold or too far to side of frame to make a good estimate
            if relative_position is None:
                continue
            ball = BallRelativeToRobot(relative_position[0], relative_position[1])
            print(ball, confidence)
            balls_relative_to_robot.append(ball)

        # sort balls based on relative distance to the robot
        logging.debug(f"Detection Data Before Sort: {balls_relative_to_robot}")
        balls_relative_to_robot = sorted(balls_relative_to_robot, key=lambda item: item.distance)
        logging.info(f"Detection Data After Sort: {balls_relative_to_robot}")

        return balls_relative_to_robot
    
    def get_box_centre_coords(self, save_image=True) -> list[BoxRelativeToRobot]:
        boxes_xywh, confidences, _ = self.capture_and_process_image(save_image=save_image)

        boxes_relative_to_robot = []
        for box_xywh, confidence in zip(boxes_xywh, confidences):
            x, y, w, h = box_xywh
            if h > self.img_height * 0.985:
                logging.warning(f"Skipping box with dims: x{x},y{y},w{w},h{h} because bounding box full image")
                continue
            box = BoxRelativeToRobot(x, y, w, h)
            print(box, confidence)
            boxes_relative_to_robot.append(box)

        if len(boxes_relative_to_robot) > 1:
            logging.error("More than one box detected! Sorting by area to get largest box, but this is an issue.")

        logging.debug(f"Boxes Data Before Sort: {boxes_relative_to_robot}")
        boxes_relative_to_robot = sorted(boxes_relative_to_robot, key=lambda item: item.area, reverse=True)
        logging.info(f"Boxes Data After Sort: {boxes_relative_to_robot}")

        return boxes_relative_to_robot
    
    def estimate_relative_pose(self, bounding_box, distance_threshold=5.0):
        """
        function:
            estimate the pose of a target based on size and location of its bounding box and the corresponding robot pose
        input:
            image_width: size of image captured by the camera e.g. [640, 480] -> [width, height]
            camera_matrix: list, the intrinsic matrix computed from camera calibration (read from 'param/intrinsic.txt')
                |f_x, s,   c_x|
                |0,   f_y, c_y|
                |0,   0,   1  |
                (f_x, f_y): focal length in pixels
                (c_x, c_y): optical centre in pixels
                s: skew coefficient (should be 0 for PenguinPi)
            bounding_box: list, an individual bounding box in an image (x,y,width,height)
            tennis_ball_dims: list, the dimensions of the target object [width, depth, height] (metres)
            distance_threshold: float, max distance to be considered for relative pose estimation (metres)
        output:
            relative_pose: [x, y] coordinates of the target in the world frame
        """
        image_width, image_height = self.camera.get_dimensions()

        # read in camera matrix (from camera calibration results)
        camera_matrix = self.camera.intrinsic_matrix
        focal_length_x = camera_matrix[0][0]
        focal_length_y = camera_matrix[1][1]

        # estimate target pose using bounding box and robot pose
        true_width = self.object_dimensions[0]
        true_height = self.object_dimensions[2]

        pixel_x = bounding_box[0]
        pixel_y = bounding_box[1]
        pixel_width = bounding_box[2]
        pixel_height = bounding_box[3]

        logging.debug(f"Bounding Box: {bounding_box}")
        logging.debug(f"True Width: {true_width}, True Height: {true_height}")
        logging.debug(f"Pixel Width: {pixel_width}, Pixel Height: {pixel_height}")
        logging.debug(f"Focal Length X: {focal_length_x}, Focal Length Y: {focal_length_y}")

        # estimated distance between the ball and the centre of the image plane based on height
        distance_h = float(true_height / pixel_height * focal_length_y)
        # estimated distance between the ball and the centre of the image plane based on width
        distance_w = float(true_width / pixel_width * focal_length_x)

        logging.debug(f"Height dist: {distance_h}")
        logging.debug(f"Width dist: {distance_w}")
        
        distances_close = utils.check_close(distance_h, distance_w, 10)
        if distances_close:
            logging.debug("Distances match within +/- 10%")
            distance = (distance_h + distance_w) / 2.0
        else:
            # TODO: do we need this? I've basically made it void
            # we probably are happy with average distance since we will 
            # approach precisely using laser sensor
            logging.warning("WARNING: distances don't match between height and width calculations.")
            if 1.01 < pixel_width / pixel_height < 5:
                logging.debug(f"Using width distance as the ratio of width to height is {pixel_width / pixel_height}. Ball likely out of bottom of frame")
                distance = distance_w
            elif 1.01 < pixel_height / pixel_width < 5:
                logging.debug(f"Using height distance as the ratio of height to width is {pixel_height / pixel_width}. Ball likely out of side of frame")
                distance = distance_h
            else:
                logging.debug("Ball not in a great position to estimate relative pose, returning None")
                if pixel_y > image_height * 0.85:
                    logging.debug("Ball is near the bottom of the frame. SWITCH TO LASER SENSOR")
                    return None
                elif  pixel_x < image_width * 0.1 or pixel_x > image_width * 0.9:
                    logging.debug("Ball is near the edge of the frame. No trusting measurement, but know which direction to turn")
                    return None
                else:
                    return None
                
        logging.debug(f"Distance to ball: {distance}")

        # TODO ignore balls detected at far distance?
        if distance > distance_threshold:
            logging.debug("WARNING: faraway detection of ball. May want to ignore.")
            return None

        # NOTE: this is from excel sheet on Ben's laptop if you need to fix it
        distance = distance  # empirical correction factor
        
        x_shift = image_width/2 - pixel_x        # x distance between bounding box centre and centreline in camera view
        theta = np.arctan(x_shift/focal_length_x)     # angle of object relative to the robot

        # relative object location
        distance_obj = distance/np.cos(theta) # relative distance between robot and object
        x_relative = distance_obj * np.cos(theta) # relative x pose
        y_relative = distance_obj * np.sin(theta) # relative y pose
        relative_pose_camera = [x_relative, y_relative]
        logging.debug(f'relative pose to camera: {relative_pose_camera}')

        relative_pose_robot = self.camera.camera_coordinates_to_robot(relative_pose_camera)
        logging.debug(f'relative pose to robot before ball shift: {relative_pose_robot}')


        # TODO: this is wrong but seems to work ok
        #       I think it is something more like the distance between
        #       the centre of the ball and the edge of the ball and its 
        #       relation to the centre of the image. this would then give
        #       some real world distance (in x) that need to offset to the front
        #       of the tennis ball in robot coords. 

        # perform shift due to tennis ball centre in image
        # theta is the angle in addition to the tilt caused by camera
        phi = np.arctan2(image_height/2 - pixel_y, focal_length_y)
        ball_centre_theta_horizontal = np.deg2rad(self.camera.get_angle()) - theta
        true_x_robot = relative_pose_robot[0] - true_height/2 * np.sin(ball_centre_theta_horizontal)

        relative_pose_robot = (true_x_robot, relative_pose_robot[1])
        logging.debug(f"relative pose to robot after ball shift: {relative_pose_robot}")

        def quadratic_function(x, a, b, c):
            return a * x ** 2 + b * x + c

        # scale = 1
        if true_x_robot < 1.2:
            scaled_true_x_robot = quadratic_function(true_x_robot, 0.0833, 1, 0)
            scaled_true_y_robot = quadratic_function(relative_pose_robot[1], 0.0833, 1, 0)
        # elif true_x_robot < 1.8:
        #     scaled_true_x_robot = true_x_robot * 1.15
        #     scaled_true_y_robot = relative_pose_robot[1] * 1.15
        else:
            scaled_true_x_robot = quadratic_function(true_x_robot, 0.04, 1.11689, -0.077874)
            scaled_true_y_robot = quadratic_function(relative_pose_robot[1], 0.04, 1.11689, -0.077874)

        # scaled_true_x_robot = true_x_robot * scale
        # scaled_true_y_robot = relative_pose_robot[1] * scale

        relative_pose_robot = (scaled_true_x_robot, scaled_true_y_robot)
        logging.debug(f"relative pose to robot after scaling: {relative_pose_robot}")
                                                                  
        return relative_pose_robot

def ball_detection_loop(camera: Camera, detector: Detector, save_image=True):
    while True:
        ret, img = camera.read()
        if not ret:
            break
        
        start_time = time.time()
        balls_relative_to_robot = detector.get_relative_positions(save_image=save_image)
        logging.info(str(balls_relative_to_robot))
        end_time = time.time()
        logging.info(f"Time taken: {end_time - start_time:.2f}s")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    MODE = "BALL"

    # Load configuration from YAML file
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())

    if MODE == "NO PI":
        class CameraNoPi:

            def __init__(self):
                self._camera = cv2.VideoCapture(0)
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            def clear_image_buffer(self, num_frames=5):
                for _ in range(num_frames):
                    self._camera.grab()

            def read(self):
                return self._camera.read()
            
            def get_dimensions(self):
                return (640, 480)

        camera = CameraNoPi()
    else:
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
        
        camera.reset_to_highest_angle()

    if MODE == "BASIC_DETECT":        
        detector = Detector(
            camera, config.ball_model_path, 
            confidence_thresh=config.ball_confidence_thresh, iou_thresh=config.ball_iou_thresh, 
            object_dimensions=config.ball_dimensions, 
            distance_threshold=config.ball_distance_thresh)
        # detector = Detector(camera, model_path="models/box_yolov8.pt")

        width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logging.info(f"h={height},w={width}")

        sum_time = 0
        num_frames = 0
        while True:
            
            # Capture an image, process it, and retrieve detection data
            start_time = time.time()
            boxes_xywh, _, _ = detector.capture_and_process_image(save_image=False)
            if len(boxes_xywh) > 0:
                logging.info(str(boxes_xywh))
            end_time = time.time()
            sum_time += end_time - start_time
            num_frames += 1
            logging.info(f"Time taken to process image: {end_time - start_time:.2f} seconds")
            logging.info(f"Average time: {sum_time / num_frames:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

    elif MODE == "BALL":       
        detector = Detector(
            camera, config.ball_model_path, 
            confidence_thresh=config.ball_confidence_thresh, iou_thresh=config.ball_iou_thresh, 
            object_dimensions=config.ball_dimensions, 
            distance_threshold=config.ball_distance_thresh)

        ball_detection_loop(camera, detector)

    elif MODE == "BOX":       
        detector = Detector(
            camera, config.box_model_path, 
            confidence_thresh=config.box_confidence_thresh, iou_thresh=config.box_iou_thresh, 
            object_dimensions=[0.0, 0.0, 0.0], 
            distance_threshold=0.0)
        
        while True:
            start_time = time.time()
            boxes_relative_to_robot = detector.get_box_centre_coords(save_image=True)

            end_time = time.time()
            logging.info(f"Time taken: {end_time - start_time:.2f}s")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    elif MODE == "NO PI":
        # detector = Detector(
        #     camera, config.ball_model_path, 
        #     confidence_thresh=config.ball_confidence_thresh, iou_thresh=config.ball_iou_thresh, 
        #     object_dimensions=config.ball_dimensions, 
        #     distance_threshold=config.ball_distance_thresh)
        detector = Detector(
            camera, config.box_model_path, 
            confidence_thresh=config.box_confidence_thresh, iou_thresh=config.box_iou_thresh, 
            object_dimensions=[0.0, 0.0, 0.0], 
            distance_threshold=0.0
        )
        # detector = Detector(camera, model_path="models/box_yolov8.pt")

        sum_time = 0
        num_frames = 0
        while True:
            
            # Capture an image, process it, and retrieve detection data
            start_time = time.time()
            boxes_xywh, _, _ = detector.capture_and_process_image(save_image=True)
            if len(boxes_xywh) > 0:
                logging.info(str(boxes_xywh))
            end_time = time.time()
            sum_time += end_time - start_time
            num_frames += 1
            logging.info(f"Time taken to process image: {end_time - start_time:.2f} seconds")
            logging.info(f"Average time: {sum_time / num_frames:.2f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()



