import math
import time
import multiprocessing
import logging
import sys
from typing import Optional

import numpy as np

from config import Config
from motion_controller import MotionController
from detect import Detector
from camera import Camera
from laser_sensor import LaserSensor
from tennis_map import TennisMap
from tennis_types import BallRelativeToRobot, BoxRelativeToRobot
from scooper import Scooper
from reardoor import RearDoor

class ScoopAndDump:

    def __init__(self, config: Config):
        self.config = config

        self._camera: Camera = Camera(
            self.config.camera_index, self.config.camera_calibration_file_path, 
            self.config.camera_height, self.config.camera_pivot_length,
            self.config.camera_angles_horizontal, 
            self.config.camera_thresholds, self.config.camera_ball_close_to_threshold,
            self.config.camera_servo_angles, self.config.camera_servo_gpio_pin)

        pics_dir = "pics"
        # YOLO tennis ball detector
        self._ball_detector = Detector(
            self._camera, self.config.ball_model_path, 
            self.config.ball_confidence_thresh, self.config.ball_iou_thresh,
            self.config.ball_dimensions, self.config.ball_distance_thresh,
            output_directory=pics_dir)
        # YOLO box detector
        self._box_detector = Detector(
            self._camera, self.config.box_model_path, 
            self.config.box_confidence_thresh, self.config.box_iou_thresh,
            [0.0, 0.0, 0.0], 0.0,
            output_directory=pics_dir)

        self._motion_controller = MotionController()

        self._laser_sensor = LaserSensor(
            config.laser_sensor_ball_threshold, 
            config.laser_sensor_box_near_threshold, 
            config.laser_sensor_box_far_threshold,
            config.laser_sensor_timing_budget)

        self._scooper = Scooper(self.config.claw_servo_gpio_pin,
                                self.config.arm_servo_gpio_pin,
                                self.config.claw_open_angle,
                                self.config.claw_close_angle,
                                self.config.arm_up_angle,
                                self.config.arm_down_angle)
        
        self._rear_door = RearDoor(self.config.rear_door_servo_gpio_pin,
                                   self.config.rear_door_open_angle,
                                   self.config.rear_door_close_angle)

        self._map = TennisMap(
            self.config.starting_state, self.config.arena, self.config.arena_dimensions, 
            self.config.box_dimensions, self.config.view_box_offsets,
            self.config.ball_merge_threshold, self.config.map_distance_travelled_threshold,
            self.config.map_distance_balls_close_no_remove)

        self._balls_deposited = 0
        self._balls_in_storage = 0

        self._last_ball_distance = 100000  # ball infinitely far away

        self._done_360 = False # whether we have done a full 360 rotation in search

        self._timer = time.time() # use timer to return to box 
        self._time_limit = self.config.return_to_box_time_limit # seconds
        self._full_time_limit = self.config.full_time_limit # seconds

    def __del__(self):
        self._motion_controller.really_stop()
        time.sleep(1)
        self._motion_controller.set_stop_event()
        self._reader_proc.join()
        self._writer_proc.join()

    def start_motion_controller_processes(self):
        self._reader_proc = multiprocessing.Process(target=self._motion_controller.read_from_serial, args=())
        self._writer_proc = multiprocessing.Process(target=self._motion_controller.write_to_serial, args=())

        self._reader_proc.start()
        self._writer_proc.start()

    #TODO: implement LED writer
    def write_LED(self, searching=False, collecting=False, depositing=False):
        pass

    # ROBOT STATE METHODS - also updates the map
    def get_robot_state(self):
        state = self._motion_controller.get_state()
        self._map.update_robot_state(state)
        return state
    
    def get_robot_theta(self):
        state = self.get_robot_state()
        return state[2]
    
    def reset_angle_at_box_waypoint(self):
        time.sleep(0.2) # allow state to be propagated and read back from serial
        arena_middle = self._map._middle_box
        x, y, th = self.get_robot_state()
        dx = arena_middle[0] - x
        dy = arena_middle[1] - y
        middle_th = np.arctan2(dy, dx)
        new_state = (x, y, middle_th)
        logging.info("RESETTING ANGLE")
        logging.info(f"Resetting robot state to {new_state}")
        self._map.correct_robot_state(new_state)
        self._motion_controller.reset_state_at_box(new_state[0], new_state[1], new_state[2])
        time.sleep(0.2) # allow state to be propagated and read back from serial
        logging.info("Robot state reset")
    
    def reset_robot_state(self):
        time.sleep(0.2) # allow state to be propagated and read back from serial
        theta = self.get_robot_theta()
        box_waypoint = self._map.box_waypoint
        x = box_waypoint[0] - 0.24
        y = box_waypoint[1]
        new_state = (x, y, theta)
        logging.info("RESETTING XY")
        logging.info(f"Resetting robot state to {new_state}")
        self._map.correct_robot_state(new_state)
        self._motion_controller.reset_state_at_box(new_state[0], new_state[1], new_state[2])
        time.sleep(0.2) # allow state to be propagated and read back from serial
        logging.info("Robot state reset")

    # CAMERA METHODS
    def take_image_and_get_target_ball(self, with_map=True) -> Optional[BallRelativeToRobot]:
        """Takes image and returns the target ball. Also updates the last ball distance"""
        logging.debug("Small sleep to stabilise camera")
        time.sleep(0.2)
        balls_relative_to_robot = self._ball_detector.get_relative_positions()
        self.get_robot_state() # get most recent robot state before map update
        if with_map:
            filtered_balls_relative_to_robot = self._map.add_balls(balls_relative_to_robot)
        else:
            filtered_balls_relative_to_robot = balls_relative_to_robot

        if filtered_balls_relative_to_robot:
            target_ball = filtered_balls_relative_to_robot[0]
            self._last_ball_distance = target_ball.distance
            return target_ball
        return None
    
    def take_image_and_get_box(self) -> Optional[BoxRelativeToRobot]:
        # TODO: change this to triangulation of box distance using rotation of the camera?
        logging.debug("Small sleep to stabilise camera")
        time.sleep(0.1)
        centre_coords = self._box_detector.get_box_centre_coords()
        self.get_robot_state() # get most recent robot state before map update
        self._map.visualise_map()
        if centre_coords is not None and len(centre_coords) == 1:
            return centre_coords[0]
        
        logging.warning(f"Took image and found {len(centre_coords) if centre_coords is not None else 0} boxes. Returning None")
        return None
    
    
    #######################
    # SEARCHING FOR BALL METHODS
    #######################
    @property
    def max_search_angle(self):
        return 2 * math.pi

    @property
    def fail_search_angle(self):
        return 15 * math.pi / 180
    
    @property
    def box_fail_search_angle(self):
        return 30 * math.pi / 180
    
    def do_initial_search(self) -> Optional[BallRelativeToRobot]:
        """Do initial search and return target ball if found"""
        logging.info("DOING INITIAL SEARCH")
        self.write_LED(searching=True)
        waypoint = self._map.get_next_safe_waypoint()
        angle_to_first_waypoint = abs(np.arctan2(waypoint[1], waypoint[0]))
        target_ball = self.search_for_ball_for_angle(angle_to_first_waypoint)
        if target_ball is not None:
            return target_ball
        
        # no ball found, traverse to safe waypoint and hopefully find a ball
        while waypoint[0] < min(self.config.arena_dimensions) // 2:
            self.traverse_to_waypoint(waypoint, self.config.FORWARD_SPEED_TRAVERSE)
            print(f"STOPPING AT WAYPOINT AND TAKING PHOTO: {waypoint}")
            target_ball = self.take_image_and_get_target_ball()
            if target_ball is not None:
                return target_ball
            waypoint = self._map.get_next_safe_waypoint()
        return None
    
    def do_initial_sweep(self, angle=np.radians(75)):
        direction = self.config.search_direction
        start_theta = self.get_robot_theta()
        logging.info(f"SEARCHING FOR {angle * 180 / np.pi:.2f} deg: Starting theta: {start_theta}")
        while abs(self.get_robot_theta() - start_theta) < abs(angle):
            logging.info(f"Diff in angle from start: {abs(self.get_robot_theta() - start_theta)}")
            logging.info("STOPPING AND TAKING IMAGE IN INITIAL SWEEP")
            self._motion_controller.stop()
            target_ball = self.take_image_and_get_target_ball()
            if direction == "CW":
                self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_SEARCH)
            else:
                self._motion_controller.rotate_ccw(self.config.ROTATION_SPEED_SEARCH)

            time.sleep(self.config.SEARCH_SLEEP_TIME)
    
    def search_for_ball(self) -> Optional[BallRelativeToRobot]:
        self._done_360 = False
        x, y, th = self.get_robot_state()
        ball_coords = self._map.get_closest_ball_position_to_robot()
        if ball_coords is not None:
            logging.debug(f"THERE IS ANOTHER BALL ON THE MAP. WE WILL GO TO IT AT {ball_coords}")
            ball_x, ball_y = ball_coords
            dx = ball_x - x
            dy = ball_y - y
            goal_th = np.arctan2(dy, dx)
            th = np.arctan2(np.sin(th), np.cos(th))
            dtheta = goal_th - th
            dtheta = np.arctan2(np.sin(dtheta), np.cos(dtheta))
            relative_distance = np.sqrt(dx**2 + dy**2)
            if relative_distance > self.config.ball_memory_threshold:
                logging.info(f"FOUND BALL IN MEMORY, BUT TOO FAR AWAY: {relative_distance:.2f}m. Doing random search")
                return self.search_for_ball_for_angle(self.max_search_angle)
            
            # TODO: probably better off actually just rotating to where we think the ball
            # is and then doing a little sweep side to side and up/down with camera
            # which I need to do for robustness anyway
            # set the camera angle so we can see it when we rotate
            self._camera.set_angle_based_on_ball_distance(relative_distance)
            while not self.rotate_to_theta(goal_th, self.config.ROTATION_SPEED_SEARCH):
                self.get_robot_state() # update state for visualisation of map
                self._map.visualise_map()
                time.sleep(self.config.SLEEP_TIME)
            target_ball = self.rotate_ball_to_centre()
            # target_ball = self.search_for_ball_for_angle(dtheta, direction="CW" if dtheta < 0 else "CCW")
            if target_ball is not None:
                return target_ball
            target_ball = self.find_ball_backup(self.fail_search_angle)
            if target_ball is not None:
                distance = np.sqrt(target_ball.x - dx)**2 + (target_ball.y - dy)**2
                logging.info(f"FOUND BALL IN BACKUP SEARCH, DISTANCE: {distance:.2f}m")
                # check if ball found was close to what we thought it was
                if distance < self.config.ball_merge_threshold:
                    logging.info("FOUND THE EXPECTED BALL")
                    return target_ball
                logging.info("FOUND BALL BUT NOT THE EXPECTED BALL, REMOVING FROM MAP")
            self._map.remove_ball_by_position(ball_coords)
            return target_ball
        
        self._camera.reset_to_highest_angle()
        # get direction based on the angle to the furthest corner
        angle_to_furthest_corner = self._map.get_angle_to_furthest_corner()
        direction = "CW" if angle_to_furthest_corner < 0 else "CCW"
        self._done_360 = True
        return self.search_for_ball_for_angle(self.max_search_angle, direction)
    
    def search_for_ball_for_angle(self, angle, direction=None) -> Optional[BallRelativeToRobot]:
        """
        Parameters:
            angle: float - angle to search for tennis ball in radians
            direction: str - direction to search in (CW or CCW)
        Search for tennis ball and return if found
        Returns target_ball if found, None otherwise
        """
        if direction is None:
            direction = self.config.search_direction
        start_theta = self.get_robot_theta()
        logging.info(f"SEARCHING FOR {angle * 180 / np.pi:.2f} deg: Starting theta: {start_theta}")
        while abs(self.get_robot_theta() - start_theta) < abs(angle):
            logging.info(f"Diff in angle from start: {abs(self.get_robot_theta() - start_theta)}")
            logging.info("STOPPING AND TAKING IMAGE IN SEARCH")
            self._motion_controller.stop()
            target_ball = self.take_image_and_get_target_ball()
            if target_ball is not None:
                return target_ball
            
            logging.info(f"HAVEN'T FOUND BALL ROTATING FOR: {self.config.SEARCH_SLEEP_TIME:.2f}s")
            if direction == "CW":
                self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_SEARCH)
            else:
                self._motion_controller.rotate_ccw(self.config.ROTATION_SPEED_SEARCH)

            time.sleep(self.config.SEARCH_SLEEP_TIME)

        return None
    
    def get_rotation_speed(self, angle_deg, limit) -> float:
        """Get the rotation speed based on how far ball is from centre of image"""
        gain = 0.5
        angle_deg = abs(angle_deg)
        limit = abs(limit)
        diff = angle_deg - limit
        multiplier = 1 + gain * diff / angle_deg
        return self.config.ROTATION_SPEED_SEARCH * multiplier
    
    def do_rotate_ball_to_centre(self, ball: BallRelativeToRobot) -> bool:
        """Returns whether the ball is in the centre of the image"""
        lims = (-self.config.CENTRE_THRESHOLD_DEG, self.config.CENTRE_THRESHOLD_DEG)
        # if the ball is far away, accept a wider margin
        # if ball.distance > self._camera.get_maximum_distance_threshold():
        #     lims = (-self.config.CENTRE_THRESHOLD_DEG*1.5, self.config.CENTRE_THRESHOLD_DEG*1.5)

        angle_deg = np.degrees(np.arctan2(ball.y, ball.x))
        logging.info(f"Relative position of ball ({ball.x},{ball.y}). Angle: {angle_deg} deg. Lims: {lims}")
        if lims[0] <= angle_deg <= lims[1]:
            self._motion_controller.stop()
            logging.info("STOPPING, BALL IS IN CENTRE OF IMAGE")
            return True
        elif angle_deg < lims[0]:
            self._motion_controller.rotate_cw(self.get_rotation_speed(angle_deg, lims[0]))
            logging.info("ROTATING CW TO GET BALL IN CENTRE")
            return False
        elif angle_deg > lims[1]:
            self._motion_controller.rotate_ccw(self.get_rotation_speed(angle_deg, lims[1]))
            logging.info("ROTATING CCW TO GET BALL IN CENTRE")
            return False
        
    def get_rotate_ball_to_centre_time(self, target_ball: BallRelativeToRobot):
        angle_deg = abs(np.degrees(np.arctan2(target_ball.y, target_ball.x)))
        ball_multiplier = 1 if target_ball.distance > self._camera.get_ball_close_for_rotation_threshold() else 0.6
        multiplier = ball_multiplier * (1 + abs(angle_deg) / 45)
        multiplier = min(1.5, multiplier)
        # limit = abs(limit)
        # diff = angle_deg - limit
        # multiplier = 1 + gain * 
        return self.config.SLEEP_TIME * multiplier

    def rotate_ball_to_centre(self, known_target_ball: BallRelativeToRobot=None, with_map=True) -> Optional[BallRelativeToRobot]:
        """Returns target ball if moved to centre, None otherwise"""
        while True:
            self._motion_controller.stop()
            logging.info("STOPPING AND TAKING IMAGE: ROTATE TO CENTRE")
            # use the image from searching first to determine which way to rotate
            if known_target_ball is None:
                target_ball = self.take_image_and_get_target_ball(with_map)
            else:
                target_ball = known_target_ball
                known_target_ball = None

            if target_ball is None:
                self._motion_controller.stop()
                logging.info("TRYING TO ROTATE --- NO BALL FOUND GOING BACK TO MAIN ROTATION")
                return None
            
            ball_in_centre = self.do_rotate_ball_to_centre(target_ball)
            if ball_in_centre:
                return target_ball
            
            sleep_time = self.get_rotate_ball_to_centre_time(target_ball)
            time.sleep(sleep_time)

    def find_ball_backup(self, rotate_radians) -> Optional[BallRelativeToRobot]:
        """Failed to find an object, turn a few degrees each direction and move camera up and down"""
        logging.warning("FAILED TO FIND OBJECT, ROTATING AND MOVING CAMERA")
        target_ball = self.search_for_ball_for_angle(rotate_radians, direction="CW")
        if target_ball is not None:
            return target_ball
        target_ball = self.search_for_ball_for_angle(rotate_radians * 2, direction="CCW")
        if target_ball is not None:
            return target_ball
        
        # # rotate down and try again
        # self._camera.rotate_down()
        # target_ball = self.search_for_ball_for_angle(rotate_radians, direction="CW")
        # if target_ball is not None:
        #     return target_ball
        # target_ball = self.search_for_ball_for_angle(rotate_radians*2, direction="CCW")
        # if target_ball is not None:
        #     return target_ball
        # self._camera.rotate_up()
        # self._camera.rotate_up()
        # target_ball = self.search_for_ball_for_angle(rotate_radians, direction="CW")
        # if target_ball is not None:
        #     return target_ball
        # target_ball = self.search_for_ball_for_angle(rotate_radians * 2, direction="CCW")
        
        self._camera.reset_to_highest_angle()
        return None

    #######################
    # TRAVERSING TO BALL METHODS
    #######################
    # @property
    # def ball_drift_correction_time(self):
    #     """Drive forward blindly based on how far away ball is"""
    #     f_speed = self.config.FORWARD_SPEED_TRAVERSE
    #     if self._camera.ball_is_close_to_distance_threshold(self._last_ball_distance):
    #         return 0
    #     else:
    #         time_to_ball_threshold = (self._last_ball_distance - self._camera.get_current_distance_threshold()) / f_speed
    #         # TODO: probably don't want this - need something better for close balls
    #         if self._camera.get_current_distance_threshold() < 0.25:
    #             time_to_ball_threshold  *= 0.5
    #         return min(time_to_ball_threshold, self.config.MAX_DRIFT_CORRECTION_TIME) 
        
    def get_drift_correction_time_and_speed(self, target_ball: BallRelativeToRobot):
        """Drive forward blindly based on how far away ball is.
        Returns: drift_correction_time, forward_speed"""
        if target_ball is None:
            return None, None

        if target_ball.distance > self._camera.get_maximum_distance_threshold():
            f_speed = self.config.FORWARD_SPEED_TRAVERSE
        else:
            f_speed = self.config.FORWARD_SPEED_TRAVERSE * 0.5
        
        # increase speed if we are still far away from the first threshold
        # distance_multiplier = 1
        # if target_ball.distance > self._camera.get_maximum_distance_threshold():
        #     distance_to_max_threshold = target_ball.distance - self._camera.get_maximum_distance_threshold()
        #     distance_multiplier = 1 + distance_to_max_threshold / target_ball.distance
        # f_speed *= distance_multiplier

        if self._camera.ball_is_close_to_distance_threshold(self._last_ball_distance):
            return 0, 0
        else:
            time_to_ball_threshold = (self._last_ball_distance - self._camera.get_current_distance_threshold()) / f_speed
            # TODO: probably don't want this - need something better for close balls
            if self._camera.get_current_distance_threshold() < 0.25:
                time_to_ball_threshold  *= 0.5
            return min(time_to_ball_threshold, self.config.MAX_DRIFT_CORRECTION_TIME), min(f_speed, self.config.FORWARD_SPEED_TRAVERSE)
        
    # def traverse_to_ball(self) -> Optional[BallRelativeToRobot]:
    #     """
    #     Returns target_ball if it is close enough to pick up, None otherwise
    #     """
    #     logging.info("TRAVERSING WITH CAMERA, BALL WAS IN CENTRE")
    #     time_at_last_drift_correction = time.time()
    #     # approach ball until within first distance threshold for camera
    #     while True:
    #         logging.info(f"Trvaerse forward blindly with camera angle ({self._camera.get_angle()}): {(time.time() - time_at_last_drift_correction):.2f}s")
    #         self._motion_controller.drive_forward(self.config.FORWARD_SPEED_TRAVERSE)
    #         time.sleep(self.config.SLEEP_TIME)

    #         # correct for drift every DRIFT_CORRECTION_TIME seconds until ball is going out of frame
    #         if time.time() - time_at_last_drift_correction > self.ball_drift_correction_time:
    #             logging.info(f"DRIFT CORRECTION AFTER: {self.ball_drift_correction_time}s")
    #             target_ball = self.rotate_ball_to_centre()
    #             if target_ball is None:
    #                 return None
                
    #             logging.info(f"\n\n\nDISTANCE: {self._last_ball_distance}")
    #             if target_ball.distance <= self.config.CAMERA_DISTANCE_THRESHOLD:
    #                 return target_ball
                
    #             logging.info("TENNIS BALL STILL FAR AWAY, CONTINuING")
    #             time_at_last_drift_correction = time.time()


    def traverse_to_ball_with_rotating_camera(self, target_ball: BallRelativeToRobot, with_map=True) -> Optional[BallRelativeToRobot]:
        logging.info("TRAVERSING WITH CAMERA, BALL WAS IN CENTRE")
        time_at_last_drift_correction = time.time()
        rotated_down_from_overdrive = False
        drift_correction_time, forward_speed = self.get_drift_correction_time_and_speed(target_ball)
        distance_to_ball = target_ball.distance
        distance_driven = 0
        start_x, start_y, _ = self.get_robot_state()
        while True:
            logging.info(f"traversing forward blindly: {(time.time() - time_at_last_drift_correction):.2f}s")
            self._motion_controller.drive_forward(forward_speed)
            time.sleep(self.config.SLEEP_TIME)

            # Correct for drift every DRIFT_CORRECTION_TIME seconds
            if time.time() - time_at_last_drift_correction > drift_correction_time:
                logging.info(f"DRIFT CORRECTION AFTER: {drift_correction_time}s")
                end_x, end_y, _ = self.get_robot_state()
                distance_driven = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                target_ball = self.rotate_ball_to_centre(with_map=with_map)
                if target_ball is not None and target_ball.distance + distance_driven - distance_to_ball > self.config.ball_traverse_threshold:
                    self._camera.rotate_down()
                    rotated_down_from_overdrive = True
                    continue
                
                # if lost ball, could be because angle too high, try rotating down
                if target_ball is None and not rotated_down_from_overdrive:
                    self._camera.rotate_down()
                    rotated_down_from_overdrive = True
                    continue
                elif target_ball is None:
                    self._camera.reset_to_highest_angle()
                    return None
                rotated_down_from_overdrive = False
                
                logging.info(f"\n\n\nDISTANCE: {self._last_ball_distance}")
                if self._camera.ball_is_close_to_distance_threshold(target_ball.distance):
                    logging.info("Reached camera threshold, moving camera down")
                    if not self._camera.rotate_down():
                        logging.info("Reached max camera angle, returning to let laser sensor take over")
                        self._camera.reset_to_highest_angle()
                        target_ball = self.rotate_ball_to_centre(target_ball, with_map=with_map)
                        return target_ball
                
                logging.info("TENNIS BALL STILL FAR AWAY, CONTINuING")
                drift_correction_time, forward_speed = self.get_drift_correction_time_and_speed(target_ball)
                time_at_last_drift_correction = time.time()

    #######################
    # APPROACHING BALL AND PICKING UP METHODS
    #######################

    def approach_ball_with_laser_sensor(self, target_ball: BallRelativeToRobot) -> bool:
        logging.info("APPROACHING TENNIS BALL SLOWLY WITH LASER SENSOR")
        max_distance = self._camera.get_ball_close_to_approach_threshold()
        start_x, start_y, _ = self.get_robot_state()
        while not self._laser_sensor.ball_in_range():
            end_x, end_y, _ = self.get_robot_state()
            distance_driven = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            logging.info(f"Distance driven when approaching ball with laser sensor: {distance_driven:.2f}m")
            if distance_driven > max_distance:
                logging.info("LASER SENSOR DIDN'T ACTIVATE IN EXPECTED DISTANCE, PROBABLY SUNNY")
                break
            self._motion_controller.drive_forward(self.config.FORWARD_SPEED_APPROACH)
            time.sleep(self.config.SHORT_SLEEP_TIME)
        self._motion_controller.stop()
        logging.info("BALL IN RANGE, STOPPING")
        return True

    def traverse_to_ball_and_pick_up(self, target_ball: BallRelativeToRobot) -> bool:
        target_ball = self.traverse_to_ball_with_rotating_camera(target_ball)
        if target_ball is None:
            return False
        
        ball_close_enough_to_pickup = self.approach_ball_with_laser_sensor(target_ball)

        if ball_close_enough_to_pickup:
            # remove the ball from the map, assuming we picked it up
            robot_x, robot_y, _ = self.get_robot_state()
            self._map.remove_ball_by_position((robot_x, robot_y))
            self._map.visualise_map()

        return ball_close_enough_to_pickup
    
    def pickup_ball(self):
        """Pick up the tennis ball when it is close enough to do so."""
        self._camera.get_out_of_way()
        self._scooper.pick_up_ball()
        self._camera.reset_to_highest_angle()

    def remove_collected_ball_from_map(self):
        # remove the ball from the map, assuming we picked it up
        robot_x, robot_y, _ = self.get_robot_state()
        self._map.remove_ball_by_position((robot_x, robot_y))
        self._map.visualise_map()

    #######################
    # SEARCHING FOR BOX METHODS
    #######################
    @property
    def box_search_angle(self):
        return 25 * math.pi / 180

    def really_find_box(self) -> BoxRelativeToRobot: 
        #TODO: this should stop looping after a while
        #      should switch to line following/tracking as a backup
        """REALLY REALLY REALLY FIND THE BOX, WE HAVE NO OTHER OPTION
        THIS COULD INFINITE LOOP

        Returns:
            BoxRelativeToRobot: box if we found it
        """
        logging.info("REALLY FINDING BOX. THIS COULD INFINITE LOOP")
        # take picture to get updated relative box position
        box_relative_to_robot = self.take_image_and_get_box()
        if box_relative_to_robot is None:
            box_relative_to_robot = self.find_box_backup(self.box_fail_search_angle)
            self._camera.reset_to_highest_angle()
            if box_relative_to_robot is None:
                logging.error("BOX NOT FOUND AFTER BACKUP SEARCH")
                return None
            
        # rotate box to centre
        # TODO: switch this to find a line so that we can approach at right angles?
        box_relative_to_robot = self.rotate_box_to_centre(box_relative_to_robot)
        if box_relative_to_robot is None:
            logging.error("BOX LOST WHEN ROTATING TRY AGAIN")
            return self.really_find_box()
        
        return box_relative_to_robot

    def search_for_box_for_angle(self, angle, direction=None) -> Optional[BoxRelativeToRobot]:
        """
        Parameters:
            angle: float - angle to search for box in radians
            direction: str - direction to search in (CW or CCW)
        Search for box and return if found
        """
        if direction is None:
            direction = self.config.search_direction
        start_theta = self.get_robot_theta()
        logging.info(f"SEARCHING FOR {angle * 180 / np.pi:.2f} deg: Starting theta: {start_theta}")
        while abs(self.get_robot_theta() - start_theta) < abs(angle):
            logging.info(f"Diff in angle from start: {abs(self.get_robot_theta() - start_theta)}")
            logging.info("STOPPING AND TAKING IMAGE IN BOX SEARCH")
            self._motion_controller.stop()
            box_relative_to_robot = self.take_image_and_get_box()
            if box_relative_to_robot is not None:
                return box_relative_to_robot
            
            logging.info(f"HAVEN'T FOUND BOX ROTATING FOR: {self.config.BOX_APPROACH_SLEEP_TIME:.2f}s")
            if direction == "CW":
                self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_SEARCH)
            else:
                self._motion_controller.rotate_ccw(self.config.ROTATION_SPEED_SEARCH)

            time.sleep(self.config.BOX_APPROACH_SLEEP_TIME)

        return None
    
    def find_box_backup(self, rotate_radians) -> Optional[BoxRelativeToRobot]:
        """Failed to find the box, turn a few degrees each direction and move camera up and down"""
        logging.warning("FAILED TO FIND BOX, ROTATING AND MOVING CAMERA")
        box_relative_to_robot = self.search_for_box_for_angle(rotate_radians, direction="CW")
        if box_relative_to_robot is not None:
            return box_relative_to_robot
        box_relative_to_robot = self.search_for_box_for_angle(rotate_radians * 2, direction="CCW")
        if box_relative_to_robot is not None:
            return box_relative_to_robot
        
        # rotate down and try again
        self._camera.rotate_down()
        box_relative_to_robot = self.search_for_box_for_angle(rotate_radians * 2, direction="CW")
        if box_relative_to_robot is not None:
            return box_relative_to_robot
        box_relative_to_robot = self.search_for_box_for_angle(rotate_radians*2, direction="CCW")
        if box_relative_to_robot is not None:
            return box_relative_to_robot
        self._camera.rotate_up()

        logging.error("BOX NOT FOUND, DOING A FULL ROTATION")
        box_relative_to_robot = self.search_for_box_for_angle(self.max_search_angle, direction="CCW")
        if box_relative_to_robot is not None:
            return box_relative_to_robot
        self._camera.reset_to_highest_angle()
        return None

    def do_rotate_box_to_centre(self, box_relative_to_robot: BoxRelativeToRobot) -> bool:
        """Returns whether the box is in the centre of the image"""
        lims = self._camera.get_midpoint_lims()
        box_x = box_relative_to_robot.x
        logging.info(f"Box coords: {box_relative_to_robot}. Lims: {lims}")
        if lims[0] <= box_x <= lims[1]:
            self._motion_controller.stop()
            logging.info("STOPPING, BOX IS IN CENTRE OF IMAGE")
            return True
        elif box_x < lims[0]:
            self._motion_controller.rotate_ccw(self.config.ROTATION_SPEED_SEARCH)
            logging.info("ROTATING CCW TO GET POINT IN CENTRE")
            return False
        elif box_x > lims[1]:
            self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_SEARCH)
            logging.info("ROTATING CW TO GET POINT IN CENTRE")
            return False

    def rotate_box_to_centre(self, known_box: BoxRelativeToRobot=None) -> Optional[BoxRelativeToRobot]:
        while True:
            self._motion_controller.stop()
            logging.info("STOPPING AND TAKING IMAGE: ROTATE BOX TO CENTRE")
            # use the image from found box first to determine which way to rotate
            if known_box is None:
                box_relative_to_robot = self.take_image_and_get_box()
            else:
                box_relative_to_robot = known_box
                known_box = None

            if box_relative_to_robot is None:
                self._motion_controller.stop()
                logging.info("TRYING TO ROTATE --- NO BOX FOUND. FAILING")
                return None
            
            ball_in_centre = self.do_rotate_box_to_centre(box_relative_to_robot)
            if ball_in_centre:
                return box_relative_to_robot
            
            time.sleep(self.config.SLEEP_TIME)

    def check_box_in_laser_sensor_range(self, box_relative_to_robot: BoxRelativeToRobot) -> bool:
        logging.info("CHECKING IF BOX IS IN LASER SENSOR RANGE BY COORDINATES")
        if box_relative_to_robot is None:
            return False
        _, h = self._camera.get_dimensions()
        logging.info(f"BOX BOTTOM COORDINATE: {box_relative_to_robot.ybot}. HEIGHT: {h}")
        logging.info(f"BOX IN LASER SENSOR RANGE: {box_relative_to_robot.ybot > h - h // 10}")
        return box_relative_to_robot.ybot > h - h // 100
    
    #######################
    # APPROACHING BOX AND DEPOSITING METHODS
    #######################

    def clear_tennis_balls_from_path(self, laser_distance):
        """Loops through camera angles until no tennis balls are in middle of image"""
        #TODO: can make this smarter by rotating angle based on laser sensor reading
        logging.info("CLEARING TENNIS BALLS FROM PATH")
        cleared_balls = False
        while True: # for every time we think there is a tennis ball in path
            self._camera.reset_to_highest_angle()
            while True: # for every camera angle
                target_ball = self.take_image_and_get_target_ball(with_map=False)
                # if we find a ball, clear it by traversing to it
                # TODO: this could cause bug where there is a closer tennis ball but slightly off centre which we traverse to and pick up
                if target_ball is not None:
                    angle_deg = np.degrees(np.arctan2(target_ball.y, target_ball.x))
                    logging.info(f"Relative position of ball ({target_ball.x},{target_ball.y}). Angle: {angle_deg} deg")
                    lims = (-self.config.CENTRE_THRESHOLD_DEG_BOX, self.config.CENTRE_THRESHOLD_DEG_BOX)
                    if lims[0] <= angle_deg <= lims[1]:
                        # collect the ball
                        target_ball = self.traverse_to_ball_with_rotating_camera(target_ball, with_map=False)
                        if target_ball is None:
                            break
                        ball_close_enough_to_pickup = self.approach_ball_with_laser_sensor(target_ball)
                        if not ball_close_enough_to_pickup:
                            break
                        # remove the ball from the map, assuming we picked it up
                        self._scooper.pick_up_ball()
                        robot_x, robot_y, _ = self.get_robot_state()
                        self._map.remove_ball_by_position((robot_x, robot_y))
                        self._map.visualise_map()
                        cleared_balls = True
                        break
                if not self._camera.rotate_down(clearing_path=True):
                    self._camera.reset_to_highest_angle()
                    return cleared_balls        

    def approach_box_with_laser_sensor(self) -> bool:
        logging.info("APPROACHING BOX SLOWLY WITH LASER SENSOR")
        start_x, start_y, _ = self.get_robot_state()
        while not self._laser_sensor.box_in_near_range():
            end_x, end_y, _ = self.get_robot_state()
            distance_driven = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
            if self.config.is_sunny and distance_driven > self.config.laser_sensor_box_far_threshold - self.config.laser_sensor_box_near_threshold:
                logging.info("LASER SENSOR DIDN'T ACTIVATE IN EXPECTED DISTANCE, PROBABLY SUNNY")
                self._motion_controller.stop()
                return False
            
            # approach box more slowly once we think we are close
            if distance_driven < self.config.laser_sensor_box_far_threshold - self.config.laser_sensor_box_near_threshold * 2:
                self._motion_controller.drive_forward(self.config.FORWARD_SPEED_APPROACH)
            else:
                self._motion_controller.drive_forward(self.config.FORWARD_SPEED_APPROACH*0.5)
            time.sleep(self.config.SHORT_SLEEP_TIME)
        self._motion_controller.stop()
        logging.info("BOX IN RANGE, STOPPING")
        return True
    
    def get_laser_sensor_min_distance_over_rotation(self, angle, direction, init_distance=10000) -> float:
        logging.info(f"DOING LASER SENSOR SWEEP FOR ANGLE {angle}")
        minimum_distance = init_distance
        start_angle = self.get_robot_theta()
        while abs(self.get_robot_theta() - start_angle) < angle:
            distance = self._laser_sensor.get_distance()
            if distance is not None and distance < minimum_distance:
                minimum_distance = distance
            if direction == "CW":
                self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_SEARCH)
            else:
                self._motion_controller.rotate_ccw(self.config.ROTATION_SPEED_SEARCH)
            time.sleep(self.config.SLEEP_TIME)
        self._motion_controller.stop()
        return minimum_distance
    
    def do_laser_sensor_sweep(self, angle) -> bool:
        """Sweep the laser sensor to find the closest point to the box"""
        init_distance = 10000
        cw_distance = self.get_laser_sensor_min_distance_over_rotation(angle, "CW", init_distance)
        ccw_distance = self.get_laser_sensor_min_distance_over_rotation(angle * 2, "CCW", init_distance)

        min_distance = min(cw_distance, ccw_distance)
        if min_distance == init_distance:
            logging.error("NO MINIMUM DISTANCE FOUND IN LASER SENSOR SWEEP. PROBABLY SUNNY")
            return None
        
        logging.info(f"Minimum distance found in laser sensor sweep: {min_distance:.2f}cm")
        # rotate until within threshold of minimum distance
        distance_thresh = min_distance + self.config.laser_sensor_sweep_threshold
        distance = distance_thresh + 1
        start_angle = self.get_robot_theta()
        while abs(self.get_robot_theta() - start_angle) < angle * 2 and distance >= distance_thresh:
            laser_dist = self._laser_sensor.get_distance()
            distance = laser_dist if laser_dist is not None else distance
            logging.debug(f"Distance: {distance:.2f}cm")
            self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_APPROACH)
            time.sleep(self.config.SLEEP_TIME)

        if distance >= distance_thresh:
            logging.error("DID NOT FIND SHORTEST POINT IN LASER SENSOR SWEEP, RETURNING TO INITIAL ANGLE")
            start_angle = self.get_robot_theta()
            while abs(self.get_robot_theta() - start_angle) < angle:
                self._motion_controller.rotate_ccw(self.config.ROTATION_SPEED_APPROACH)
                time.sleep(self.config.SLEEP_TIME)
            self._motion_controller.stop()
            return None

        return distance
    
    def do_jerk(self):
        # jerk back and forth
        logging.info("JERKING BACK AND FORTH")
        self._motion_controller.drive_forward(self.config.FORWARD_SPEED_TRAVERSE*2)
        time.sleep(self.config.SHORT_SLEEP_TIME)
        self._motion_controller.drive_backward(self.config.FORWARD_SPEED_TRAVERSE*2)
        time.sleep(self.config.SHORT_SLEEP_TIME)
        self._motion_controller.stop()
    
    def traverse_to_view_box_waypoint(self, view_box_waypoint=None):
        if view_box_waypoint == None:
            view_box_waypoint = self._map.view_box_waypoint
        logging.info("Traversing to the view box waypoint")
        box_waypoint = self._map.box_waypoint
        self.traverse_to_waypoint(view_box_waypoint, self.config.FORWARD_SPEED_TRAVERSE)

        logging.info("Rotating to face box at the view box waypoint")
        goal_theta = np.arctan2(box_waypoint[1]-view_box_waypoint[1], box_waypoint[0]-view_box_waypoint[0])
        while not self.rotate_to_theta(goal_theta, self.config.ROTATION_SPEED_SEARCH):
            self.get_robot_state() # update state for visualisation of map
            self._map.visualise_map()
            time.sleep(self.config.SLEEP_TIME)

    def traverse_to_box(self):
        self._camera.reset_to_highest_angle()
        logging.debug("Sleeping in traverse to box to let camera reset")
        time.sleep(1)
        # TODO: uncomment this
        self.traverse_to_view_box_waypoint()

        logging.info("Attempting to find the box model along with a sensor reading of it")
        while True:
            box_relative_to_robot = self.really_find_box()
            if box_relative_to_robot is None:
                # give next view box waypoint
                next_view_box_waypoint = self._map.get_next_view_box_waypoint()
                self.traverse_to_view_box_waypoint(next_view_box_waypoint)
                continue
            self.reset_angle_at_box_waypoint()
            laser_distance = self._laser_sensor.get_distance()
            if self.check_box_in_laser_sensor_range(box_relative_to_robot): #and 
                #laser_distance is not None and laser_distance < self._laser_sensor.get_box_far_threshold():
                # check that it isn't a tennis ball
                # TODO: getting rid of this
                break
                # cleared_balls = self.clear_tennis_balls_from_path(laser_distance)
                # if not cleared_balls:
                #     logging.error("NO TENNIS BALLS IN PATH, MOVING ON")
                #     break
                # else:
                #     logging.info("CLEAREDsear TENNIS BALLS, CONTINUING TO BOX BLINDLY")
                #     break
            else:
                logging.info(f"BOX NOT IN FAR RANGE, CONTINUING FORWARD TO FIND BOX: {laser_distance}cm")
                # min_distance_to_near_threshold_metres = (self._laser_sensor.get_box_far_threshold() - self._laser_sensor.get_box_near_threshold()) / 100
                # blind_forward_time_s = min_distance_to_near_threshold_metres / self.config.FORWARD_SPEED_TRAVERSE
                blind_forward_time_s = self.config.BOX_APPROACH_SLEEP_TIME
                logging.info(f"Traversing forward blindly for: {blind_forward_time_s:.2f}s, then trying to find box again")
                start_time = time.time()
                while time.time() - start_time < blind_forward_time_s:
                    self._motion_controller.drive_forward(self.config.FORWARD_SPEED_TRAVERSE)
                    time.sleep(self.config.SHORT_SLEEP_TIME)
                self._motion_controller.stop()

        logging.info("FOUND BOX, IN CENTRE AND NO TENNIS BALLS IN PATH. APPROACHING WITH LASER SENSOR")
        box_found_with_laser = self.approach_box_with_laser_sensor()

        if box_found_with_laser:        
            logging.debug("Box was found with laser sensor, doing sweep and reverse")
            last_distance = self.do_laser_sensor_sweep(self.fail_search_angle)
        else:
            logging.warning("Box was not found with laser sensor, skipping sweep and reverse. Hope for the best!")

        if last_distance is None:
            # reverse a little bit blindly
            logging.info("DOING A LITTLE REVERSE")
            start_time = time.time()
            while time.time() - start_time < self.config.box_reverse_time/2:
                self._motion_controller.drive_backward(self.config.FORWARD_SPEED_APPROACH)
                time.sleep(self.config.SHORT_SLEEP_TIME)
            self._motion_controller.stop()
        else:
            logging.info(f"DOING A LITTLE REVERSE WITH LAST DISTANCE: {last_distance:.2f}cm")
            distance_to_reverse = self.config.laser_sensor_box_near_threshold - last_distance
            start_x, start_y, _ = self.get_robot_state()
            while distance_to_reverse > 0:
                self._motion_controller.drive_backward(self.config.FORWARD_SPEED_APPROACH)
                time.sleep(self.config.SHORT_SLEEP_TIME)
                end_x, end_y, _ = self.get_robot_state()
                distance_driven = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                if distance_driven > distance_to_reverse:
                    break


        # DO 180 TURN
        logging.info("BOX IN RANGE, STOPPING AND DOING 180 TURN")
        start_angle = self.get_robot_theta()
        while abs(self.get_robot_theta() - start_angle) < np.pi:
            self._motion_controller.rotate_cw(self.config.ROTATION_SPEED_SEARCH)
            time.sleep(self.config.SLEEP_TIME)
        self._motion_controller.stop()

        # reverse a little bit
        logging.info("DOING A LITTLE REVERSE")
        start_time = time.time()
        start_x, start_y, _ = self.get_robot_state()
        while time.time() - start_time < self.config.box_reverse_time:
            self._motion_controller.drive_backward(self.config.FORWARD_SPEED_APPROACH)
            time.sleep(self.config.SHORT_SLEEP_TIME)
            # new_x, new_y, _ = self.get_robot_state()
            # distance = np.sqrt((new_x-start_x)**2 + (new_y-start_y)**2)
            # if distance > self.config.box_reverse_distance / 2:
            #     break
        self._motion_controller.stop()
        return True

    def deposit_balls(self):
        logging.warning("DEPOSITING TENNIS BALLS")
        time.sleep(1)
        self._rear_door.open_door()
        time.sleep(1)
        self.do_jerk()
        time.sleep(3)
        self._rear_door.close_door()
        self._balls_deposited += self._balls_in_storage
        self._balls_in_storage = 0
        logging.warning("DEPOSITED BALLS")

    #######################
    # BLIND DRIVING METHODS (VIA WAYPOINTS)
    #######################
    def get_waypoint_straight(self, distance_away):
        """For now use knowledge that ball is 12cm straight
        in front of robot when the area threshold is met"""
        x, y, th = self._motion_controller.get_state()
        x = x + distance_away * math.cos(th)
        y = y + distance_away * math.sin(th)
        return x, y
    
    def rotate_to_theta(self, goal_theta, rotation_speed):
        _, _, theta = self._motion_controller.get_state()
        dtheta = goal_theta - theta
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        is_aligned = abs(dtheta) <= self.config.WAYPOINT_ANGLE_THRESH_RADIANS
        logging.info(f"Goal theta: {goal_theta}, Current theta: {theta}, Diff: {dtheta}, Aligned: {is_aligned}")
        if is_aligned:
            self._motion_controller.stop()
            return True
        direction = dtheta / abs(dtheta)
        self._motion_controller.rotate(direction * rotation_speed)
        return False
    
    def drive_to_point(self, waypoint, forward_speed):
        x, y, theta = self.get_robot_state()
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        angle_to_waypoint = math.atan2(dy,dx)

        theta = math.atan2(math.sin(theta), math.cos(theta))
        dtheta = angle_to_waypoint - theta
        dtheta = math.atan2(math.sin(dtheta), math.cos(dtheta))

        is_aligned = abs(dtheta) <= self.config.WAYPOINT_ANGLE_THRESH_RADIANS        

        if is_aligned:
            self._motion_controller.drive_forward(forward_speed)
        else:
            direction = dtheta / abs(dtheta)
            self._motion_controller.rotate(direction * self.config.ROTATION_SPEED_SEARCH)

    def reached_point(self, waypoint):
        x, y, _ = self._motion_controller.get_state()
        dx = waypoint[0] - x
        dy = waypoint[1] - y
        distance = math.sqrt(dx**2 + dy**2)
        return distance < self.config.WAYPOINT_DISTANCE_THRESH
    
    def traverse_to_waypoint(self, waypoint, forward_speed):
        """Traverse to waypoint based on odometry"""
        while True:
            self.drive_to_point(waypoint, forward_speed)
            self._map.visualise_map()
            if self.reached_point(waypoint):
                logging.info(f"\n\nReached waypoint: {waypoint}\n\n")
                self._motion_controller.stop()
                break
            time.sleep(self.config.SLEEP_TIME)

    def try_safe_waypoint(self) -> Optional[BallRelativeToRobot]:
        waypoint = self._map.get_next_safe_waypoint()
        self.traverse_to_waypoint(waypoint, self.config.FORWARD_SPEED_TRAVERSE)
        print(f"STOPPING AT WAYPOINT AND TAKING PHOTO: {waypoint}")
        return self.take_image_and_get_target_ball()  

    def drive_to_middle(self):
        waypoint = self._map.get_middle_waypoint()
        self.traverse_to_waypoint(waypoint, self.config.FORWARD_SPEED_TRAVERSE)
        print(f"REACHED MIDDLE OF ARENA: {waypoint}")

    def drive_to_after_box_point(self):
        waypoint = self._map.get_after_box_point()
        self.traverse_to_waypoint(waypoint, self.config.FORWARD_SPEED_TRAVERSE)
        print(f"REACHED MIDDLE OF ARENA: {waypoint}")
    
    #######################
    # MAIN LOOP
    #######################

    def in_time_limit(self) -> bool:
        return time.time() - self._timer < self._time_limit
    
    def in_full_time_limit(self) -> bool:
        return time.time() - self._timer < self._full_time_limit

    def main_loop(self):
        at_initial_position = True
        while self.in_full_time_limit():
            if at_initial_position:
                # do a blind 90 degree turn map all balls
                self.do_initial_sweep()
                target_ball = None
                at_initial_position = False
            else:
                # waypoint = self.get_waypoint_straight(0.75)
                # self.traverse_to_waypoint(waypoint, self.config.FORWARD_SPEED_TRAVERSE)
                self.drive_to_after_box_point()
                # self._motion_controller.drive_forward(self.config.FORWARD_SPEED_APPROACH)
                # time.sleep(1)

            while self._balls_in_storage < self.config.MAX_BALLS_IN_STORAGE:
                if not self.in_time_limit():
                    logging.warning("TIME LIMIT REACHED, RETURNING TO BOX")
                    break

                self._last_ball_distance = 0 # reset ball distance as we don't have a target
                
                # TURN ON LED TO SHOW WE ARE IN TENNIS BALL COLLECTING MODE
                self.write_LED(searching=True)

                # search for ball if didn't find one by traversing to safe waypoint (on startup)
                target_ball = self.search_for_ball()
                
                # if rotated a full 360 and have a ball in storage, return to box
                if target_ball is None:
                    if self._done_360 and self._balls_in_storage > 0:
                        break
                    elif self._done_360:
                        # go to middle of the arena and try again
                        self.drive_to_middle()
                        continue
                
                logging.warning("FOUND TENNIS BALL, ROTATING TO CENTRE")
                self.write_LED(searching=False, collecting=True)
                
                # rotate until closest ball is in centre of image
                target_ball = self.rotate_ball_to_centre(known_target_ball=target_ball) # pass the target ball found in search
                self._motion_controller.stop()
                # lost tennis ball, do search again
                if target_ball is None:
                    continue

                # # check that ball isn't too close already 
                # if target_ball.distance > self.config.CAMERA_DISTANCE_THRESHOLD:
                #     # traverse until area of largest ball becomes larger than threshold
                #     target_ball = self.traverse_to_ball()
                #     self._motion_controller.stop()
                #     # lost tennis ball, do search again
                #     if target_ball is None:
                #         continue
                
                target_ball = self.traverse_to_ball_with_rotating_camera(target_ball)
                if target_ball is None:
                    continue

                ball_close_enough_to_pickup = self.approach_ball_with_laser_sensor(target_ball)

                if ball_close_enough_to_pickup:
                    self.pickup_ball()
                    self.remove_collected_ball_from_map()
                    self._balls_in_storage += 1


            logging.warning("COLLECTED TENNIS BALLS, GOING TO BOX TO DEPOSIT")
            # DEPOSITING (return to line)
            self.write_LED(searching=False, collecting=False, depositing=True)
            self.traverse_to_box()
            self.deposit_balls()
            self.reset_robot_state() #np.radians(self.config.reset_theta_angle_deg)) # we know we're at the box

            if not self.in_full_time_limit():
                break

        logging.warning(f"TIME LIMIT REACHED, EXITING. Deposited {self._balls_deposited} balls")

if __name__ == "__main__":
    # Load configuration from YAML file
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())

    # Pass the entire config dictionary to ScoopAndDump
    sad = ScoopAndDump(config)

    sad.start_motion_controller_processes()

    try:
        sad.main_loop()
    except KeyboardInterrupt:
        print("\n\n\n\n\n\n\n\n\n\n\nFinished testing\n\n\n\n\n\n\n\n\n\n\n\n")

    del(sad)
