import logging
from typing import Optional

import numpy as np
import cv2

from tennis_types import BallOnMap, BallRelativeToRobot

class TennisMap:

    def __init__(self, 
                 starting_state, arena_no, arena_dimensions, 
                 box_dimensions, view_box_offsets,
                 ball_merge_threshold, distance_travelled_threshold,
                 distance_balls_close_no_remove_map):
        """Represents the map of the tennis court, robot's position in it, the box and tennis balls.

        Args:
            starting_state (tuple(float, float, float)): initial state of the robot (x, y, theta)
            arena_no (int): number of the arena (1, 2, 3, 4)
            arena_dimensions (tuple(float, float)): dimensions of the arena (length, width)
            box_dimensions (tuple(float, float)): dimensions of the box (length, width)
            view_box_offsets (tuple(float,float)): distance from the box to view it with camera (x, y)
            ball_merge_threshold (float): distance threshold to merge two balls
            distance_travelled_threshold (float): distance travelled threshold to remove balls from map
            distance_balls_close_no_remove_map (float): distance threshold to not remove balls from map when close enough
        """
        self._robot_state = starting_state
        self._arena = arena_no
        self._arena_dimensions = arena_dimensions
        self._box_dimensions = box_dimensions
        self._ball_merge_threshold = ball_merge_threshold

        self._balls = {} # ball_id: BallOnMap
        self._ball_id = 0

        self.initialise_safe_waypoints()
        self.initialise_home_waypoint()
        self.initialise_box_waypoint(view_box_offsets)
        self._current_box_offset = 0

        self._arena_corners = [(0, 0), (0, self._arena_dimensions[1]), 
                               (self._arena_dimensions[0], self._arena_dimensions[1]), 
                               (self._arena_dimensions[0], 0)]
        self._middle_box = self._arena_corners[2]
        
        self._distance_travelled = 0.0 # to remove balls from map after some distance travelled
        self._distance_travelled_threshold = distance_travelled_threshold
        self._distance_balls_close_no_remove_map = distance_balls_close_no_remove_map

    def initialise_safe_waypoints(self):
        # safe waypoints are defined along 45 degree line from origin 
        # and are scaled by 0.5, 1, 1.5, ... int(min(arena_dimensions))-1
        step = 0.5
        max_dim = min(self._arena_dimensions)
        safe_waypoints_scale = list(np.arange(step, max_dim, step))
        self._safe_waypoint_scale_iter = iter(safe_waypoints_scale)

    def initialise_home_waypoint(self):
        self._home_waypoint = (0.0, 0.0)

    def initialise_box_waypoint(self, view_box_offsets):
        """Box waypoint is corner of box closest to starting point.
        View box waypoint is offset from the box waypoint to view the box with camera."""
        if self._arena == 1 or self._arena == 4:
            raw_waypoint = list(np.array(self._arena_dimensions) - np.array(self._box_dimensions) / 2)
            self._box_waypoint = [raw_waypoint[0], -raw_waypoint[1]]
            self._view_box_waypoint = [self._arena_dimensions[0] - view_box_offsets[0], -(self._arena_dimensions[1] - view_box_offsets[1])]
        elif self._arena == 2 or self._arena == 3:
            self._box_waypoint = [self._arena_dimensions[0] - self._box_dimensions[0] / 2, self._arena_dimensions[1]]
            # list(np.array(self._arena_dimensions) - np.array(self._box_dimensions) / 2)
            self._view_box_waypoint = [self._arena_dimensions[0] - view_box_offsets[0], self._arena_dimensions[1] - view_box_offsets[1]]
        logging.info(f"Box waypoint: {self._box_waypoint}, View box waypoint: {self._view_box_waypoint}")
    
    ## GETTERS ##
    #############
    def get_next_safe_waypoint(self):
        try:
            waypoint_scale = next(self._safe_waypoint_scale_iter)
        except StopIteration:
            logging.info("Ran out of safe waypoints, resetting")
            self.initialise_safe_waypoints()
            waypoint_scale = next(self._safe_waypoint_scale_iter)

        # if self.search_direction == "CW":
        #     return [(next_scale * 1.0, -next_scale * 1.0) for next_scale in waypoint_scales]
        # elif self.search_direction == "CCW":
        #     return [(next_scale * 1.0, next_scale * 1.0) for next_scale in waypoint_scales]
        if self._arena == 1 or self._arena == 4:
            return (waypoint_scale * 1.0, -waypoint_scale * 1.0)
        elif self._arena == 2 or self._arena == 3:
            return (waypoint_scale * 1.0, waypoint_scale * 1.0)
        
    def get_after_box_point(self):
        return (self._arena_dimensions[0] - 1, self._arena_dimensions[1] - 1)
    
    @property
    def home_waypoint(self):
        return self._home_waypoint
    
    @property
    def box_waypoint(self):
        return self._box_waypoint
    
    @property
    def view_box_waypoint(self):
        return self._view_box_waypoint
    
    def get_next_view_box_waypoint(self):
        self._current_box_offset -= 0.2
        return self._view_box_waypoint[0], self._view_box_waypoint[1] + self._current_box_offset
    
    def get_middle_waypoint(self):
        return self._arena_dimensions[0] / 2, self._arena_dimensions[1] / 2
    
    def get_angle_to_furthest_corner(self):
        x, y, th = self._robot_state
        max_distance = 0
        for corner in self._arena_corners:
            distance = np.sqrt((corner[0] - x)**2 + (corner[1] - y)**2)
            if distance > max_distance:
                max_distance = distance
                furthest_corner = corner
        # angle to furthest corner
        relative_angle = np.arctan2(furthest_corner[1] - y, furthest_corner[0] - x)
        th = np.arctan2(np.sin(th), np.cos(th))
        angle = relative_angle - th
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        return angle

    
    ## MODIFICATION METHODS ##
    ##########################
    def update_robot_state(self, new_state):
        """Update robot state from odometry"""
        # track distance travelled
        x, y, th = self._robot_state
        new_x, new_y, new_th = new_state
        self._distance_travelled += np.sqrt((new_x - x)**2 + (new_y - y)**2)

        self._robot_state = new_state

    def correct_robot_state(self, corrected_state):
        """To be called when we reach some known position (e.g. box and have deposited ball)"""
        logging.warning(f"OVERWRITING ROBOT STATE IN MAP WITH CORRECTED STATE. MAKE SURE THIS IS A KNOWN POSITION: {self._robot_state} to {corrected_state}")
        self.update_robot_state(corrected_state)

    def clear_balls(self):
        """To be called if we reached some known position and want to restart"""
        logging.warning("REMOVING ALL BALLS FROM MAP")
        self._balls = {}

    def add_balls(self, balls_relative_to_robot: list[BallRelativeToRobot]) -> list[BallRelativeToRobot]:
        """Add balls to the map when new balls are detected. Returns balls that were successfully added to the map."""
        balls_added = []
        for ball in balls_relative_to_robot:
            if self.add_ball((ball.x, ball.y)):
                balls_added.append(ball)

        # TODO: probably move this out of here
        self.visualise_map()
        return balls_added

    def add_ball(self, relative_position) -> bool:
        """Add a ball to the map when a new ball is detected. 
        Checks whether ball is in the arena.
        Checks whether ball should be merged with an existing ball already in the map.

        Args:
            relative_position (tuple(float, float)): relative position of the ball in the robot frame

        Returns:
            bool: whether the ball was added to the map or not
        """
        # convert relative position in robot frame to map frame
        # merge ball with balls currently in map
        x_ball_rel, y_ball_rel = relative_position
        x, y, th = self._robot_state
        x_ball_map = x + x_ball_rel * np.cos(th) - y_ball_rel * np.sin(th)
        y_ball_map = y + x_ball_rel * np.sin(th) + y_ball_rel * np.cos(th)

        if not self._check_if_position_in_bounds((x_ball_map, y_ball_map)):
            logging.debug(f"Ball detected out of bounds at {x_ball_map, y_ball_map}")
            return False

        ball_id = self._check_if_merge_ball((x_ball_map, y_ball_map))
        if ball_id is not None:
            merged_coords = self._merge_ball(ball_id, (x_ball_map, y_ball_map))
            logging.debug(f"Merged ball with existing ball: {ball_id} at {merged_coords}")
        else:
            self._add_ball_to_map((x_ball_map, y_ball_map))
            logging.debug(f"Added new ball to map at {x_ball_map, y_ball_map}")
        return True

    def remove_ball_by_position(self, rough_position):
        """Remove ball by rough position on map when a ball has been collected or determined to be out of bounds"""
        ball = self._get_closest_ball(rough_position)
        if ball is not None:
            self._remove_ball_by_id(ball.id)
            logging.debug(f"Removed ball from map at {rough_position}")
        else:
            logging.warning(f"Could not find ball to remove at {rough_position}")

    def get_closest_ball_position_to_robot(self):
        """Get the position of the closest ball to the robot (x,y). Returns None if no balls in map."""
        ball = self._get_closest_ball_to_robot()
        if ball is not None:
            logging.info(f"Closest ball to robot ({ball.id}) at {ball.x, ball.y}")
            return ball.x, ball.y
        else:
            return None

    def get_closest_ball_position_to(self, position):
        """Get the position of the closest ball to a given position (x,y). Returns None if no balls in map."""
        ball = self._get_closest_ball(position)
        if ball is not None:
            return ball.x, ball.y
        else:
            return None

    ## PRIVATE METHODS TO MANAGE BALLS ##
    #####################################
    def _add_ball_to_map(self, map_position):
        """Add ball to map when a new ball is detected"""
        self._ball_id += 1
        self._balls[self._ball_id] = BallOnMap(self._ball_id, map_position[0], map_position[1], self._distance_travelled)

    def _remove_ball_by_id(self, ball_id):
        """Remove ball by id when a ball has been collected or determined to be out of bounds"""
        self._balls.pop(ball_id)

    def _remove_stale_balls(self):
        """Remove balls that have been in the map for too long"""
        stale_ball_ids = []
        for ball_id, ball in self._balls.items():
            x, y, th = self._robot_state
            dist = np.sqrt((ball.x - x)**2 + (ball.y - y)**2)
            if dist < self._distance_balls_close_no_remove_map:
                continue
            if self._distance_travelled - ball.distance_when_added > self._distance_travelled_threshold:
                stale_ball_ids.append(ball_id)
        for ball_id in stale_ball_ids:
            self._remove_ball_by_id(ball_id)
            logging.debug(f"Removed stale ball from map: {ball_id}")

    def _get_closest_ball_to_robot(self) -> Optional[BallOnMap]:
        """Get BallOnMap object of the closest ball to the robot. Returns None if no balls in map."""
        robot_x, robot_y, _ = self._robot_state
        return self._get_closest_ball((robot_x, robot_y))
    
    def _get_closest_ball(self, position) -> Optional[BallOnMap]:
        """Get BallOnMap object of the closest ball to a given position (x,y). Returns None if no balls in map."""
        # TODO: only gets closest ball based on x/y. should consider rotation time too

        # remove balls that have been in the map for too long
        self._remove_stale_balls()

        if len(self._balls) == 0:
            return None
        min_dist = float('inf')
        x, y = position
        for ball in self._balls.values():
            dist = np.sqrt((ball.x - x)**2 + (ball.y - y)**2)
            if dist < min_dist:
                min_dist = dist
                closest_ball_id = ball.id
        return self._balls[closest_ball_id]

    def _check_if_merge_ball(self, map_position):
        """Check if the ball is the same as one already in the map.

        Args:
            map_position (tuple(float, float)): x, y position of the ball in the map frame
        Returns:
            int or None: ball id if the ball is within the merge threshold of another ball, None otherwise
        """
        closest_ball = self._get_closest_ball(map_position)
        if closest_ball is None:
            return None

        dist = np.sqrt((closest_ball.x - map_position[0])**2 + (closest_ball.y - map_position[1])**2)
        if dist < self._ball_merge_threshold:
            return closest_ball.id
        else:
            return None

    def _merge_ball(self, ball_id, map_position):
        """
        To be called when a ball is determined to be the same.
        Will perform fusion based on one of:
        - variance (x and y)
        - simple average
        """
        ball = self._balls[ball_id]
        new_x = (ball.x + map_position[0]) / 2
        new_y = (ball.y + map_position[1]) / 2
        
        self._balls[ball_id].x = new_x
        self._balls[ball_id].y = new_y

        return new_x, new_y

    def _check_if_ball_in_bounds(self, ball_id):
        """Check if a ball is in bounds of the arena"""
        ball = self._balls[ball_id]
        if ball.x < 0 or ball.x > self._arena_dimensions[0] or ball.y < 0 or ball.y > self._arena_dimensions[1]:
            return False
        return True
    
    def _check_if_position_in_bounds(self, position):
        """Check if a position is in bounds of the arena"""
        x, y = position
        if x < 0 or x > self._arena_dimensions[0] or y < 0 or y > self._arena_dimensions[1]:
            return False
        return True
    
    def visualise_map(self):
        # Arena dimensions (add margin for the white lines at the boundaries)
        margin = 10
        scale = 100
        arena_length, arena_width = self._arena_dimensions
        img_width = int(arena_width * scale + 2 * margin)  # Scale up to make it more visible
        img_height = int(arena_length * scale + 2 * margin)
        
        # Create a blue background
        img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        img[:] = (184, 125, 55) # Blue background (BGR)

        # Draw white boundary lines
        cv2.rectangle(img, (margin, margin), (img_width - margin, img_height - margin), (255, 255, 255), 2)

        # Draw the brown box in the top-left corner (centered at the corner)
        box_length, box_width = self._box_dimensions
        box_length, box_width = box_length / 2, box_width / 2 # only half in our arena
        box_top_left = (margin, margin)
        box_bottom_right = (int(box_width * scale), int(box_length * scale))
        cv2.rectangle(img, box_top_left, box_bottom_right, (121, 171, 234), -1)  # Brown color (BGR)

        # Draw the robot as a pink circle
        robot_x, robot_y, robot_theta = self._robot_state
        robot_center = (int(img_width - robot_y * scale - margin), int(img_height - robot_x * scale - margin))
        robot_radius = int(0.15 * scale) 
        cv2.circle(img, robot_center, robot_radius, (255, 105, 180), -1)  # Pink color (BGR)

        arrow_length = 0.25 * scale
        arrow_end = (
            int(robot_center[0] - arrow_length * np.sin(robot_theta)),
            int(robot_center[1] - arrow_length * np.cos(robot_theta))
        )
        cv2.arrowedLine(img, robot_center, arrow_end, (255, 105, 180), 2, tipLength=0.3)

        # Draw tennis balls as yellow/green circles
        ball_radius = int(0.031 * scale)
        for ball in self._balls.values():
            ball_center = (int(img_width - ball.y * scale - margin), int(img_height - ball.x * scale - margin))
            cv2.circle(img, ball_center, ball_radius, (119, 223, 172), -1)

        # Display the image
        cv2.imshow('TennisMap', img)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()


if __name__ == "__main__":
    from detect import Detector
    from config import Config
    from camera import Camera
    import sys

    # Load configuration from YAML file
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())

    tennis_map = TennisMap(config.starting_state, config.arena, config.arena_dimensions, 
                           config.box_dimensions, config.view_box_offsets, config.ball_merge_threshold)
    
    # create camera
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
    # create detector to add balls
    ball_detector = Detector(camera, config.ball_model_path, config.ball_confidence_thresh, config.ball_iou_thresh, config.ball_dimensions, config.ball_distance_thresh, "pics")
    
    # loop adding balls to map
    while True:
        balls_relative_to_robot = ball_detector.get_relative_positions(save_image=True)
        tennis_map.add_balls(balls_relative_to_robot)
        tennis_map.visualise_map()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
