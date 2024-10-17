import time
import logging
import sys

import board
import adafruit_vl53l1x

from config import Config

class LaserSensor:

    def __init__(self, ball_distance_threshold, box_near_threshold, box_far_threshold, timing_budget):
        """Laser sensor (distance) using 

        Args:
            ball_distance_threshold (float): Distance threshold in metres for detecting ball
            box_near_threshold (float): Distance threshold in metres for detecting box close enough to 180 turn to deposit
            box_far_threshold (float): Distance threshold in metres for detecting box to drive straight at it
            timing_budget (int, optional): Timing budget in ms. 
                                            increasing = longer distance and improves repeatability
                                            + average power increases with duration
        """
        self._ball_threshold = ball_distance_threshold * 100 # convert to cm
        self._box_near_threshold = box_near_threshold * 100 # convert to cm
        self._box_far_threshold = box_far_threshold * 100 # convert to cm
        logging.info("Ball distance threshold: {} cm".format(self._ball_threshold))
        logging.info("Box far distance threshold: {} cm".format(self._box_far_threshold))
        logging.info("Box near distance threshold: {} cm".format(self._box_near_threshold))

        self.i2c = board.I2C()
        self.laser_sensor = adafruit_vl53l1x.VL53L1X(self.i2c)

        self.laser_sensor.distance_mode = 1  # short mode - more robust in high ambient light
        self.laser_sensor.timing_budget = timing_budget

        self.laser_sensor.start_ranging()

    def ball_in_range(self):
        """Returns True if ball is within threshold distance"""
        distance = self.get_distance()
        if distance is None:
            logging.warning("No distance data when checking ball in range")
            return False
        logging.debug(f"Distance when checking ball in range: {distance} cm")
        return distance < self._ball_threshold
    
    def get_ball_distance_threshold(self):
        return self._ball_threshold
    
    def box_in_near_range(self):
        """Returns True if box is within near threshold distance"""
        distance = self.get_distance()
        if distance is None:
            logging.warning("No distance data when checking box in near range")
            return False
        logging.debug(f"Distance when checking box in near range: {distance} cm")
        return distance < self._box_near_threshold
    
    def get_box_near_threshold(self):
        return self._box_near_threshold
    
    def box_in_far_range(self):
        """Returns True if box is within far threshold distance"""
        distance = self.get_distance()
        if distance is None:
            logging.warning("No distance data when checking box in far range")
            return False
        logging.debug(f"Distance when checking box in far range: {distance} cm")
        return distance < self._box_far_threshold

    def get_box_far_threshold(self):
        return self._box_far_threshold

    def get_distance(self):
        """Returns distance in cm"""
        # TODO: is this data stale?
        if self.laser_sensor.data_ready:
            distance = self.laser_sensor.distance
            self.laser_sensor.clear_interrupt()
            return distance
        return None
    
    def stop_ranging(self):
        self.laser_sensor.stop_ranging()


if __name__ == "__main__":
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())

    laser_sensor = LaserSensor(
        config.laser_sensor_ball_threshold, 
        config.laser_sensor_box_near_threshold, config.laser_sensor_box_far_threshold,
        config.laser_sensor_timing_budget)

    # TODO: do I need to clear interrupt over and over?
    while True:
        ball_in_range = laser_sensor.ball_in_range()
        print("Ball in range: {}".format(ball_in_range))
        time.sleep(0.4)
