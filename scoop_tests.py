import logging
import sys
import time
import numpy as np

from scoop_and_dump import ScoopAndDump, Config

class ScoopAndDumpTest(ScoopAndDump):

    def __init__(self, config: Config):
        super().__init__(config)

    # TODO: write integration tests

    
    
    def find_tennis_ball_and_traverse_with_camera_only(self):
        target_ball = self.search_for_ball()

        logging.warning("FOUND TENNIS BALL, ROTATING TO CENTRE")
        self.write_LED(searching=False, collecting=True)

        # rotate until closest ball is in centre of image
        target_ball = self.rotate_ball_to_centre(known_target_ball=target_ball)
        self._motion_controller.stop()
        # lost tennis ball, do random search
        if target_ball is None:
            return False
        
        target_ball = self.traverse_to_ball_with_rotating_camera(target_ball)
        if target_ball is None:
            return False
        
        return True

    def find_tennis_ball_and_traverse_with_camera_and_laser_sensor(self):
        target_ball = self.search_for_ball()

        logging.warning("FOUND TENNIS BALL, ROTATING TO CENTRE")
        self.write_LED(searching=False, collecting=True)

        # rotate until closest ball is in centre of image
        target_ball = self.rotate_ball_to_centre(known_target_ball=target_ball)
        self._motion_controller.stop()
        # lost tennis ball, do random search
        if target_ball is None:
            return False
        
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
    
    def find_tennis_ball_and_traverse_then_go_to_second_ball(self):
        ball_close_enough_to_pickup = self.find_tennis_ball_and_traverse_with_camera_and_laser_sensor()

        if not ball_close_enough_to_pickup:
            return False
        
        # go to next target (check if any stored in map)
        ball_close_enough_to_pickup2 = self.find_tennis_ball_and_traverse_with_camera_and_laser_sensor()

        return ball_close_enough_to_pickup and ball_close_enough_to_pickup2
    
    def find_tennis_ball_and_traverse_and_pickup(self):
        ball_close_enough_to_pickup = self.find_tennis_ball_and_traverse_with_camera_and_laser_sensor()

        if not ball_close_enough_to_pickup:
            return False
        
        self.pickup_ball()

        return True
    
    def find_tennis_ball_and_traverse_and_pickup_and_go_to_second_ball(self):
        ball1 = self.find_tennis_ball_and_traverse_and_pickup()
        ball2 = self.find_tennis_ball_and_traverse_and_pickup()
        return ball1 and ball2
    
    def test_pickup_ball(self):
        self.pickup_ball()

    def find_tennis_ball_and_traverse_and_pickup_and_drop(self):
        ball1 = self.find_tennis_ball_and_traverse_and_pickup()
        if ball1:
            self.deposit_balls()
        return ball1
    
    def find_2_tennis_ball_and_traverse_and_pickup_and_drop(self):
        ball1 = self.find_tennis_ball_and_traverse_and_pickup()
        ball2 = self.find_tennis_ball_and_traverse_and_pickup()
        if ball1 and ball2:
            self.deposit_balls()
        return ball1 and ball2
    
    def find_balls_and_traverse_and_pickup_infinite(self):
        try:
            balls_found = 0
            while True:
                ball_found = self.find_tennis_ball_and_traverse_and_pickup()
                if not ball_found:
                    continue
                else:
                    balls_found += 1
                
                if balls_found == 5:
                    break
            self.deposit_balls()
        except KeyboardInterrupt:
            return True
        
    def traverse_to_box_and_deposit_balls(self):
        ready_to_deposit = self.traverse_to_box()
        if not ready_to_deposit:
            return False
        
        self.deposit_balls()
        return True
    
    def find_balls_and_traverse_and_deposit_balls(self):
        try:
            balls_found = 0
            while True:
                ball_found = self.find_tennis_ball_and_traverse_and_pickup()
                if not ball_found:
                    continue
                else:
                    balls_found += 1
                
                if balls_found == 4:
                    break
            return self.traverse_to_box_and_deposit_balls()
        except KeyboardInterrupt:
            return True
        
    def comp_scenario(self):
        self.do_initial_sweep()

        self.find_balls_and_traverse_and_deposit_balls()

    def test_jerk(self):
        self._rear_door.open_door()
        time.sleep(2)
        self.do_jerk()
        time.sleep(2)
        self._rear_door.close_door()

    def test_reset_odometry(self):
        # drive for a bit
        start_time = time.time()
        while time.time() - start_time < 2:
            self._motion_controller.drive_straight(0.1)
            time.sleep(0.1)
        self._motion_controller.stop()
        before_state = self.get_robot_state()

        self.reset_robot_state()
        after_state = self.get_robot_state()

        logging.info(f"Before reset: {before_state}")
        logging.info(f"After reset: {after_state}")
        return before_state != after_state

if __name__ == "__main__":
    TEST_NAME = "MAIN"


    # Load configuration from YAML file
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())

    # Pass the entire config dictionary to ScoopAndDump
    sad_test = ScoopAndDumpTest(config)

    sad_test.start_motion_controller_processes()

    # TODO: call the test functions we want to run here
    if TEST_NAME == "TENNIS_BALL_DRIVE":
        while input("more tests? (y/n): ") == "y":
            print("Which test?")
            print("1. find_tennis_ball_and_traverse_with_camera_only")
            print("2. find_tennis_ball_and_traverse_with_camera_and_laser_sensor")
            print("3. find_tennis_ball_and_traverse_then_go_to_second_ball")
            print("4. find_tennis_ball_and_traverse_and_pickup")
            print("5. find_tennis_ball_and_traverse_and_pickup_and_go_to_second_ball")  
            print("6. find_tennis_ball_and_traverse_and_pickup_and_drop")
            print("7. find_2_tennis_ball_and_traverse_and_pickup_and_drop")
            print("8. find_balls_and_traverse_and_pickup_infinite")
            test = input("Test: ")
            if test == "1":
                success = sad_test.find_tennis_ball_and_traverse_with_camera_only()
                if not success:
                    logging.error("Failed to find tennis ball")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "2":
                success = sad_test.find_tennis_ball_and_traverse_with_camera_and_laser_sensor()
                if not success:
                    logging.error("Failed to find tennis ball")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "3":
                success = sad_test.find_tennis_ball_and_traverse_then_go_to_second_ball()
                if not success:
                    logging.error("Failed to pick up both balls")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "4":
                success = sad_test.find_tennis_ball_and_traverse_and_pickup()
                if not success:
                    logging.error("Failed to pick up ball")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "5":
                success = sad_test.find_tennis_ball_and_traverse_and_pickup_and_go_to_second_ball()
                if not success:
                    logging.error("Failed to pick up both balls")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "6":
                success = sad_test.find_tennis_ball_and_traverse_and_pickup_and_drop()
                if not success:
                    logging.error("Failed to pick up and drop ball")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "7":
                success = sad_test.find_2_tennis_ball_and_traverse_and_pickup_and_drop()
                if not success:
                    logging.error("Failed to pick up and drop both balls")
                else:
                    logging.info("TEST SUCCESS")
            elif test == "8":
                success = sad_test.find_balls_and_traverse_and_pickup_infinite()
                if not success:
                    logging.error("Failed to pick up and drop both balls")
                else:
                    logging.info("TEST SUCCESS")
            else:
                print("Invalid test")
            sad_test.reset_robot_state()
    elif TEST_NAME == "PICKUP_BALL":
        while input("more tests? (y/n): ") == "y":
            sad_test.test_pickup_ball()
    elif TEST_NAME == "BOX":
        while input("more tests? (y/n): ") == "y":
            sad_test.traverse_to_box_and_deposit_balls()
    elif TEST_NAME == "JERK":
        while input("more tests? (y/n): ") == "y":
            sad_test.test_jerk()
    elif TEST_NAME == "RESET":
        while input("more tests? (y/n): ") == "y":
            sad_test.test_reset_odometry()
    elif TEST_NAME == "INTEGRATION":
        while input("more tests? (y/n): ") == "y":
            sad_test.find_balls_and_traverse_and_deposit_balls()
    elif TEST_NAME == "COMP":
        while input("more tests? (y/n): ") == "y":
            sad_test.comp_scenario()
    elif TEST_NAME == "MAIN":
        try:
            sad_test.main_loop()
        except KeyboardInterrupt:
            print("Finished testing")

    del(sad_test)
