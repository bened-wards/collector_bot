"""
Class for the scoop arm, this will control servos and 


"""
from gpiozero import AngularServo
from time import sleep
import logging
from gpiozero.pins.pigpio import PiGPIOFactory # to use hardware PWM

class Scooper:

    def __init__(self, claw_pin, arm_pin, 
                 claw_open_angle, claw_close_angle, 
                 arm_up_angle, arm_down_angle):
        
        self.hardware_pwm_factory = PiGPIOFactory()
        self.claw_servo = AngularServo(claw_pin, 
                                       initial_angle=claw_open_angle, 
                                       min_pulse_width=0.0006, max_pulse_width=0.0023,
                                       pin_factory=self.hardware_pwm_factory) # SG90  
        self.arm_servo = AngularServo(
            arm_pin, initial_angle=arm_down_angle, min_pulse_width=0.0006, max_pulse_width=0.0023,
            pin_factory=self.hardware_pwm_factory)
        
        self.claw_open_angle = claw_open_angle
        self.claw_close_angle = claw_close_angle
        self.arm_up_angle = arm_up_angle
        self.arm_down_angle = arm_down_angle

        logging.info(f"Claw servo on pin {claw_pin} initialized at {self.claw_servo.angle} degrees")
        logging.info(f"Arm servo on pin {arm_pin} initialized at {self.arm_servo.angle} degrees")
        logging.info(f"Claw open angle: {claw_open_angle}")
        logging.info(f"Claw close angle: {claw_close_angle}")
        logging.info(f"Arm up angle: {arm_up_angle}")
        logging.info(f"Arm down angle: {arm_down_angle}")

        self.sleep_time = 0.01 # seconds

        self.slow_resolution = 1
        self.fast_resolution = 5

        self.reset_arm()

    # Adjust sleep times based on testing 
    def open_claw(self):
        # self.claw_servo.angle = self.claw_open_angle
        start_angle = int(self.claw_servo.angle)
        end_angle = self.claw_open_angle

        resolution = self.fast_resolution if start_angle < end_angle else -self.fast_resolution
        for angle in range(start_angle, end_angle, resolution):
            self.claw_servo.angle = angle
            sleep(self.sleep_time)

    def close_claw(self):
        # self.claw_servo.angle = self.claw_close_angle
        start_angle = int(self.claw_servo.angle)
        end_angle = self.claw_close_angle

        resolution = self.fast_resolution if start_angle < end_angle else -self.fast_resolution
        for angle in range(start_angle, end_angle, resolution):
            self.claw_servo.angle = angle
            sleep(self.sleep_time)
        self.claw_servo.angle = end_angle

    def lift_arm(self):
        # self.arm_servo.angle = self.arm_up_angle
        start_angle = int(self.arm_servo.angle)
        end_angle = self.arm_up_angle

        resolution = self.fast_resolution if start_angle < end_angle else -self.fast_resolution
        for angle in range(start_angle, end_angle, resolution):
            self.arm_servo.angle = angle
            sleep(self.sleep_time)
        self.arm_servo.angle = end_angle

    def drop_arm(self):
        start_angle = int(self.arm_servo.angle)
        end_angle = self.arm_down_angle

        resolution = self.slow_resolution if start_angle < end_angle else -self.slow_resolution
        for angle in range(start_angle, end_angle, resolution):
            self.arm_servo.angle = angle
            sleep(self.sleep_time)

    def reset_arm(self):
        logging.debug("Resetting arm")
        self.drop_arm()
        self.open_claw()

    def pick_up_ball(self):
        logging.debug("Picking up ball")
        self.close_claw()
        self.lift_arm()
        self.open_claw()

        sleep(1) # Time for ball to roll into box

        self.drop_arm()

    def test_claw(self):
        for i in range(15):
            print("Claw Open")
            self.open_claw()
            sleep(3)
            print("Claw Closed")
            self.close_claw()
            sleep(3)

    def test_arm(self):
        for i in range(15):
            print("Arm Up")
            self.lift_arm()
            sleep(3)
            print("Arm Down")
            self.drop_arm()
            sleep(3)


if __name__ == "__main__":
    from config import Config
    import sys
    config = Config("config.yaml")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logging.getLogger().setLevel(config.log_level.upper())



    CLAW_OPEN_ANGLE = config.claw_open_angle
    CLAW_CLOSE_ANGLE = config.claw_close_angle
    ARM_UP_ANGLE = config.arm_up_angle
    ARM_DOWN_ANGLE = config.arm_down_angle

    claw_pin = config.claw_servo_gpio_pin
    arm_pin = config.arm_servo_gpio_pin

    arm = Scooper(claw_pin, arm_pin, CLAW_OPEN_ANGLE, CLAW_CLOSE_ANGLE, ARM_UP_ANGLE, ARM_DOWN_ANGLE)

    # arm.open_claw()
    arm.drop_arm()

    # try:
    #     arm.test_claw()
    # except KeyboardInterrupt:
    #     print("Finished testing")

