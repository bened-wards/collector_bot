"""
Class for the rear door, to control the servos :)


"""
from gpiozero import AngularServo
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory # to use hardware PWM


class RearDoor:

    def __init__(self, door_pin, open_angle, close_angle):
        # Is this servo correct 
        hardware_pwm_factory = PiGPIOFactory()
        self.door_servo = AngularServo(door_pin, 
                                       initial_angle=close_angle, 
                                       min_pulse_width=0.0006, max_pulse_width=0.0023, 
                                       pin_factory=hardware_pwm_factory) # SG90  

        self.open_door_angle = open_angle
        self.close_door_angle = close_angle
        
        # Adjust sleep times based on testing 
        self.sleep_time = 0.01
        self.slow_resolution = 1  
        self.fast_resolution = 5

        self.close_door()

    def open_door(self):
        # self.door_servo.angle = self.open_door_angle

        start_angle = int(self.door_servo.angle)
        end_angle = self.open_door_angle

        resolution = self.fast_resolution if start_angle < end_angle else -self.fast_resolution
        for angle in range(start_angle, end_angle, resolution):
            self.door_servo.angle = angle
            sleep(self.sleep_time)
        self.door_servo.angle = end_angle

    def close_door(self):
        start_angle = int(self.door_servo.angle)
        end_angle = self.close_door_angle

        resolution = self.slow_resolution if start_angle < end_angle else -self.slow_resolution
        for angle in range(start_angle, end_angle, resolution):
            self.door_servo.angle = angle
            sleep(self.sleep_time)
        self.door_servo.angle = end_angle


def test_door(rear_door):
    print("Door Open")
    rear_door.open_door()
    sleep(3)
    print("Door Closed")
    rear_door.close_door()
    sleep(3)

if __name__ == "__main__":

    from config import Config
    config = Config("config.yaml")

    rear_door = RearDoor(config.rear_door_servo_gpio_pin, config.rear_door_open_angle, config.rear_door_close_angle)

    try:
        test_door(rear_door)
    except KeyboardInterrupt:
        print("Finished testing")
