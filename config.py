import yaml
import numpy as np

class Config:

    def __init__(self, yaml_path):
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        # Set all attributes from config dictionary
        for key, value in config.items():
            setattr(self, key, value)

        self.WAYPOINT_ANGLE_THRESH_RADIANS = np.radians(self.WAYPOINT_ANGLE_THRESH_DEGREES)
