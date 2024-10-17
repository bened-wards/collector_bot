import numpy as np

class BallRelativeToRobot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.distance = np.sqrt(x**2 + y**2)

    def __str__(self):
        return f'BallToRobot: x: {self.x:.2f}, y: {self.y:.2f}, distance: {self.distance:.2f}'
    
    def __repr__(self):
        return self.__str__()

# TODO: invalidate balls if they are outside of our area
class BallOnMap:
    def __init__(self, id, x, y, distance_when_added, cov=np.eye(2), is_valid=True):
        self.id = id
        self.x = x
        self.y = y
        self.cov = cov
        self.is_valid = is_valid
        self.distance_when_added = distance_when_added

    def __str__(self):
        return f'BallOnMap: id: {self.id}, x: {self.x:.2f}, y: {self.y:.2f}, cov: {self.cov}'
    
    def __repr__(self):
        return self.__str__()

class BoxRelativeToRobot:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

        self.area = w * h

        self.ytop = y - h / 2
        self.ybot = y + h / 2

    def __str__(self):
        return f'BoxToRobot: x: {self.x:.2f}, y: {self.y:.2f}, w: {self.w:.2f}, h: {self.h:.2f}, area: {self.area:.2f}. ytop: {self.ytop:.2f}'
    
    def __repr__(self):
        return self.__str__()
