import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerPIDBasic(Controller):
    def __init__(self, kp=2.0, ki=0.0001, kd=0.5):
        self.path = None
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.acc_ep = 0
        self.last_ep = 0
        self.last_idx = 0
        self.lookahead = 2  

    def set_path(self, path):
        super().set_path(path)
        self.acc_ep = 0
        self.last_ep = 0
        self.last_idx = 0

    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        # Extract State
        x, y, dt, yaw = info["x"], info["y"], info["dt"], info["yaw"]

        # Search  Target
        min_idx, min_dist = utils.search_nearest(self.path, (x, y))
        self.last_idx = max(self.last_idx, min_idx)
        target_idx = min(self.last_idx + self.lookahead, len(self.path) - 1)
        target = self.path[target_idx]

        # TODO: PID Control for Basic Kinematic Model
        ang = np.arctan2(target[1] - y, target[0] - x)
        ep = min_dist * np.sin(ang-np.deg2rad(yaw))
        self.acc_ep += dt * ep
        diff_ep = (ep - self.last_ep) / dt
        next_w = self.kp * ep + self.ki * self.acc_ep + self.kd * diff_ep
        next_w = np.clip(next_w, -100, 100)
        self.last_ep = ep
        return next_w, target
