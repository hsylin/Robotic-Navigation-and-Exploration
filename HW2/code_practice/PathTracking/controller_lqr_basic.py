import sys
import numpy as np 
sys.path.append("..")
import PathTracking.utils as utils
from PathTracking.controller import Controller

class ControllerLQRBasic(Controller):
    def __init__(self, Q=np.eye(4), R=np.eye(1)):
        self.path = None
        self.Q = Q
        self.Q[0,0] = 100
        self.Q[1,1] = 1
        self.Q[2,2] = 100
        self.Q[3,3] = 1
        self.R = R*5000
        self.pe = 0
        self.pth_e = 0

    def set_path(self, path):
        super().set_path(path)
        self.pe = 0
        self.pth_e = 0
    
    def _solve_DARE(self, A, B, Q, R, max_iter=150, eps=0.01): # Discrete-time Algebra Riccati Equation (DARE)
        P = Q.copy()
        for i in range(max_iter):
            temp = np.linalg.inv(R + B.T @ P @ B)
            Pn = A.T @ P @ A - A.T @ P @ B @ temp @ B.T @ P @ A + Q
            if np.abs(Pn - P).max() < eps:
                break
            P = Pn
        return Pn

    # State: [x, y, yaw, delta, v, l, dt]
    def feedback(self, info):
        # Check Path
        if self.path is None:
            print("No path !!")
            return None, None
        
        # Extract State 
        x, y, yaw, v, dt = info["x"], info["y"], info["yaw"], info["v"], info["dt"]
        yaw = utils.angle_norm(yaw)
        
        # Search Nesrest Target
        min_idx, min_dist = utils.search_nearest(self.path, (x,y))
        target = self.path[min_idx]
        target[2] = utils.angle_norm(target[2])
        
        # TODO: LQR Control for Basic Kinematic Model
        e = -np.sin(np.deg2rad(yaw)) * (x -target[0]) + np.cos(np.deg2rad(yaw)) * (y-target[1])
        th_e = utils.angle_norm(yaw - target[2])
        th_e = np.deg2rad(th_e)
        X = np.array([[e], [(e-self.pe)/dt], [th_e], [(th_e-self.pth_e)/dt]])  
        A = np.array([
            [1, dt, 0, 0],
            [0, 0, v, 0],
            [0, 0, 1, dt],
            [0, 0, 0, 0]
        
        ])
        B = np.array([
            [0],
            [0],
            [0],
            [1]
        ])
	
        P_next = self._solve_DARE(A, B, self.Q, self.R)
        K = np.linalg.inv(self.R + B.T @ P_next @ B) @ B.T @ P_next @ A
        
        u = -K @ X
        next_w = u[0, 0]
        next_w = np.rad2deg(next_w )
        #next_delta = utils.angle_norm(next_delta)
        self.pe = e
        self.pth_e = th_e
        return next_w, target
