import numpy as np
import control

class LQRControl:
    def __init__(self, **params):
        """
        Initialize the controller.
        
        Parameters: 
        params: dictionary with parameters of the rotor's 
        dynamics. These parameters are assumed to be constant.
            mass: float, mass of the vehicle.
            gravity: float, gravity constant. 
        """
        m = params["mass"] 
        g = params["gravity"]

        # Feedforward gain.
        self.uf = (m*g/2)*np.ones(2)

    def solve(self, x, **params):
        """
        Solve optimization problem and return gain matrix.
        
        Parameters:     
        x : numpy array
            Current state [x, y, theta, vx, vy, w]
        params: dictionary with parameters of the rotor's 
        dynamics. These parameters are assumed to be constant.
            mass: float, mass of the vehicle.
            inertia: float, inertia of the vehicle.
            beam_distance: float, distance between rotors.
            gravity: float, gravity constant. 
        """
        # Short notation for parameters.
        m = params["mass"] 
        J = params["inertia"]
        d = params["beam_distance"]
        g = params["gravity"]

        A = np.array([
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
            [0,0,-g,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0]
        ])
        B = np.array([
            [0, 0, 0, 0, 1/m, d/(2*J)],
            [0, 0, 0, 0, 1/m, -d/(2*J)]
        ]).T

        # Define state (Q) nd control (W) weighting matrices.
        Q = 0.1*np.eye(6)
        R = np.eye(2)
        # Solve problem.
        K, _, _ = control.lqr(A, B, Q, R)

        return -K

    def compute_input(self, x, K):
        """
        Compute control input u as u = Kx + uf, where
        uf is a feedforward term.
        
        Parameters:
        x: numpy array
            Current state [x, y, theta, vx, vy, w]
        K: numpu array matrix-alike
            Gain matrix.
        """
        du = K@x.reshape([6,1])
        return du.flatten() + self.uf