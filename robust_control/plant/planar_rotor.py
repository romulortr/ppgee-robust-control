from scipy.integrate import solve_ivp

import numpy as np
import math

class PlanarRotor:
    def __init__(self, **params):
        """
        Initialize the plant.
        
        Parameters: 
        **params: dictionary with parameters of the rotor's 
        dynamics. These parameters are assumed to be constant.
            mass: float, mass of the vehicle.
            inertia: float, inertia of the vehicle.
            beam_distance: float, distance between rotors.
            gravity: float, gravity constant. 
        """
        self.x = np.zeros(6)
        self.m = params["mass"] if "mass" in params else 1
        self.J = params["inertia"] if "inertia" in params else 0.01
        self.d = params["beam_distance"] if "beam_distance" in params else 0.5
        self.g = params["gravity"] if "gravity" in params else 9.8

    def set_state(self, x):
        """
        Define the current state of the plant. 
        
        x : array-like
            Current state [x, y, theta, vx, vy, w]
        """
        self.x = x.flatten()

    def dynamics(self, t, x, u):
        """
        Defines the system dynamics dx/dt = [x3, x4, u1, u2]
        
        Parameters:
        t : float
            Current time (required by solve_ivp)
        x : array-like
            Current state [x, y, theta, vx, vy, w]
        u : array-like
            Input vector [u1, u2]
        """

        # Short notation for plant's parameters.
        m = self.m
        J = self.J
        d = self.d
        g = self.g
        
        # System dynamics.
        cos_theta, sin_theta = math.cos(x[2]), math.sin(x[2])
        A = np.array([
            [0,0,0,1,0,0],
            [0,0,0,0,1,0],
            [0,0,0,0,0,1],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
            [0,0,0,0,0,0],
        ])
        B = np.array([
            [0,0],
            [0,0],
            [0,0],
            [-sin_theta/m, -sin_theta/m],
            [cos_theta/m, cos_theta/m],
            [d/(2*J), -d/(2*J)]
        ])
        c = np.array([
            0,0,0,0,-g,0
        ]).reshape([6,1])
        
        dx = A@x.reshape([6,1]) + B@u.reshape([2,1]) + c        
        
        return dx.flatten()

    def iterate(self, u, dt):
        """
        Propagate the system forward in time by 'dt' using 
        numerical integration.
        
        Parameters:
        u : array-like
            Input vector [u1, u2]
        dt : float
            Time step
        """
        sol = solve_ivp(self.dynamics, [0, dt], self.x, 
            args=(u.flatten(),), method="RK45", t_eval=[dt])
        self.x = sol.y[:, -1]
        return self.x
