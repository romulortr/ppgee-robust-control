import numpy as np
from robust_control.plant.planar_rotor import PlanarRotor
from robust_control.controller.lqr import LQRControl
from robust_control.vizualisation.viz_planar_rotor import plot_planar_rotor_state_and_input

def main():
    # Simulation parameters.
    final_time = 10
    sampling_time = 0.05
    # Plant parameters.
    plantParams = {
        "mass": 1,
        "inertia": 0.01,
        "beam_distance": 0.5,
        "gravity": 9.8
    }
    # Initial state.
    x0 = np.array([0.5, -0.5, 0, 0, 0, 0])

    # Create plant and set its initial condition.
    plant = PlanarRotor(**plantParams)  
    plant.set_state(x0)

    # Create vectors for time, state, and state dynamics.
    t = np.arange(start=0, stop=final_time, step=sampling_time, dtype=float)
    x = np.zeros([6, len(t)+1], dtype=float)
    dx = np.zeros([6, len(t)], dtype=float)
    u = np.zeros([2, len(t)], dtype=float)
    x[:,0] = x0

    # Create contoller.
    control = LQRControl(**plantParams)
    
    # Find gains.
    K = control.solve(x0, **plantParams)

    for i, _ in enumerate(t, start=0):
        u[:,i] = control.compute_input(x[:,i], K)
        x[:,i+1] = plant.iterate(u[:,i], sampling_time)
    
    plot_planar_rotor_state_and_input(t, x[:,:-1], u)

        
if __name__ == "__main__":
    main()
