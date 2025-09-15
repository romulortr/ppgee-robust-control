import matplotlib.pyplot as plt

def plot_planar_rotor_state_and_input(t, state, input):
    # Create a figure with 6 subplots
    fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)

    # Labels for the states
    state_labels = [r'$x$ [m]', r'$y$ [m]', r'$\theta$ [rad]', 
                    r'$\dot{x}$ [m/s]', r'$\dot{y}$ [m/s]', r'$\dot{\theta}$ [rad/s]']
    # Plot each state
    for i in range(6):
        axes[i].plot(t, state[i, :], label=state_labels[i])
        axes[i].set_ylabel(state_labels[i], fontsize=14)
        axes[i].tick_params(axis='both', labelsize=12)  
        axes[i].legend()
        axes[i].grid(True)

    # Labels for the states
    input_labels = [r'$f_1$ [N]', r'$f_2$ [N]']

    # Plot each state
    for i in range(2):
        axes[6].plot(t, input[i, :], label=input_labels[i])
    axes[6].set_ylabel(r'$\mathbf{u}$ [N]', fontsize=14)
    axes[6].tick_params(axis='both', labelsize=12)  
    axes[6].legend()
    axes[6].grid(True)

    # Common x-label
    axes[-1].set_xlabel('Time [s]', fontsize=14)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
