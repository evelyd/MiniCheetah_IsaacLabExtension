import numpy as np

import matplotlib.pyplot as plt

# Load the .npy file
file_path = "../logs/rsl_rl/mini_cheetah_flat/2025-03-26_21-28-50/obs_action_pairs.npy"  # Replace with the actual file path
data = np.load(file_path, allow_pickle=True)

obs_data = np.array([traj['obs'] for traj in data])

# Validate the shape of the data
if obs_data.shape[2] != 53:
    raise ValueError(f"Expected obs_data with 53 features, but got {obs_data.shape[2]}")

# Extract the components
base_lin_vel = obs_data[:, :, 0:3]
base_ang_vel = obs_data[:, :, 3:6]
projected_gravity = obs_data[:, :, 6:9]
velocity_commands = obs_data[:, :, 9:12]
joint_pos = obs_data[:, :, 12:24]
joint_vel = obs_data[:, :, 24:36]
actions = obs_data[:, :, 36:48]
base_z = obs_data[:, :, 48]
base_quat = obs_data[:, :, 49:53]

# Plotting function
def plot_component(component, title, labels):
    plt.figure(figsize=(10, 6))
    for i in range(component.shape[1]):
        plt.plot(component[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()

# Plot each component
random_index = np.random.randint(0, 4096)
plot_component(base_lin_vel[:, random_index, :], "Base Linear Velocity", ["x", "y", "z"])
plot_component(base_ang_vel[:, random_index, :], "Base Angular Velocity", ["x", "y", "z"])
plot_component(projected_gravity[:, random_index, :], "Projected Gravity", ["x", "y", "z"])
plot_component(velocity_commands[:, random_index, :], "Velocity Commands", ["x", "y", "z"])
plot_component(joint_pos[:, random_index, :], "Joint Positions", [f"Joint {i+1}" for i in range(12)])
plot_component(joint_vel[:, random_index, :], "Joint Velocities", [f"Joint {i+1}" for i in range(12)])
plot_component(actions[:, random_index, :], "Actions", [f"Action {i+1}" for i in range(12)])

# Plot base_z
plt.figure(figsize=(10, 6))
plt.plot(base_z[:, random_index], label="Base Z")
plt.title("Base Z")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid()
plt.show()

# Plot base_quat
plot_component(base_quat[:, random_index, :], "Base Quaternion (wxyz)", ["w", "x", "y", "z"])