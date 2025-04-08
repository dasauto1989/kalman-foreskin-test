import numpy as np
import matplotlib.pyplot as plt

dt = 1.0  # Time step

# Initial state: [pos_x, pos_y, vel_x, vel_y]
x = np.array([[0], [0], [1], [0.5]])

# Initial covariance
P = np.eye(4) * 500

# State transition model (constant velocity)
F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0 ],
    [0, 0, 0, 1 ]
])

# Measurement model (we only measure position)
H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0]
])

# Measurement noise
R = np.eye(2) * 10

# Process noise
Q = np.eye(4) * 0.1

# Identity matrix
I = np.eye(4)

# Simulation variables
true_pos = np.array([[0.0], [0.0]])
true_vel = np.array([[1.0], [0.5]])

measured_positions = []
estimated_positions = []
true_positions = []

for _ in range(30):
    # Simulate true position update
    true_pos += true_vel * dt
    true_positions.append(true_pos.flatten())

    # Simulated noisy measurement
    z = true_pos + np.random.normal(0, 3, (2, 1))
    measured_positions.append(z.flatten())

    # ----- PREDICTION -----
    x = F @ x
    P = F @ P @ F.T + Q

    # ----- UPDATE -----
    y = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)
    x = x + K @ y
    P = (I - K @ H) @ P

    estimated_positions.append(x[:2].flatten())

# ----- Plotting -----
true_positions = np.array(true_positions)
measured_positions = np.array(measured_positions)
estimated_positions = np.array(estimated_positions)

plt.plot(true_positions[:,0], true_positions[:,1], label='True Path')
plt.scatter(measured_positions[:,0], measured_positions[:,1], label='Measurements', alpha=0.6)
plt.plot(estimated_positions[:,0], estimated_positions[:,1], label='Kalman Estimate', linestyle='--')
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Kalman Filter (Constant Velocity)')
plt.grid(True)
plt.axis('equal')
plt.show()
