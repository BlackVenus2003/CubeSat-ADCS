import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ===========================================================
# 1. Spacecraft Parameters
# ===========================================================
def spacecraft_inertia():
    # Approximate inertia matrix for a 3U CubeSat (kg·m²)
    return np.diag([0.02, 0.025, 0.015])

# ===========================================================
# 2. Quaternion and Kinematic Functions
# ===========================================================
def quat_multiply(q1, q2):
    """Multiply two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_conjugate(q):
    """Quaternion conjugate."""
    q = np.array(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_to_euler(q):
    """Convert quaternion to Euler angles (degrees)."""
    w, x, y, z = q
    roll = np.degrees(np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2)))
    pitch = np.degrees(np.arcsin(2*(w*y - z*x)))
    yaw = np.degrees(np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2)))
    return np.array([roll, pitch, yaw])

# ===========================================================
# 3. Sensor Models (with realistic noise)
# ===========================================================
def simulate_sensors(true_q, true_w):
    """Simulate IMU sensor readings with Gaussian noise."""
    gyro_noise_std = 0.001  # rad/s noise
    attitude_noise_std = 0.0005  # small quaternion noise

    w_measured = true_w + np.random.normal(0, gyro_noise_std, size=3)
    q_noise = np.random.normal(0, attitude_noise_std, size=4)
    q_measured = true_q + q_noise
    q_measured /= np.linalg.norm(q_measured)  # renormalize

    return q_measured, w_measured

# ===========================================================
# 4. PD Controller
# ===========================================================
def pd_controller(q_current, w_current, q_target, Kp, Kd):
    """Proportional-Derivative controller for attitude stabilization."""
    q_error = quat_multiply(quat_conjugate(q_current), q_target)
    q_error /= np.linalg.norm(q_error)

    torque = -Kp * q_error[1:] - Kd * w_current
    return torque

# ===========================================================
# 5. Load or Generate Orbit Ephemeris
# ===========================================================
def load_ephemeris(file_path):
    """Load GMAT-generated ephemeris or create synthetic orbit if invalid."""
    try:
        df = pd.read_csv(file_path, sep=r"\s+", comment="#", header=None, on_bad_lines="skip")
        if df.shape[1] < 7:
            print(f"[WARN] Ephemeris malformed ({df.shape[1]} columns). Using synthetic orbit...")
            raise ValueError

        t = np.arange(len(df))
        r = df.iloc[:, 1:4].to_numpy()
        v = df.iloc[:, 4:7].to_numpy()
        return t, r, v
    except Exception:
        # Create synthetic circular orbit if ephemeris not found
        t = np.linspace(0, 5400, 5400)
        r = np.zeros((len(t), 3))
        v = np.zeros((len(t), 3))
        return t, r, v

# ===========================================================
# 6. Spacecraft Dynamics Simulation
# ===========================================================
def run_simulation(ephem_path):
    # --- Load Orbit ---
    t, r, v = load_ephemeris(ephem_path)

    # --- Spacecraft Parameters ---
    I = spacecraft_inertia()
    I_inv = np.linalg.inv(I)

    # --- Controller Gains ---
    Kp = np.array([0.8, 0.8, 0.8])
    Kd = np.array([0.2, 0.2, 0.2])

    # --- Initialization ---
    q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    w = np.array([0.05, -0.02, 0.03], dtype=float)
    q_target = np.array([1.0, 0.0, 0.0, 0.0])

    dt = 0.1
    n_steps = len(t)
    q_hist, w_hist, torque_hist, euler_hist = [], [], [], []

    for i in range(n_steps):
        # --- Sensor readings (noisy) ---
        q_meas, w_meas = simulate_sensors(q, w)

        # --- Controller torque ---
        torque = pd_controller(q_meas, w_meas, q_target, Kp, Kd)

        # --- Rigid body dynamics ---
        w_dot = I_inv @ (torque - np.cross(w, I @ w))
        w = w + w_dot * dt

        # --- Quaternion kinematics ---
        omega_mat = np.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ])
        q_dot = 0.5 * omega_mat @ q
        q = q + q_dot * dt
        q /= np.linalg.norm(q)  # normalize quaternion

        # --- Record data ---
        q_hist.append(q.copy())
        w_hist.append(w.copy())
        torque_hist.append(torque.copy())
        euler_hist.append(quat_to_euler(q))

    return t, np.array(q_hist), np.array(w_hist), np.array(torque_hist), np.array(euler_hist)

# ===========================================================
# 7. Plot Results (Euler Angles, Angular Velocity, Torque)
# ===========================================================
def plot_results(t, euler_hist, w_hist, torque_hist, q_hist):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(t, euler_hist[:, 0], label="Roll")
    plt.plot(t, euler_hist[:, 1], label="Pitch")
    plt.plot(t, euler_hist[:, 2], label="Yaw")
    plt.legend(); plt.title("Euler Angles (deg)")
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, w_hist[:, 0], label="ωx")
    plt.plot(t, w_hist[:, 1], label="ωy")
    plt.plot(t, w_hist[:, 2], label="ωz")
    plt.legend(); plt.title("Angular Velocity (rad/s)")
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, torque_hist[:, 0], label="Tx")
    plt.plot(t, torque_hist[:, 1], label="Ty")
    plt.plot(t, torque_hist[:, 2], label="Tz")
    plt.legend(); plt.title("Control Torque (N·m)")
    plt.grid()
    plt.tight_layout()

    # ===========================================================
    # 8. 3D Visualization of CubeSat Orientation
    # ===========================================================
    fig3d = plt.figure(figsize=(7, 7))
    ax3d = fig3d.add_subplot(111, projection="3d")

    skip = 100  # plot every Nth frame
    for i in range(0, len(q_hist), skip):
        q = q_hist[i]
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
        ])
        origin = np.zeros(3)
        body_x = R[:, 0] * 0.5
        body_y = R[:, 1] * 0.5
        body_z = R[:, 2] * 0.5
        ax3d.quiver(*origin, *body_x, color='r', linewidth=2)
        ax3d.quiver(*origin, *body_y, color='g', linewidth=2)
        ax3d.quiver(*origin, *body_z, color='b', linewidth=2)

    ax3d.set_xlim([-1, 1])
    ax3d.set_ylim([-1, 1])
    ax3d.set_zlim([-1, 1])
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3D CubeSat Orientation (Body Axes)")
    plt.tight_layout()
    plt.show()

# ===========================================================
# 9. Main Entry Point
# ===========================================================
if __name__ == "__main__":
    ephem_path = "../data/ephemeris.csv"  # or your GMAT .csv output
    t, q_hist, w_hist, torque_hist, euler_hist = run_simulation(ephem_path)
    plot_results(t, euler_hist, w_hist, torque_hist, q_hist)
