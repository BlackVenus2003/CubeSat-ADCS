import numpy as np
import matplotlib.pyplot as plt

# ==============================================================
# 1️⃣  Function to Load GMAT Ephemeris (.e) File
# ==============================================================

def load_ephemeris(filename):
    """Reads a GMAT/STK .e ephemeris file and extracts time, position, velocity."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find data section
    data_start = next(i for i, l in enumerate(lines) if "EphemerisTimePosVel" in l) + 1
    data_end = next(i for i, l in enumerate(lines) if "END Ephemeris" in l)

    # Extract numeric data
    raw = [l.strip().split() for l in lines[data_start:data_end]]
    data = np.array(raw, dtype=float)

    # Columns: [time, x, y, z, vx, vy, vz]
    t = data[:, 0]                           # seconds since epoch
    r = data[:, 1:4] * 1000                 # km → meters
    v = data[:, 4:7] * 1000
    return t, r, v


# ==============================================================
# 2️⃣  Quaternion & Dynamics Utilities
# ==============================================================

def quat_dot(q, w):
    """Compute quaternion derivative from angular velocity."""
    q0, q1, q2, q3 = q
    w1, w2, w3 = w
    return 0.5 * np.array([
        - (q1*w1 + q2*w2 + q3*w3),
         q0*w1 + q2*w3 - q3*w2,
         q0*w2 - q1*w3 + q3*w1,
         q0*w3 + q1*w2 - q2*w1
    ])

def normalize(q):
    """Normalize quaternion."""
    return q / np.linalg.norm(q)


# ==============================================================
# 3️⃣  ADCS Simulation (PD Control)
# ==============================================================

def simulate_adcs(t, Kp=0.05, Kd=0.1, dt=0.1):
    """Simulate CubeSat attitude stabilization with PD control."""
    I = np.diag([0.02, 0.02, 0.03])    # kg·m² inertia tensor
    I_inv = np.linalg.inv(I)

    # Initial conditions
    q = np.array([1, 0.1, 0.1, 0.05])  # small initial rotation
    q = normalize(q)
    w = np.array([0.05, 0.02, -0.03])  # initial angular velocity (rad/s)

    qs, ws = [], []

    for _ in range(len(t)):
        q_err = q[1:]                       # assume desired q = [1,0,0,0]
        torque = -Kp * q_err - Kd * w       # PD law

        # Angular dynamics
        w_dot = I_inv @ (torque - np.cross(w, I @ w))
        w += w_dot * dt

        # Quaternion kinematics
        q_dot = quat_dot(q, w)
        q += q_dot * dt
        q = normalize(q)

        qs.append(q.copy())
        ws.append(w.copy())

    return np.array(qs), np.array(ws)


# ==============================================================
# 4️⃣  Main Script — Run Orbit Load + Attitude Simulation
# ==============================================================

if __name__ == "__main__":
    # --- Load orbit data from GMAT ---
    t, r, v = load_ephemeris('../data/CubeSat.e')
    print(f"Loaded {len(t)} GMAT ephemeris points.")

    # --- Plot Orbit in 3D ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(r[:,0], r[:,1], r[:,2], color='b')
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('CubeSat Orbit (from GMAT)')
    plt.show()

    # --- Run Attitude Simulation ---
    qs, ws = simulate_adcs(t)

    # --- Plot Angular Velocity ---
    plt.figure()
    plt.plot(t, ws)
    plt.xlabel('Time (s)')
    plt.ylabel('Angular Velocity (rad/s)')
    plt.title('Angular Velocity vs Time')
    plt.legend(['ωx', 'ωy', 'ωz'])
    plt.grid(True)

    # --- Plot Quaternion Scalar Component ---
    plt.figure()
    plt.plot(t, qs[:,0])
    plt.xlabel('Time (s)')
    plt.ylabel('q₀')
    plt.title('Quaternion Scalar Component (Attitude Stabilization)')
    plt.grid(True)
    plt.show()

    print("Simulation complete — ADCS stabilization visualized.")
