"""Module running the basic kalman filter."""

import numpy as np
import matplotlib.pyplot as plt
from basic_kf import PhysicalModel, KalmanFilter
from copy import deepcopy
from filterpy.kalman import KalmanFilter as Filter

# This is a simple example of a kalman filter
Delta_t = 0.1

# create the physical motion_model
motion_model = PhysicalModel(
    state_v=np.array([0, 0]).reshape(-1, 1),
    input_v=np.array([0]).reshape(-1, 1),
    measurement_v=np.array([0]).reshape(-1, 1),
    transition_mat=np.array([[1, Delta_t], [0, 1]]),
    measurement_mat=np.array([[1, 0]]),
    # input_mat=np.array([[0.5 * Delta_t**2], [Delta_t]]),
    input_mat=np.array([[0.0], [Delta_t]]),
    output_mat=np.array([[0]]),
    # process_error_cov_mat=np.array([[0.5 * Delta_t ** 4, 0],
    #                                [0, Delta_t ** 2]]),
    process_error_cov_mat=np.eye(2) * 0.01,
    meas_noise_cov_mat=np.array([[0.01]]),
    process_noise_cov_mat=np.eye(2) * 0.01,
)

# lets say we have a simple linear motion model
# x = x + vt
# then state variables are
# x, v
motion_model.state = {"x": 0, "v": 0}

# initialize the motion_model
motion_model.constants = {
    "delta_t": Delta_t,
    "process_noise_cov": 0.01,
    "meas_noise_cov": 0.01,
}

# create a deep copy of the motion model
motion_model_kf = deepcopy(motion_model)

# initialize the kalman filter
kf = KalmanFilter(motion_model, Delta_t, motion_model.constants)

# lets say we have a sensor that measures the position
# of the object
acceleration_inp = np.array([2 for i in range(int(10 / Delta_t))])
measurements = []

for i in range(len(acceleration_inp)):
    motion_model.state_v = kf.predict_next_state(acceleration_inp[i].reshape(-1, 1))
    measurement_estimate = kf.predict_measurement(acceleration_inp[i].reshape(-1, 1))
    measurements.append(measurement_estimate)


# lets add some noise to the measurement
noisy_measurements = [
    measurement + np.random.normal(0, 10) for measurement in measurements
]

# initialize new KF
kf = KalmanFilter(motion_model_kf, Delta_t, motion_model_kf.constants)

estimated_positions = []
estimated_error = []
# # now lets run the kalman filter
for i in range(len(acceleration_inp)):
    # run one kalman step
    kf.kalman_step(acceleration_inp[i].reshape(-1, 1), noisy_measurements[i])
    # get the estimated position
    estimated_positions.append(kf.model.state_v[0])
    # calculate the confidence interval
    estimated_error.append(6 * np.sqrt(kf.model.process_error_cov_mat[0, 0])) 

# I want to compare my implementation of kalman filter with the one from
# pykalman
kf_pykalman = Filter(dim_x=2, dim_z=1)
kf_pykalman.x = np.array([[0], [0]])
kf_pykalman.P = np.eye(2) * 0.01
kf_pykalman.F = np.array([[1, Delta_t], [0, 1]])
kf_pykalman.H = np.array([[1, 0]])
kf_pykalman.Q = np.eye(2) * 0.01
kf_pykalman.R = np.array([[0.01]])
# kf_pykalman.B = np.array([[0.5 * Delta_t**2], [Delta_t]])
kf_pykalman.B = np.array([[0.0], [Delta_t]])
# kf_pykalman.u = np.array([[0]])

estimated_positions_pykalman = []
estimated_error_pykalman = []
# # now lets run the kalman filter
# # with pykalman
for i in range(len(acceleration_inp)):
    # run one kalman step
    kf_pykalman.predict(acceleration_inp[i].reshape(-1, 1))
    kf_pykalman.update(noisy_measurements[i])
    # get the estimated position
    estimated_positions_pykalman.append(kf_pykalman.x[0])
    # calculate the confidence interval
    estimated_error_pykalman.append(6 * np.sqrt(kf_pykalman.P[0, 0]))

# calculate the error between my kalman and pykalman
# get the rms error between the two estimated positions
estimated_positions = np.array(estimated_positions)
estimated_positions_pykalman = np.array(estimated_positions_pykalman)
rms_error = np.sqrt(
    np.mean((estimated_positions - estimated_positions_pykalman) ** 2)
)
print(f"RMS error: {rms_error}")

error = np.squeeze(estimated_positions) - np.squeeze(measurements)

## bring error to unit norm
# error = (error - np.mean(error)) / np.std(error)
error = error / error.max()
print(f"Error: {error.max()}")
print(f"Confidence max: {np.max(estimated_error)}")

# plot the estimated positions
plt.plot(estimated_positions, label="Estimated position")
plt.plot(
    estimated_positions_pykalman, label="Estimated position pykalman", linestyle="--"
)
# scatter plot the measurements in red
plt.scatter(
    range(len(noisy_measurements)),
    noisy_measurements,
    label="Measured position",
    color="red",
)
# plot the actual position
plt.plot(np.squeeze(measurements), label="Actual position")
plt.xlabel("Time")
plt.ylabel("Position")
plt.legend()
plt.show()

# # plot the error
# plt.scatter(range(len(error)), error, label="Error")

# plt.fill_between(
#     range(len(estimated_positions)),
#     -np.squeeze(estimated_error),
#     np.squeeze(estimated_error),
#     color="orange",
#     alpha=0.5,
# )
# plt.show()
