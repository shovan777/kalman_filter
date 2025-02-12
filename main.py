"""Module running the basic kalman filter."""
import numpy as np
import matplotlib.pyplot as plt
from basic_kf import PhysicalModel, KalmanFilter
from copy import deepcopy

Delta_t = 1

# create the physical motion_model
motion_model = PhysicalModel(
    state_v=np.array([0, 0]).reshape(-1, 1),
    input_v=np.array([0]).reshape(-1, 1),
    measurement_v=np.array([0]).reshape(-1, 1),
    transition_mat=np.array([[1, Delta_t],
                             [0, 1]]),
    measurement_mat=np.array([[1, 0]]),
    input_mat=np.array([[0.5 * Delta_t ** 2],
                        [Delta_t]]),
    output_mat=np.array([[0]]),
    # process_error_cov_mat=np.array([[0.5 * Delta_t ** 4, 0],
    #                                [0, Delta_t ** 2]]),
    process_error_cov_mat=np.eye(2) * 0.01,
    meas_noise_cov_mat=np.array([[0.01]]),
    process_noise_cov_mat=np.eye(2) * 0.01
)

# lets say we have a simple linear motion model
# x = x + vt
# then state variables are
# x, v
motion_model.state = {
    'x': 0,
    'v': 0
}

# initialize the motion_model
motion_model.constants = {
    'delta_t': Delta_t,
    'process_noise_cov': 0.01,
    'meas_noise_cov': 0.01
}

# define the motion_model matrices
# # state transition matrix
# motion_model.transition_mat = np.array([[1, motion_model.constants['delta_t']],
#                                   [0, 1]])

# measurement matrix
# motion_model.measurement_mat = np.array([[1, 0]])

# input matrix
# motion_model.input_mat = np.array([[0.5 * motion_model.constants['delta_t'] ** 2],
#                              [motion_model.constants['delta_t']]])

# process noise covariance matrix
# motion_model.process_noise_cov_mat = np.array([[0.5 * motion_model.constants['delta_t'] ** 4, 0],
#                                         [0, motion_model.constants['delta_t'] ** 2]])

# measurement noise covariance matrix
# motion_model.meas_noise_cov_mat = np.array([[motion_model.constants['meas_noise_cov']]])
# make a deep copy of the model for KF
motion_model_kf = deepcopy(motion_model)

# initialize the kalman filter
kf = KalmanFilter(motion_model, Delta_t ,motion_model.constants)

# lets say we have a sensor that measures the position
# of the object
# Generation dummy input acceleration input data
# acceleration_inp = np.random.normal(0, 0.1, 10)
# constant acceleration
acceleration_inp = np.array([2 for i in range(10)])
# If object is at rest, then the velocity is zero and starting point is 0
# lets calculate the measurements 
# measurements = [lambda x: x + 0.5 * Delta_t ** 2 * acceleration_inp for i in range(len(acceleration_inp))]
measurements = []

for i in range(len(acceleration_inp)):
    motion_model.state_v = kf.predict_next_state(acceleration_inp[i].reshape(-1, 1))
    measurement_estimate = kf.predict_measurement(acceleration_inp[i].reshape(-1, 1))
    measurements.append(measurement_estimate)


# lets add some noise to the measurement
measurements = [measurement + np.random.normal(0, 5) for measurement in measurements]

# initialize new KF
kf = KalmanFilter(motion_model_kf, Delta_t, motion_model_kf.constants)

estimated_positions = []
# # now lets run the kalman filter
for i in range(len(acceleration_inp)):
    # run one kalman step
    kf.kalman_step(acceleration_inp[i].reshape(-1, 1), measurements[i])
    # get the estimated position
    estimated_positions.append(kf.model.state_v[0])

# plot the measurements
# plot the position measurements
plt.plot(estimated_positions, label='Estimated position')
plt.plot(np.squeeze(measurements), label='Actual position')
plt.xlabel('Time')
plt.ylabel('Position')
plt.legend()
plt.show()