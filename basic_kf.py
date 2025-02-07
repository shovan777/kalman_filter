"""Module to create the basic structure of kalman filter."""
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict

@dataclass
class PhysicalModel():
    constants: dict = {}
    state: dict = OrderedDict() # contains all the state variables, X
    input: dict = OrderedDict() # contains  all the input variables, U
    measurement: dict = OrderedDict() # contains all output variables
    # also known as measurements Z or Y
    # this also represents the order of the state variables
    state_v: np.ndarray # state vector, X
    input_v: np.ndarray # model inputs or input excitations
    measurement_v: np.ndarray # sensor measurement vector, Z
    transition_mat: np.ndarray # state transition matrix F
    measurement_mat: np.ndarray # measurement matrix or observation matrix H
    input_mat: np.ndarray # input matrix B
    output_mat: np.ndarray # output matrix D if inputs influence measurements
    process_error_cov_mat: np.ndarray # process error covariance matrix, P
    meas_noise_cov_mat: np.ndarray # measurement noise covariance matrix, Q
    process_noise_cov_mat: np.ndarray # process noise covariance matrix, R
    

def zero_initializer(mat_size: tuple) -> np.ndarray:
    """Initialize a matrix with zeros.

    Args:
        mat_size (tuple): size of the zeroth matrix

    Returns:
        np.ndarray: ndarray of size mat_size
    """
    return np.zeros(mat_size)

def identity_initializer(row_size: int) -> np.ndarray:
    """Initialize an identity matrix.

    Args:
        row_size (int): size of the row/ column of matrix

    Returns:
        np.ndarray: identity matrix of given size
    """    
    return np.ones(row_size)


class KalmanFilter:
    def __init__(self, model: PhysicalModel, init_vals: dict):
        """Initialize the Kalman filter.
        Load all the constants with values and also
        initialize the covariance matrices.

        Args:
            model (dict): dict containing the physical model.
            init_vals (dict): dict containing all the constants.
        """
        self.model = model
        self.delta_t = 0.01
        for key in self.model.constants.keys:
            try:
                self.model[key] = init_vals[key]
            except KeyError as ke:
                raise "Initializer should have all model constants."
    
    def predict_next_state(self, input_v: np.ndarray) -> np.ndarray:
        next_state_priori_v = (
            self.model.transition_mat @ self.model.state_v + 
            self.model.input_mat @ self.model.input_v
        )
        return next_state_priori_v

    def predict_measurement(self, input_v: np.ndarray) -> np.ndarray:
        # most of the time output_mat will be zero
        # because the measurement is not dependent on the input
        # in general
        measurement_estimate = (
            self.model.measurement_mat @ self.model.state_v +
            self.model.output_mat @ self.model.input_v
        )
        return measurement_estimate

    def estimate_process_error(self):
        return (
            self.model.transition_mat @ self.model.process_error_cov_mat @ self.model.transition_mat.T +
            self.model.meas_noise_cov_mat
        )

    def calculate_gain(self):
        return (
            (self.model.process_error_cov_mat @ self.model.measurement_mat.T) @
            np.linalg.inv(
                ((self.model.measurement_mat @ self.model.process_error_cov_mat) @
                self.model.measurement_mat.T) + self.model.process_noise_cov_mat
            )
        )
    
    def calculate_innovation(self, measurement_estimate):
        return self.model.measurement_v - measurement_estimate
    
    @staticmethod
    def update_next_state(
            self,
            innovation_v: np.ndarray, 
            next_state_priori_v: np.ndarray,
            kalman_gain: np.ndarray
            ):
        self.model.state_v = next_state_priori_v + kalman_gain @ innovation_v

    def updade_err_cov_mat(self, estimated_process_noise: np.ndarray, kalman_gain: np.ndarray):
        self.model.process_error_cov_mat = (np.eye(len(self.model.state_v)) - kalman_gain @ self.model.meas_noise_cov_mat) @ estimated_process_noise

    def kalman_step(self, input_v: np.ndarray):
        # prediction step
        # x = Fx + Bu
        next_state_priori_v = self.predict_next_state(input_v)

        # z = Hx + Du
        measurement_estimate = self.predict_measurement(input_v)
        
        # P = FPF^T + Q
        estimated_process_noise = self.estimate_process_error()

        # update step
        # K = PH^T(HPH^T + R)^-1
        kalman_gain = self.calculate_gain()
        # y_err = z - Hx
        innovation_v = self.calculate_innovation(measurement_estimate)
        # x = x + K*y_err
        self.update_next_state(innovation_v, next_state_priori_v, kalman_gain)
        # P = (I - KH)P
        self.updade_err_cov_mat(estimated_process_noise, kalman_gain)