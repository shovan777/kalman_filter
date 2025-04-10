"""Module to create the basic structure of kalman filter."""
import numpy as np
from models.base_model import PhysicalModel
    

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
    def __init__(self, model: PhysicalModel, time_delta:float, init_vals: dict):
        """Initialize the Kalman filter.
        Load all the constants with values and also
        initialize the covariance matrices.

        Args:
            model (dict): dict containing the physical model.
            init_vals (dict): dict containing all the constants.
        """
        self.model = model
        self.delta_t = time_delta
        # TODO: remove initialization to model itself
        # not necessary for KF to know about the model
        # for key in self.model.constants.keys:
        #     try:
        #         self.model[key] = init_vals[key]
        #     except KeyError as ke:
        #         raise "Initializer should have all model constants."
    
    def predict_next_state(self, input_v: np.ndarray) -> np.ndarray:
        next_state_priori = (
            self.model.transition_mat @ self.model.state_v + 
            self.model.input_mat @ input_v
        )
        return next_state_priori

    def predict_measurement(self, input_v: np.ndarray) -> np.ndarray:
        # most of the time output_mat will be zero
        # because the measurement is not dependent on the input
        # in general
        measurement_estimate = (
            self.model.measurement_mat @ self.model.state_v +
            self.model.output_mat @ input_v
        )
        return measurement_estimate

    def estimate_error_cov(self):
        return (
            self.model.transition_mat @ self.model.process_error_cov_mat @ self.model.transition_mat.T +
            self.model.process_noise_cov_mat
        )

    def calculate_gain(self):
        return (
            (self.model.process_error_cov_mat @ self.model.measurement_mat.T) @
            np.linalg.inv(
                ((self.model.measurement_mat @ self.model.process_error_cov_mat) @
                self.model.measurement_mat.T) + self.model.meas_noise_cov_mat
            )
        )
    
    def calculate_innovation(self, actual_measurement: np.ndarray) -> np.ndarray:
        return actual_measurement - self.model.measurement_v
    
    def update_next_state(
            self,
            innovation_v: np.ndarray, 
            next_state_priori_v: np.ndarray,
            kalman_gain: np.ndarray
            ):
        self.model.state_v = next_state_priori_v + kalman_gain @ innovation_v

    def updade_err_cov_mat(self, estimated_error_cov: np.ndarray, kalman_gain: np.ndarray):
        self.model.process_error_cov_mat = (np.eye(len(self.model.state_v)) - kalman_gain @ self.model.measurement_mat) @ estimated_error_cov

    def predict_step(self, input_v: np.ndarray):
        # prediction step
        # x = Fx + Bu
        self.model.state_v = self.predict_next_state(input_v)

        # z = Hx + Du
        self.model.measurement_v = self.predict_measurement(input_v)
        
        # P = FPF^T + Q
        estimated_process_noise = self.estimate_error_cov()
        self.model.process_error_cov_mat = estimated_process_noise

    def update_step(self, actual_measurement: np.ndarray):
        # update step
        # K = PH^T(HPH^T + R)^-1
        kalman_gain = self.calculate_gain()

        # innovation/error
        # y_err = z - Hx
        innovation_v = self.calculate_innovation(actual_measurement)

        # x = x + K*y_err
        self.update_next_state(innovation_v, self.model.state_v, kalman_gain)

        # P = (I - KH)P
        self.updade_err_cov_mat(self.model.process_error_cov_mat, kalman_gain)
    

    def kalman_step(self, input_v: np.ndarray, actual_measurement: np.ndarray):
        # prediction step
        # x = Fx + Bu
        self.model.state_v = self.predict_next_state(input_v)

        # z = Hx + Du
        self.model.measurement_v = self.predict_measurement(input_v)
        
        # P = FPF^T + Q
        estimated_process_noise = self.estimate_error_cov()
        self.model.process_error_cov_mat = estimated_process_noise

        # update step
        # K = PH^T(HPH^T + R)^-1
        kalman_gain = self.calculate_gain()

        # innovation/error
        # y_err = z - Hx
        innovation_v = self.calculate_innovation(actual_measurement)

        # x = x + K*y_err
        self.update_next_state(innovation_v, self.model.state_v, kalman_gain)

        # P = (I - KH)P
        self.updade_err_cov_mat(estimated_process_noise, kalman_gain)


class Runner:
    """Runner class to run the kalman filter."""
    def __init__(self, kf: KalmanFilter, delta_t: float):
        self.delta_t = delta_t
        self.kf = kf

    def run_kalman(self, input_v: np.ndarray, actual_measurement: np.ndarray):
        self.kf.kalman_step(input_v, actual_measurement)
        return self.model.state_v
