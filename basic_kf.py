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
    process_cov_mat: np.ndarray # process noise covariance matrix, P
    meas_cov_mat: np.ndarray # measurement noise covariance matrix, Q
    

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
    def __init__(model: PhysicalModel, init_vals: dict):
        """Initialize the Kalman filter.
        Load all the constants with values and also
        initialize the covariance matrices.

        Args:
            model (dict): dict containing the physical model.
            init_vals (dict): dict containing all the constants.
        """
        for key in model.constants.keys:
            try:
                model[key] = init_vals[key]
            except KeyError as ke:
                raise "Initilizer should have all model constants."
    
    def predict_next_state(model: PhysicalModel, input_v: np.ndarray) -> np.ndarray:
        next_state_priori_v = (
            model.transition_mat @ model.state_v + 
            model.input_mat @ model.input_v
        )
        return next_state_priori_v

    def predict_measurement(model: PhysicalModel, input_v: np.ndarray) -> np.ndarray:
        measurement_estimate = (
            model.measurement_mat @ model.state_v +
            model.output_mat @ model.input_v
        )
        return measurement_estimate

    def estimate_process_noise(model: PhysicalModel):
        return (
            model.transition_mat @ model.process_cov_mat @ model.transition_mat.T +
            model.meas_cov_mat
        )

    def calculate_gain(model: PhysicalModel):
        return (
            (model.process_cov_mat @ model.measurement_mat.T) @
            np.linalg.inv(
                ((model.measurement_mat @ model.process_cov_mat) @
                model.measurement_mat.T) + model.meas_noise_mat
            )
        )
    
    def calculate_innovation(model: PhysicalModel, measurement_estimate):
        return model.measurement_v - measurement_estimate
    
    def update_next_state(
            model: PhysicalModel, innovation_v: np.ndarray, 
            next_state_priori_v: np.ndarray,
            kalman_gain: np.ndarray
            ):
        model.state_v = next_state_priori_v + kalman_gain @ innovation_v

    def update_cov_mat(model: PhysicalModel, estimated_process_noise: np.ndarray, kalman_gain: np.ndarray):
        model.process_cov_mat = (np.eye(len(model.state_v)) - kalman_gain @ model.meas_cov_mat) @ estimated_process_noise