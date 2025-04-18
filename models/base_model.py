from dataclasses import dataclass, field
from collections import OrderedDict
import numpy as np

@dataclass
class PhysicalModel:
    state_v: np.ndarray # state vector, X
    input_v: np.ndarray # model inputs or input excitations
    measurement_v: np.ndarray # sensor measurement vector, Z
    transition_mat: np.ndarray # state transition matrix F
    measurement_mat: np.ndarray # measurement matrix or observation matrix H
    input_mat: np.ndarray # input matrix B
    output_mat: np.ndarray # output matrix D if inputs influence measurements
    process_error_cov_mat: np.ndarray # process error covariance matrix, P
    meas_noise_cov_mat: np.ndarray # measurement noise covariance matrix, R
    process_noise_cov_mat: np.ndarray # process noise covariance matrix, Q
    # variables with default values
    constants: dict = field(default_factory=OrderedDict)# contains all the constants
    state: dict = field(default_factory=OrderedDict)# contains all the state variables, X
    input: dict = field(default_factory=OrderedDict)# contains  all the input variables, U
    measurement: dict = field(default_factory=OrderedDict) # contains all output variables
    # also known as measurements Z or Y
    # this also represents the order of the state variables

    def predict_next_state(self, input_v: np.ndarray) -> np.ndarray:
        """Predict the next state using the state transition matrix.

        Args:
            input_v (np.ndarray): The input vector.

        Returns:
            np.ndarray: The predicted next state.
        """
        # Predict the next state using the state transition matrix
        next_state_priori = (
            self.transition_mat @ self.state_v + 
            self.input_mat @ input_v
        )
        return next_state_priori
    
    def predict_measurement(self, input_v: np.ndarray) -> np.ndarray:
        """Predict the measurement using the measurement matrix.

        Args:
            input_v (np.ndarray): The input vector.

        Returns:
            np.ndarray: The predicted measurement.
        """
        # Predict the measurement using the measurement matrix
        measurement_estimate = (
            self.measurement_mat @ self.state_v +
            self.output_mat @ input_v
        )
        return measurement_estimate