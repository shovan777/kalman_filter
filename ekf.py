import numpy as np

from basic_kf import KalmanFilter



class ExtendedKalmanFilter(KalmanFilter):
    """Extended Kalman Filter class."""
    def __init__(self, model, time_delta: float, init_vals: dict):
        """Initialize the Extended Kalman filter.

        Args:
            model (dict): dict containing the physical model.
            init_vals (dict): dict containing all the constants.
        """
        super().__init__(model, time_delta, init_vals)
            
    # use the calculated jacobians to estimate P error covariance 
    # instead of the transition matrix
    def estimate_error_cov(self, input_v: np.ndarray) -> np.ndarray:
        """Estimate the error covariance using the Extended Kalman Filter.

        Args:
            input_v (np.ndarray): The input vector.

        Returns:
            np.ndarray: The estimated error covariance.
        """
        F = self.model.calculate_jacobian(input_v)
        
        # Predict the error covariance
        P_priori = (
            F @ self.model.process_error_cov_mat @ F.T +
            self.model.process_noise_cov_mat
        )
        
        return P_priori
    
    def predict_step(self, input_v: np.ndarray):
        # prediction step
        # x = Fx + Bu
        self.model.state_v = self.predict_next_state(input_v)

        # z = Hx + Du
        self.model.measurement_v = self.predict_measurement(input_v)
        
        # P = FPF^T + Q
        estimated_process_noise = self.estimate_error_cov(input_v)
        self.model.process_error_cov_mat = estimated_process_noise
    
    