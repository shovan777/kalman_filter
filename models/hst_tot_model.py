import numpy as np
from .base_model import PhysicalModel


class SUSATOTHSTModel(PhysicalModel):
    """
    A class representing the SUSA model 
    for transformer thermal state estimation.

    This model is based on the work of Susa et al. (2005) and is used for
    estimating the thermal state of transformers. It includes two states
    for the top oil temperature and one state for the oil exponent.
    The model is designed to be used with a Kalman filter for state
    estimation and prediction.
    """

    def __init__(self, **kwargs):
        # tow state variables
        # 1. Hot spot temperature
        # 2. Top oil temperature
        super().__init__(
            transition_mat=np.eye(2),
            measurement_mat=np.array([[0, 1]]),
            input_mat=np.array([[0], [0]]),
            output_mat=np.array([[0], [0]]),
            process_error_cov_mat=np.eye(2) * 0.1,
            meas_noise_cov_mat=np.array([[0.01], [0.01]]),
            process_noise_cov_mat=np.eye(2) * 0.01,
            **kwargs,
        )

    def predict_next_state(self, input_v: np.ndarray) -> np.ndarray:
        """
        Predict the next state using the non-linear state transition model.
        The model is based on the work of Susa et al. (2005) and includes
        two states for the top oil temperature and one state for the oil exponent.

        Args:
            input_v (np.ndarray): The input vector.

        Returns:
            np.ndarray: The predicted next state.
        """
        # Predict the next top oil temperature using non-linear SUSA model
        # previous state
        prev_hst = self.state_v[0]
        prev_tot = self.state_v[1]

        # load parameters
        # for TOT model
        # R: load ratio i.e load loss / no load loss
        # K: load factor i.e current load / rated load
        # delta_t: time step in hours
        # theta_amb: ambient temperature
        # tau_oil: oil time constant
        # eta_oil: oil exponent
        # theta_tot: top oil temperature
        # mu_oil: per unit oil viscosity
        # delta_tot_rated: rise in rated TOT temperature at rated load
        # tot_rated: rated temperature rise
        # theta_c: constant for oil viscosity calculation
        # theta_base: base temperature for oil viscosity calculation
        # mu: oil viscosity at current temperature
        # mu_rated: oil viscosity at rated temperature
        # mu_oil: per unit oil viscosity
        # load_contrib: contribution of load to temperature rise
        # theta_diff_amb: difference between top oil temperature and ambient temperature
        R = input_v[0]
        K = input_v[1]
        delta_t = input_v[2]
        delta_tot_rated = 50.0
        tau_oil = 360.0 / 60.0 # hr
        theta_amb = input_v[3]
        eta_oil = 1.0
        tot_rated = 50.0

        theta_c = 2797.0
        theta_base = 273.0
        mu = np.exp(theta_c / (prev_tot + theta_base))
        mu_rated = np.exp(theta_c / (tot_rated + theta_base))
        mu_oil = mu / mu_rated
        load_contrib = (1 + R * (K**2)) / (1 + R)

        theta_diff_amb = prev_tot - theta_amb

        # predict the next state
        # Predict the next top oil temperature using non-linear SUSA model
        self.state_v[1] = prev_tot + delta_t * (
            (load_contrib * (delta_tot_rated / tau_oil))
            - (
                ((theta_diff_amb) ** (eta_oil + 1))
                / (((delta_tot_rated * mu_oil) ** eta_oil) * tau_oil)
            )
        )
        print(f"Predicted TOT: {self.state_v[1]}")
        # load parameters for HST model
        # tau_h: thermal time constant of the winding
        # v_hst: temperature gradient
        tau_h = 5 / 60 # her
        v_hst = 1
        denom = 1 / (delta_t + tau_h)
        L1 = delta_t * denom
        L2 = delta_t * v_hst * denom

        # predict the next hot spot temperature
        self.state_v[0] = (1 - L1) * prev_hst + L1 * prev_tot + (
            L2 * (K ** (2 * eta_oil))
        )

        # Calculate the Jacobian matrix
        # just update the transition matrix
        self.transition_mat[0, 0] = 1 - L1
        self.transition_mat[0, 1] = L1
        self.transition_mat[1, 0] = 0
        self.transition_mat[1, 1] = 1 - delta_t * theta_diff_amb**eta_oil * mu_rated * (
            (1 + eta_oil + theta_diff_amb * theta_c * eta_oil)
            / (
                (delta_tot_rated**eta_oil)
                * tau_oil
                * mu
                * (prev_tot + theta_base) ** 2
            )
        )
        return self.state_v
    
    def predict_measurement(self, input_v: np.ndarray) -> np.ndarray:
        """
        Predict the measurement using the measurement model.
        The measurement model is based on the work of Susa et al. (2005)
        and includes the top oil temperature.
        Args:
            input_v (np.ndarray): The input vector.
        Returns:
            np.ndarray: The predicted measurement.
        """
        measurement_estimate = (
            self.measurement_mat @ self.state_v
        )
        return measurement_estimate
    
