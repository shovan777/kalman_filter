import numpy as np

from .base_model import PhysicalModel


class SUSATOTModel(PhysicalModel):
    """
    A class representing the SUSATOT model for transformer aging.
    """

    def __init__(self, **kwargs):
        # three state variables as per luo et al. 2024
        # 1. top oil temperature
        # 2. oil exponent
        # 3. oil time constant
        super().__init__(
            transition_mat=np.eye(3),
            measurement_mat=np.array([[1, 0, 0]]),
            input_mat=np.array([[0], [0], [0]]),
            output_mat=np.array([[0]]),
            process_error_cov_mat=np.eye(3) * 10,
            meas_noise_cov_mat=np.array([[0.01]]),
            process_noise_cov_mat=np.eye(3) * 0.01,
            **kwargs,
        )

    def predict_next_state(self, input_v: np.ndarray) -> np.ndarray:
        """
        Predict the next state using the non linear state transition model.
        R: load ratio i.e load loss / no load loss
        K: load factor i.e current load / rated load
        delta_tot: rise in rated TOT temperature at rated load
        theta_amb: ambient temperature
        tau_oil: oil time constant
        eta_oil: oil exponent
        theta_tot: top oil temperature
        mu_oil: per unit oil visocosity
        Args:
            input_v (np.ndarray): The input vector.
        Returns:
            np.ndarray: The predicted next state.
        """
        # Predict the next top oil temperature using non linear SUSATOT model
        R = input_v[0]
        K = input_v[1]
        delta_t = input_v[2]  # TODO: supply this from somewhere
        delta_tot_rated = 50.0
        tau_oil = 360.0  # taken from top-oil paper Luo et al. 2024
        theta_amb = input_v[3]
        eta_oil = 1.0
        tot_rated = 50.0

        theta_c = 2797.0
        theta_base = 273.0
        mu = np.exp(theta_c / (self.state_v[0] + theta_base))
        mu_rated = np.exp(theta_c / (tot_rated + theta_base))
        mu_oil = mu / mu_rated
        load_contrib = (1 + R * (K**2)) / (1 + R)

        theta_diff_amb = self.state_v[0] - theta_amb
        # TODO: need to handle numerical unstability due theta_diff_amb
        # to negative value
        # if theta_diff_amb < 0:
        #     theta_diff_amb = 0
        self.state_v[0] += delta_t * (
            (load_contrib * (delta_tot_rated / tau_oil))
            - (
                ((theta_diff_amb) ** (eta_oil + 1))
                / (((delta_tot_rated * mu_oil) ** eta_oil) * tau_oil)
            )
        )
        self.state_v[1] = eta_oil
        self.state_v[2] = tau_oil
        return self.state_v

    def calculate_jacobian(self, input_v: np.ndarray) -> np.ndarray:
        """
        Calculate the Jacobian matrix for the state transition function.
        This is a placeholder and should be replaced with the actual calculation.
        """
        # Placeholder for the Jacobian matrix
        jacobian_mat = np.eye(3)

        # Predict the next top oil temperature using non linear SUSATOT model
        R = input_v[0]
        K = input_v[1]
        delta_t = input_v[2]  # TODO: supply this from somewhere
        delta_tot_rated = 50.0
        tau_oil = 360.0
        theta_amb = input_v[3]
        eta_oil = 1.0
        tot_rated = 50.0

        theta_c = 2797.0
        theta_base = 273.0
        mu = np.exp(theta_c / (self.state_v[0] + theta_base))
        mu_rated = np.exp(theta_c / (tot_rated + theta_base))
        mu_oil = mu / mu_rated

        load_contrib = (1 + R * (K**2)) / (1 + R)

        cur_theta = self.state_v[0]

        jacobian_mat = np.eye(3)
        theta_diff_amb = cur_theta - theta_amb
        # if theta_diff_amb < 0:
        #     theta_diff_amb = 0
        # check if theta_diff_amb is nan
        # print(f"cur_theta: {cur_theta}")
        # print(f"theta_diff_amb: {theta_diff_amb}")
        jacobian_mat[0, 0] = 1 - delta_t * theta_diff_amb**eta_oil * mu_rated * (
            (1 + eta_oil + theta_diff_amb * theta_c * eta_oil)
            / (
                (delta_tot_rated**eta_oil)
                * tau_oil
                * mu
                * (cur_theta + theta_base) ** 2
            )
        )
        # print(f"jacobian_mat[0, 0]: {jacobian_mat[0, 0]}")
        # TODO: implement when we assume eta and tau are not constant
        # TODO: need to handle numerical unstability due to log of negative value
        # for theta_diff_amb
        # jacobian_mat[0, 1] = (
        #     -delta_t
        #     * theta_diff_amb
        #     * (theta_diff_amb / (mu_oil * delta_tot_rated)) ** eta_oil
        #     * (
        #         np.log(theta_diff_amb / delta_tot_rated)
        #         - (
        #             theta_c / (self.state_v[0] + theta_base)
        #             + (theta_c / (tot_rated + theta_base))
        #         )
        #     )
        # 
        # jacobian_mat[0, 2] = (
        #     delta_t
        #     * (
        #         (load_contrib * (delta_tot_rated / tau_oil))
        #         - (
        #             (theta_diff_amb) ** (eta_oil + 1)
        #             / (((delta_tot_rated * mu_oil) ** eta_oil) * tau_oil)
        #         )
        #     )
        #     / tau_oil
        # )
        return jacobian_mat

    def predict_measurement(self, input_v: np.ndarray) -> np.ndarray:
        """Predict the measurement using the measurement matrix.

        Args:
            input_v (np.ndarray): The input vector.

        Returns:
            np.ndarray: The predicted measurement.
        """
        # Predict the measurement using the measurement matrix
        measurement_estimate = (
            self.measurement_mat @ self.state_v
        )
        return measurement_estimate