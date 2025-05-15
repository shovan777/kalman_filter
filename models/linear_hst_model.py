from dataclasses import dataclass
from .base_model import PhysicalModel
import numpy as np

Delta_t = 0.1  # hr

# HST model parameters
# TODO: find tau_h and v_hst
# Tau_h = 0.05  # hr
# tau_h: thermal time constant of the winding
# v_hst: temperature gradient 
Tau_h = 5/60  # hr
v_hst = 1  # also from ieee guide
# v_hst = 1 # also from ieee guide
denom = 1 / (Delta_t + Tau_h)
L1 = Delta_t * denom
L2 = Delta_t * v_hst * denom

# transition_mat = np.array([[1 - L1, L1], [0, 1]])
transition_mat = np.array([[1 - L1]])
input_mat = np.array([[L1, L2]])
# print(input_mat.shape)
# input_mat = np.array([[L2], [0]])
# print(f"transition_mat: {transition_mat}")


@dataclass
class LinearHSTModel(PhysicalModel):
    # transition_mat = transition_mat
    # measurement_mat = np.array([[1]])
    # input_mat = input_mat
    # output_mat = np.array([[0]])
    # process_error_cov_mat = np.eye(1) * 0.01
    # meas_noise_cov_mat = np.array([[0.01]])
    # process_noise_cov_mat = np.eye(1) * 0.01

    def __init__(self, **kwargs):
        super().__init__(
            transition_mat=transition_mat,
            measurement_mat=np.array([[1]]),
            input_mat=input_mat,
            output_mat=np.array([[0,0]]),
            process_error_cov_mat=np.eye(1) * 10,
            meas_noise_cov_mat=np.array([[0.01]]),
            process_noise_cov_mat=np.eye(1) * 0.01,
            **kwargs,
        )

    # check if shapes are correct
    # assert self.process_error_cov_mat.shape == self.transition_mat.shape
    def update_delta_t(self, Delta_t: float):
        denom = 1 / (Delta_t + Tau_h)
        L1 = Delta_t * denom
        L2 = Delta_t * v_hst * denom
        self.transition_mat = np.array([[1 - L1]])
        self.input_mat = np.array([[L1, L2]])

if __name__ == '__main__':
    linear_hst_model = LinearHSTModel(
    state_v = np.array([10]).reshape(-1, 1),
    input_v = np.array([0, 0]).reshape(-1, 1),
    measurement_v = np.array([0]).reshape(-1, 1),
    )

    print(linear_hst_model.transition_mat)