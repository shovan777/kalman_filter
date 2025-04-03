import numpy as np
from matplotlib import pyplot as plt
from typing import Callable


def calculate_sos(h_x_tau: float, h_x100: float, tau: float, rev=False) -> Callable[[float], float]:
    """Calculate the state of safety (SOS) for a given input.

    This function computes the SOS based on the input value, 
    which is expected to be a float.
    The SOS is calculated using the formula: 
    SOS = 1 / (tau * ((h_x - h_x100)/(h_x_tau - h_x100))^2 + 1),
    where x is the input value.

    Args:
        h_x_tau (float): threshold value for h_x_tau.
        h_x100 (float): threshold value for h_x100.
        tau (float): time constant for the calculation.

    Returns:
        float: calculated state of safety (SOS) value.
    """

    # # Constants
    # tau = 0.5
    # h_x_tau = 0.9
    # h_x100 = 1.0
    # calculate constant m
    m = (1/tau - 1) / (h_x_tau - h_x100)**2

    if rev:
        def sos_func_rev(h_x: float) -> float:
            # Calculate SOS using the provided formula
            if h_x > h_x100:
                # it is 100% safe
                return 1
            sos = 1 / (m * (h_x - h_x100)**2 + 1)
            return sos
        return sos_func_rev, m
    else:
        # Calculate SOS using the provided
        def sos_func(h_x: float) -> float:
            # Calculate SOS using the provided formula
            if h_x < h_x100:
                # it is 100% safe
                return 1
            sos = 1 / (m * (h_x - h_x100)**2 + 1)
            return sos
        return sos_func, m
    
    

if __name__ == "__main__":

    # sos_80 = partial(calculate_sos, h_x_tau=2.7, h_x100=1.1, tau=0.25)
    safety_level = 0.8
    # base
    # h_x100 = 1.1
    # h_x_tau = 2.7

    # current
    # sos_type = "Discharge Current (C)"
    # h_x100 = 20
    # h_x_tau = 30

    # # overvoltage
    # sos_type = "Overcharge (V)"
    # h_x100 = 3.6
    # h_x_tau = 4.3

    # # undervoltage
    # sos_type = "Overdischarge (V)"
    # h_x100 = 2.0
    # h_x_tau = 1.0
    # sos_80, m = calculate_sos(h_x_tau, h_x100, safety_level, rev=True)
    
    # # temperature
    degree_sign = u'\N{DEGREE SIGN}'
    sos_type = f"Temperature ({degree_sign}C)"
    h_x100 = 55
    h_x_tau = 90
    sos_80, m = calculate_sos(h_x_tau, h_x100, safety_level)

    


    
    # get the SOS function
    # sos_80, m = calculate_sos(h_x_tau, h_x100, safety_level)

    # Example usage
    # input_value = 3.5
    # sos_value = sos_80(input_value)
    # print(f"State of Safety (SOS) for input {input_value}: {sos_value}")

    # Calculate SOS for a range of input values
    # between h_x_tau and h_X100
    # input_values = np.linspace(1.1, 4.0, 10)
    if h_x100 < h_x_tau:
        input_values = np.linspace(h_x100/2, h_x_tau*2, 10)
    else:
        input_values = np.linspace(h_x_tau/2, h_x100*2, 10)

    sos_values = [sos_80(x) for x in input_values]

    # plot the values as a black line with rectangle box markers
    plt.plot(input_values, sos_values, 'k-', marker='s', markersize=4, label='SOS')
    plt.xlabel(f"{sos_type}")
    plt.ylabel("State of Safety (SOS)")
    plt.title(f"State of Safety (SOS) at {safety_level*100}%, m={m:.6f}")
    # draw a horizontal line at safety of 1 and write "safe" above it
    plt.axhline(y=1, color='b', linestyle='-', label='safe')
    plt.axhline(y=safety_level, color='r', linestyle='-', label='still safe')
    # plot a vertical line at 2.7 and write "h_x_tau" below it
    plt.axvline(x=h_x_tau, color='black', linestyle='--')
    plt.axvline(x=h_x100, color='black', linestyle='--')
    plt.text(h_x_tau + 0.1, 0.54, "h_x_tau", fontsize=10)
    plt.text(h_x100 + 0.1, 0.54, "h_x100", fontsize=10)
    # add ticks at 1.1 and 2.7 along with other ticks at .2format
    plt.xticks([h_x_tau, h_x100] + [round(x) for x in input_values.tolist()], fontsize=8)
    plt.legend()
    # plt.xlim(1.1, 2.7)
    # plt.ylim(0, 2)
    # plt.xticks(np.arange(1.1, 2.7, 0.2))
    # plt.yticks(np.arange(0, 2, 0.2))
    plt.grid()
    plt.show()
