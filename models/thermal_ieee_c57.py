"""
Calculates the thermal model of a transformer using the IEEE C57.91-1995 standard.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

AgeingSlope = 15000
A_110 = 9.8 * (10 ** (-18))


def calculate_per_unit_life(temp: float) -> float:
    """
    Calculate the per unit life of a transformer using the IEEE C57.91-2011 standard.

    Args:
        temp (float): The temperature of the transformer.

    Returns:
        float: The per unit life of the transformer.
    """
    return A_110 * np.exp((AgeingSlope) * (1 / (temp + 273.15)))


def calculate_temperature(
    ambient_temperature: float, load: float, time: float
) -> float:
    """
    Calculate the temperature of a transformer using the IEEE C57.91-2011 standard.

    Args:
        ambient_temperature (float): The ambient temperature.
        load (float): The load on the transformer.
        time (float): The time for which the transformer has been running.

    Returns:
        float: The temperature of the transformer.
    """
    # Calculate the temperature of the transformer
    temperature = ambient_temperature + load * time
    return temperature


def calculate_percentage_life(F_EQA: float, total_time: float) -> float:
    normal_insulation_life = 180000
    return (1 - (F_EQA * total_time / normal_insulation_life)) * 100


def calculate_aging_accln_factor(temp: float) -> float:
    """
    Calculate the aging acceleration factor of a transformer using the IEEE C57.91-2011 standard.

    Args:
        temp (float): The temperature of the transformer.

    Returns:
        float: The aging acceleration factor of the transformer.
    """
    # Calculate the aging acceleration factor of the transformer
    return np.exp((AgeingSlope) * (1 / 383 - 1 / (temp + 273.15)))


def calculate_equivalent_ageing_factor(temp_arr: list[float], dtime_arr: list[float]) -> float:
    """
    Calculate the equivalent ageing factor of a transformer using the IEEE C57.91-2011 standard.
    Accumulates the effect of temperature for a certain time on the ageing of the transformer.

    Args:
        temp_arr (float): The array of temperature of the transformer.
        dtime_arr (float): The time for which the transformer
                        operated at the given temperature.

    Returns:
        float: The equivalent ageing factor of the transformer.
    """
    # Calculate the equivalent ageing factor of the transformer
    sum_eqa = 0
    sum_t = 0
    for temp, dtime in zip(temp_arr, dtime_arr):
        sum_eqa += calculate_aging_accln_factor(temp) * dtime
        sum_t += dtime
    return sum_eqa / sum_t

if __name__ == "__main__":
    # Calculate the temperature of the transformer
    # range of hotspot temperature is 60-200 degree celsius
    # spaced at 1 degree celsius
    hot_spot_temp = np.arange(60, 200, 1)

    # Calculate the per unit life of the transformer
    per_unit_life = [calculate_per_unit_life(temp) for temp in hot_spot_temp]

    # Plot the per unit life with y axis in log scale
    plt.plot(hot_spot_temp, per_unit_life)
    plt.text(180, 4e7, "A = 9.8E-18\nB = 15000")
    plt.yscale("log")
    plt.xlabel("Hotspot Temperature (°C)")
    plt.ylabel("Per Unit Life (hours)")
    plt.title("Per Unit Life vs Hotspot Temperature")
    # display A and B values in the plot
    plt.show()

    # Oil Temperature
    # load oil temperature data
    # oil_df = pd.read_csv("transformer_below60.csv")
    oil_df = pd.read_csv("transformer_all.csv")

    # get temperature data from OTI column
    oil_temp = oil_df["OTI"]

    print(oil_temp.shape)
    print(oil_temp.max())

    # caculate RUL of the transformer
    rul = np.array([calculate_per_unit_life(temp) for temp in oil_temp])

    plot_until = -1
    max_year = 180000
    hours_to_years = 1 / 8760

    rul_years = rul * hours_to_years
    lowest_rul = rul_years.min()
    highest_rul = rul_years.max()
    print(f"Lowest RUL: {lowest_rul} years")
    # plot the RUL of the transformer
    # plt.scatter(oil_temp[:plot_until] * 180000, rul[:plot_until])
    plt.scatter(oil_temp, rul_years)
    # plt.yscale("log")
    # plot the lowest RUL
    plt.axhline(y=lowest_rul, color="r", linestyle="--")
    # plot the highest RUL
    plt.axhline(y=highest_rul, color="g", linestyle="--")
    # show also the value in exponential form
    plt.text(180, lowest_rul, f"Lowest RUL: {lowest_rul:.2e} years")
    plt.text(180, highest_rul, f"Highest RUL: {highest_rul:.2f} years")
    plt.xlabel("Oil Temperature (°C)")
    plt.ylabel("Remaining Useful Life (years)")
    plt.title("Remaining Useful Life vs Oil Temperature")
    plt.show()

    # Calculate the aging acceleration factor of the transformer
    # range of hotspot temperature is 60-200 degree celsius
    # spaced at 1 degree celsius
    hot_spot_temp = np.arange(60, 200, 1)
    ageing_accln_factor = [calculate_aging_accln_factor(temp) for temp in hot_spot_temp]

    # plot the aging acceleration factor
    plt.plot(hot_spot_temp, ageing_accln_factor)
    plt.xlabel("Hotspot Temperature (°C)")
    plt.ylabel("Aging Acceleration Factor")
    plt.yscale("log")
    plt.title("Aging Acceleration Factor vs Hotspot Temperature")
    plt.show()
    print(oil_df.head())

    # Calculate the equivalent ageing factor of the transformer
    # delta_t = map(lambda x: x[1] - x[0], zip(oil_temp, oil_temp[1:]))
    # eq_ageing_factor = calculate_equivalent_ageing_factor(oil_temp, delta_t)
    # print(f"Equivalent Ageing Factor: {eq_ageing_factor}")
    # print(f"Percentage Life: {calculate_percentage_life(eq_ageing_factor, 24)}")