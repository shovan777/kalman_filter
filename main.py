"""Module running the basic kalman filter."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from ekf import ExtendedKalmanFilter
from models.tot_model import SUSATOTModel


# Oil Temperature
# load oil temperature data
# oil_df = pd.read_csv("transformer_below60.csv")
oil_df = pd.read_csv("transformer_all.csv")

print(f"Data shape original: {oil_df.shape}")
# filter out data that is above 200 degree celsius
oil_df = oil_df[oil_df["OTI"] <= 200]
print(f"Data shape after filtering 200: {oil_df.shape}")

# filter out data that is below 10 degree celsius
oil_df = oil_df[oil_df["OTI"] >= 10]
print(f"Data shape after filtering 10: {oil_df.shape}")

# filter out data if ATI is less than 10 degree celsius
oil_df = oil_df[oil_df["ATI"] >= 10]
print(f"Data shape after filtering ATI: {oil_df.shape}")

# calculate time delta
# Calculate the equivalent ageing factor of the transformer
# preserve oil_df for future use
oil_df_copy = oil_df.copy()
# add a new column delta_t to the dataframe which differences the time and convert to hrs
oil_df["delta_t"] = oil_df["DeviceTimeStamp"].apply(pd.Timestamp).diff().dt.total_seconds() / 3600
# drop first row
oil_df = oil_df[1:]
oil_df["cum_delta_t"] = oil_df["delta_t"].cumsum()

if __name__ == "__main__":
    # get the first value of OTI
    first_oti = oil_df["OTI"].iloc[0]
    first_oti = 60.0
    print(f"First OTI: {type(first_oti)}")

    delta_t = oil_df["delta_t"].iloc[0]

    susa_tot_model = SUSATOTModel(
        # top oil temp, oil exponent, oil time constant
        state_v=np.array([first_oti, 0.5, 0.6]).reshape(-1, 1),
        # R: load ratio, K load factor, delta_t, ambient temp
        input_v=np.array([1.0, 1.0, delta_t, 25.0]).reshape(-1, 1),
        measurement_v=np.array([0, 0, 0]).reshape(-1, 1),
    )

    ekf = ExtendedKalmanFilter(
        susa_tot_model,
        1,
        {},
    )


    curr_max = oil_df["IL1"].max()
    rated_current = curr_max #TODO: find out what is the actual rated current

    estimated_tots = []
    measured_tots = []

    i = 0
    start_time = datetime.datetime.now()

    # run the model over the oil temperature data
    for row in oil_df.itertuples():
        temperature = row.OTI
        curr = np.linalg.norm([row.IL1, row.IL2, row.IL3]) * (3 ** (1 / 2))
        # normalize current
        curr = curr / curr_max
        # update model based on delta_t
        delta_t = row.delta_t
        # print(f"delta_t: {delta_t}")
        # TODO: how to find R load loss ratio at rated load
        # is it constant??
        input_v = np.array([1.0, curr, delta_t, row.ATI]).reshape(-1, 1)
        # # susa_tot_model.predict_next_state(input_v)
        # ekf.kalman_step(
        #     input_v,
        #     np.array([temperature]).reshape(-1, 1),
        # )
        ekf.predict_step(input_v)
        ekf.update_step(np.array([temperature]).reshape(-1, 1))
        
        
        estimated_tots.append(np.copy(susa_tot_model.state_v[0]))
        measured_tots.append(temperature)
    
    # print time taken in milliseconds
    print(f"Time taken: {(datetime.datetime.now() - start_time).total_seconds() * 1000:.2f} ms")
    
    # plot the estimated temperature
    plt.plot(oil_df["delta_t"].cumsum(), estimated_tots, label="Predicted TOT by SUSA")
    plt.plot(oil_df["delta_t"].cumsum(), measured_tots, label="Measured TOT")
    plt.legend()
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.title("TOT prediction based on SUSA model starting at different TOT at 60 °C")
    plt.savefig("estimated_tot.png")
