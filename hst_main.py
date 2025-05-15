"""Module running the basic kalman filter."""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime

from basic_kf import KalmanFilter
from models.linear_hst_model import LinearHSTModel


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
    # create a kalman filter object
    linear_hst_model = LinearHSTModel(
        state_v=np.array([10]).reshape(-1, 1),
        input_v=np.array([0, 0]).reshape(-1, 1),
        measurement_v=np.array([0]).reshape(-1, 1),
    )

    linear_hst_model.state_v

    m = 0.8

    # linear_hst_mod
    kf = KalmanFilter(
        linear_hst_model,
        1,
        {},
    )
    estimated_temp = []
    curr_max = oil_df["IL1"].max()
    r_max = (oil_df["VL1"] / oil_df["IL1"]).max()
    print(f"Max r is {r_max}")
    print(f"length of data is {len(oil_df)}")
    # run the model over the oil temperature data
    # time this loop
    start_time = datetime.datetime.now()
    for row in oil_df.itertuples():
        temperature = row.OTI
        # curr = row.IL1
        curr = np.linalg.norm([row.IL1, row.IL2, row.IL3]) * (3 ** (1 / 2))
        # normalize current
        curr = curr / curr_max
        # update model based on delta_t
        delta_t = row.delta_t
        kf.model.update_delta_t(delta_t)
        kf.kalman_step(
            np.array([temperature, curr**(2*m)]).reshape(-1, 1),
            np.array([temperature]).reshape(-1, 1),
        )
        estimated_temp.append(kf.model.state_v[0])
    # print time taken in milliseconds
    print(f"Time taken: {(datetime.datetime.now() - start_time).total_seconds() * 1000:.2f} ms")
    # plot the estimated temperature
    plt.plot(oil_df["delta_t"].cumsum(), estimated_temp)
    plt.plot(oil_df["delta_t"].cumsum(), oil_df["OTI"])
    plt.legend(["Estimated Temperature", "Actual Temperature"])
    plt.xlabel("Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.title("Estimated Temperature vs Actual Temperature")
    # save the plot in the current directory
    plt.savefig("estimated_temperature.png")
