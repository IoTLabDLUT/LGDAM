from matplotlib import pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time

import cntk as C
from cntk.ops.functions import load_model

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

import cntk.tests.test_utils

cntk.tests.test_utils.set_device_from_pytest_env()  # (only needed for our build system)


def generate_solar_data(input_url, time_steps, normalize=1, val_size=0.1, test_size=0.1):
    """
    generate sequences to feed to rnn based on data frame with solar panel data
    the csv has the format: time ,solar.current, solar.total
     (solar.current is the current output in Watt, solar.total is the total production
      for the day so far in Watt hours)
    """
    # try to find the data file local. If it doesn't exists download it.
    cache_path = os.path.join("data", "iot")
    cache_file = os.path.join(cache_path, "solar2.csv")
    print(cache_file)
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        urlretrieve(input_url, cache_file)
        print("downloaded data successfully from ", input_url)
    else:
        print("using cache for ", input_url)

    df = pd.read_csv(cache_file, index_col="time", parse_dates=['time'], dtype=np.float32)

    df["date"] = df.index.date

    # normalize data
    #df['solar.current'] /= normalize
    df['solar.total'] /= normalize

    # group by day, find the max for a day and add a new column .max
    grouped = df.groupby(df.index.date).max()
    grouped.columns = ["solar.total.max", "date"]

    # merge continuous readings and daily max values into a single frame
    df_merged = pd.merge(df, grouped, right_index=True, on="date")
    df_merged = df_merged[["solar.total", "solar.total.max"]]
    # we group by day so we can process a day at a time.
    grouped = df_merged.groupby(df_merged.index.date)

    per_day = []
    for _, group in grouped:
        per_day.append(group)

    #out = pd.DataFrame(per_day)
    #out.to_csv('./datatest/out2.xls', float_format='%.9f')
    #print(type(per_day))
    # split the dataset into train, validatation and test sets on day boundaries
    val_size = int(len(per_day) * val_size)
    test_size = int(len(per_day) * test_size)
    next_val = 0
    next_test = 0

    result_x = {"train": [], "val": [], "test": []}
    result_y = {"train": [], "val": [], "test": []}

    # generate sequences a day at a time
    for i, day in enumerate(per_day):
        # if we have less than 8 datapoints for a day we skip over the
        # day assuming something is missing in the raw data
        total = day["solar.total"].values
        if len(total) < 8:
            continue
        if i >= next_val:
            current_set = "val"
            next_val = i + int(len(per_day) / val_size)
        elif i >= next_test:
            current_set = "test"
            next_test = i + int(len(per_day) / test_size)
        else:
            current_set = "train"
        max_total_for_day = np.array(day["solar.total.max"].values[0])
        for j in range(2, len(total)):
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break
    # make result_y a numpy array
    for ds in ["train", "val", "test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y



TIMESTEPS = 14
NORMALIZE = 20000

X, Y = generate_solar_data("F:\\cntktest\\data\\iot\\solar.csv", TIMESTEPS, normalize=NORMALIZE)

BATCH_SIZE = TIMESTEPS * 10
x = C.sequence.input_variable(1)


def next_batch(x, y, ds):
    """get the next batch for training"""

    def as_batch(data, start, count):
        return data[start:start + count]

    for i in range(0, len(x[ds]), BATCH_SIZE):
        yield as_batch(X[ds], i, BATCH_SIZE), as_batch(Y[ds], i, BATCH_SIZE)


f, a = plt.subplots(2, 1, figsize=(12, 8))

z = load_model("./Model/modelDemo.cntk")
z = z(x)

for j, ds in enumerate(["val", "test"]):
    results = []
    for x_batch, _ in next_batch(X, Y, ds):
        pred = z.eval({x: x_batch})
        #print(pred)
        #np.savetxt('./datatest/new250.csv', pred)
        # pred = abs(pred)
        # pred = np.square(pred*100)
        results.extend(pred[:, 0])

    a[j].plot((Y[ds] * NORMALIZE).flatten(), label=ds + ' raw')
    #print(Y[ds])
    #np.savetxt('./datatest/new12.csv', Y[ds])
    a[j].plot(np.array(results) * NORMALIZE, label=ds + ' pred')
    #np.savetxt('./datatest/new11.csv', np.array(results))
    #np.savetxt('./datatest/new250.csv', pred)
    a[j].legend()

plt.show()