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

    # try to find the data file local. If it doesn't exists download it.
    cache_path = os.path.join("data", "iot")
    cache_file = os.path.join(cache_path, "mem_test.csv")
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)
    if not os.path.exists(cache_file):
        urlretrieve(input_url, cache_file)
        print("downloaded data successfully from ", input_url)
    else:
        print("using cache for ", input_url)

    df = pd.read_csv(cache_file, index_col="time", parse_dates=['time'], dtype=np.float32)

    df["time"] = df.index.time

    # normalize data
    #df['solar.current'] /= normalize
    df["mem"] /= normalize
    #print(df)
    # group by minute, find the average for a minute and add a new column .avg

    #假设你想要按key1进行分组，并计算data1列的平均值，我们可以访问data1，并根据key1调用groupby：
    #grouped = df['cpu'].groupby(df.index.time).mean()

    grouped = df.groupby(df.index.time).max()
    #print(grouped)
    #print(type(grouped))
    #df['cpu.avg']=grouped
    grouped.columns = ["mem.max","time"]


    #grouped = grouped[["cpu.avg", "time"]]
    # merge continuous readings and daily max values into a single frame
    df_merged = pd.merge(df, grouped, right_index=True, on="time")
    df_merged = df_merged[[ "mem", "mem.max"]]
    # we group by minute so we can process a minute at a time.
    grouped = df_merged.groupby(df_merged.index.time)
    per_day = []
    for _, group in grouped:
        per_day.append(group)
    #print(type(per_day))
    # split the dataset into train, validatation and test sets on day boundaries
    #val_size = int(len(per_day) * val_size)
    #test_size = int(len(per_day) * test_size)
    #next_val = 0
    #next_test = 0

    result_x = {"test": []}
    result_y = {"test": []}

    # generate sequences a minute at a time
    for i, day in enumerate(per_day):
        total = day["mem.max"].values
        #print(len(total))
        current_set = "test"
        max_total_for_day = np.array(day["mem.max"].values[0])
        #print(max_total_for_day)
        for j in range(1, len(total)):
            result_x[current_set].append(total[0:j])
            result_y[current_set].append([max_total_for_day])
            if j >= time_steps:
                break
    # make result_y a numpy array
    #print(result_x)
    for ds in ["test"]:
        result_y[ds] = np.array(result_y[ds])
    return result_x, result_y


TIMESTEPS = 20
NORMALIZE = 1

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

z = load_model("./Model/mem_train2.cntk")
z = z(x)

for j, ds in enumerate([ "test"]):
    results = []
    for x_batch, _ in next_batch(X, Y, ds):
        pred = z.eval({x: x_batch})
        # print(pred)
        #np.savetxt('./datatest/cputest.csv', pred)
        # pred = abs(pred)
        # pred = np.square(pred*100)
        results.extend(pred[:, 0])
    #print(len(results))
    a[j].plot((Y[ds] * NORMALIZE).flatten(), label=ds + ' raw')
    #np.savetxt('./datatest/ec2_nab.csv', Y[ds])
    a[j].plot(np.array(results) * NORMALIZE, label=ds + ' pred')
    #np.savetxt('./datatest/ec2_pred_nab.csv', np.array(results))
    a[j].legend()

plt.show()