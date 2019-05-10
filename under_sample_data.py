import pandas as pd
import time
import numpy as np


def under_sample(dataframe, undersample_number):
    result = dataframe.groupby('event').apply(
        lambda s: s.sample(undersample_number))
    return result


if __name__ == "__main__":
    since = time.time()
    dtypes = {"crew": "int8",
              "experiment": "category",
              "time": "float32",
              "seat": "int8",
              "eeg_fp1": "float32",
              "eeg_f7": "float32",
              "eeg_f8": "float32",
              "eeg_t4": "float32",
              "eeg_t6": "float32",
              "eeg_t5": "float32",
              "eeg_t3": "float32",
              "eeg_fp2": "float32",
              "eeg_o1": "float32",
              "eeg_p3": "float32",
              "eeg_pz": "float32",
              "eeg_f3": "float32",
              "eeg_fz": "float32",
              "eeg_f4": "float32",
              "eeg_c4": "float32",
              "eeg_p4": "float32",
              "eeg_poz": "float32",
              "eeg_c3": "float32",
              "eeg_cz": "float32",
              "eeg_o2": "float32",
              "ecg": "float32",
              "r": "float32",
              "gsr": "float32",
              "event": "category",
              }

    print("read the data frame...")
    train_df = pd.read_csv(
        "../reducing-Commercial-Aviation-Fatalities/train.csv", dtype=dtypes)

    print("under sample the data set...")
    under_sample_data = under_sample(train_df, undersample_number=50000)

    print("writing the data frame to csv.")
    under_sample_data.to_csv(
        "./input/undersample_data_5w_each_event.csv", index=None, header=True)

    time_elapsed = time.time() - since
    print('Under sampling completed: {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
