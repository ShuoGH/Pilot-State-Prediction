import numpy as np
import pandas as pd
import mlp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import data_util
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from torch import nn
import time
from tqdm import tqdm
from sklearn.exceptions import DataConversionWarning
import warnings
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def format_data_set(data_set):
    '''
    input:
        DataFrame: 50000 chunksize
    Return:
        formatted data set.
    '''
    X_droped = data_util.drop_features(data_set)
    # print(X_droped.shape) # 50000 24

    # ----scale the data set----
    scaler = StandardScaler()
    X_droped_scaled = scaler.fit_transform(X_droped)
    return X_droped_scaled


if __name__ == "__main__":
    dtypes = {"crew": "int8",
              "experiment": "category",
              "time": "float32",
              "seat": "int8",
              "event": "category",
              }

    # ---- initialize the model----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_features = 24
    n_neurons = [100, 50]
    dropouts = [0.3, 0.2]

    model = mlp.MLPNetwork(n_features, n_neurons, dropouts)
    model = model.to(device)
    # ---- load the pretrained weights----
    # model.load_state_dict(torch.load(
    #     './trained_model/MLP_10thMay.model'))
    model.eval()

    # ----loading the data----
    chunksize = 500000  # num of rows to read from test file at once
    batch_size = 512
    print("loading the test data...")
    iterator = pd.read_csv(
        "../reducing-Commercial-Aviation-Fatalities/test.csv", dtype=dtypes, chunksize=chunksize)

    # ---- Store the prediction of the models----
    prediction = None
    print("Predicting...")
    for test_chunk in tqdm(iterator):
        test_chunk_copy = test_chunk.copy()
        test_data_5w = torch.from_numpy(
            format_data_set(test_chunk_copy)).float()

        test = torch.utils.data.TensorDataset(test_data_5w)
        # print(test)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

        for i, (x_batch,) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            # print(x_batch)

            y_pred = model(x_batch).to('cpu').detach().numpy()
            # ----find the index of the label----
            class_index = np.argmax(y_pred, axis=1)
            y_pred_batch = np.zeros(y_pred.shape)
            y_pred_batch[np.arange(y_pred.shape[0]), class_index] = 1
            # print(y_pred)
            # print(y_pred_batch)
            if prediction is None:
                prediction = y_pred_batch
            else:
                prediction = np.append(prediction, y_pred_batch, axis=0)

    print("Finish test prediction, the length is {}".format(len(prediction)))
    # save
    print("*"*10)
    print("Writing the submission file...")
    pd.DataFrame(prediction).to_csv("submission.csv", index=True,
                                    index_label='id', header=['A', 'B', 'C', 'D'])
