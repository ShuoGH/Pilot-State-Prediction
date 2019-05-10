import numpy as np
import pandas as pd
import mlp
from sklearn.model_selection import train_test_split
import data_util
import torch

if __name__ == "__main__":
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

    print("loading the data...")
    dst_df = pd.read_csv(
        "./input/undersample_data_5w_each_event.csv", dtype=dtypes)
    print("The dimension of data: {}".format(dst_df.shape))

    X = dst_df.drop(columns=['event'], axis=1)
    Y = dst_df['event']

    # ---- normalize data set and drop some features----
    X_normalized = data_util.normalize_data(X)
    X_normalized_droped = data_util.drop_features(X_normalized)

    # ----split the train and val data set. Freezing the random seed----
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_normalized_droped, Y, test_size=0.2, random_state=42)

    # # ---- one-hot encode the class----
    # # return ndarray, in which each row represents a class
    # y_train = np.eye(4)[y_train]
    # y_valid = np.eye(4)[y_valid]

    # ---- Convert the tensor----
    featuresTrain = torch.from_numpy(X_train).float()
    targetsTrain = torch.from_numpy(y_train).type(
        torch.LongTensor)  # data type is long

    featuresValid = torch.from_numpy(X_valid).float()
    targetsValid = torch.from_numpy(y_valid).type(
        torch.LongTensor)  # data type is long

    # ---- dataloader to load the tensor----
    batch_size = 512
    train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
    valid = torch.utils.data.TensorDataset(featuresValid, targetsValid)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

    # ---- initialize the model----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_features = 27
    n_neurons = [100, 50]
    dropouts = [0.3, 0.2]

    model = mlp.MLPNetwork(n_features, n_neurons, dropouts)
    model = model.to(device)

    # Cross Entropy Loss
    # See the doc: https://pytorch.org/docs/stable/nn.html?highlight=nllloss#torch.nn.NLLLoss
    error = nn.NLLLoss()

    # Adam Optimizer
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    # train_acc = []
    valid_losses = []
    # valid_acc = []
    num_epochs = 50
    for e in range(num_epochs):
        start_time = time.time()
        # keep track of training and validation loss
        train_loss = 0.0

        model.train()
        for i, (features, labels) in enumerate(train_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = error(outputs, labels)

            loss.backward()
            optimizer.step()
            # update training loss
            train_loss += loss.item()

            #  Calculate the accuracy on the train test
            # acc = torch.eq(outputs.round(), labels).float().mean() # accuracy
            # _, predicted = torch.max(outputs.data, 1)
            # _, actual = torch.max(labels, 1)
            # total = len(labels)
            # correct = (predicted == actual).sum()
            # train_accuracy = 100 * correct / total
            # train_losses.append(train_loss.item())
            # train_acc.append(train_accuracy.item())

        valid_loss = 0.0
        model.eval()
        for i, (features, labels) in enumerate(valid_loader):
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = error(outputs, labels)
            # loss = error(outputs, labels)
            valid_loss += loss.item()

            # _, predicted = torch.max(outputs.data, 1)
            # _, actual = torch.max(labels, 1)
            # total = len(labels)
            # correct = (predicted == actual).sum()
            # accuracy = 100 * correct / total
            # valid_losses.append(loss.item())
            # valid_acc.append(accuracy.item())

        # calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        valid_loss = valid_loss / len(valid_loader.dataset)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        elapsed_time = time.time() - start_time
        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tTime: {:.2f}'.format(
            e+1, train_loss, valid_loss, elapsed_time))
    print("*"*10)
    print("Train losses history:\n {}".format(train_losses))
    print("Validation losses history:\n {}".format(valid_losses))
    #     if e % 1 == 0:
    #         print("[{}/{}], Train Loss: {} Train Acc: {}, Validation Loss : {}, Validation Acc: {} ".format(e+1,
    #         num_epochs, np.round(train_loss.item(), 3), np.round(train_accuracy.item(), 3),
    #         np.round(loss.item(), 3), np.round(accuracy.item(), 3)))
