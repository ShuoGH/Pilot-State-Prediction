import lightgbm as lgb
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
from sklearn.metrics import confusion_matrix, log_loss

'''
For the example of using lightGBM case:
    https://github.com/Microsoft/LightGBM/blob/master/examples/python-guide/advanced_example.py#L82-L84


    save lgbm classifier:
        gbm.save_model('model.txt')
    load lgbm classifier:
        bst = lgb.Booster(model_file='model.txt')
'''


def normalize_by_pilots(df):
    pilots = df["pilot"].unique()
    for pilot in tqdm(pilots):
        ids = df[df["pilot"] == pilot].index
        scaler = MinMaxScaler()
        df.loc[ids, features_n] = scaler.fit_transform(df.loc[ids, features_n])
    return df


def run_lgb(df_train, df_test, features_used):
    # Classes as integers
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    try:
        df_train["event"] = df_train["event"].apply(lambda x: dic[x])
        df_test["event"] = df_test["event"].apply(lambda x: dic[x])
    except:
        pass

    params = {"objective": "multiclass",
              "num_class": 4,
              #   "metric": "multi_error",
              "metric": "multi_logloss",
              "num_leaves": 30,
              "min_child_weight": 50,
              "learning_rate": 0.1,
              "bagging_fraction": 0.7,
              "feature_fraction": 0.7,
              "bagging_seed": 420,
              "verbosity": -1
              }
    '''
    Train 1000 number of iterations.
    '''
    lg_train = lgb.Dataset(df_train[features], label=(df_train["event"]))
    lg_test = lgb.Dataset(df_test[features], label=(df_test["event"]))
    model = lgb.train(params, lg_train, 1000, valid_sets=[
                      lg_test], early_stopping_rounds=50, verbose_eval=100)

    # ----save the lightGBM model to file----
    model_path = './trained_model/model_lightGBM_1.txt'
    model.save_model(model_path)
    print("Ran the lightGBM model and save it successfully: {}".format(model_path))
    return model


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
    # ---- load the data set----
    print("Loading the train data and test data....")
    train_df = pd.read_csv(
        "../reducing-Commercial-Aviation-Fatalities/train.csv", dtype=dtypes)
    test_df = pd.read_csv(
        "../reducing-Commercial-Aviation-Fatalities/test.csv", dtype=dtypes)
    print("Load data successfully.")

    # ---- data processing ----
    features_n = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3",
                  "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]
    train_df['pilot'] = 100 * train_df['seat'] + train_df['crew']
    test_df['pilot'] = 100 * test_df['seat'] + test_df['crew']
    # normalize the data set
    print("Normalizing the train data frame...")
    train_df = normalize_by_pilots(train_df)
    print("Normalizing the test data frame...")
    test_df = normalize_by_pilots(test_df)

    # ---- split the data set into train and val----
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=420)

    # ---- only keep these features----
    features = ["crew", "seat"] + features_n

    # ---- run the lightGBM and save the model ----
    print("Run the lightGBM...")
    model = run_lgb(train_df, val_df, features)
    pred_val = model.predict(
        val_df[features], num_iteration=model.best_iteration)

    print("Log loss on validation data :", round(
        log_loss(np.array(val_df["event"].values), pred_val), 3))

    # ---- make prediction on the test data set----
    print("Predicting the test data set...")
    pred_test = model.predict(
        test_df[features], num_iteration=model.best_iteration)
    submission = pd.DataFrame(np.concatenate((np.arange(len(test_df))[
                              :, np.newaxis], pred_test), axis=1), columns=['id', 'A', 'B', 'C', 'D'])
    submission['id'] = submission['id'].astype(int)
    submission.to_csv("./results/submission_lightGBM_1.csv", index=False)
