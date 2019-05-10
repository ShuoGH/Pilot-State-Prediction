import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures


def drop_features(data_set):
    '''
    Drop the experiment, time, seat and id(test data set)
    '''
    dataframe = pd.DataFrame(data_set)
    for i in {'experiment', 'time', 'seat', 'id'}:
        if i in dataframe.columns:
            dataframe = dataframe.drop([i], axis=1)
    return dataframe


def data_convert(data_set):
    '''
    Used to conver the elements in data frame from object to float. (if you don't drop these columns)
    Input:
        train data set / test data set
    Return:
        converted data set
    '''
    dic = {'CA': 0, 'DA': 1, 'SS': 2, 'LOFT': 3}
    dic = {'A': 0, 'B': 1, 'C': 2, 'D': 3}

    data_set["experiment"] = data_set["experiment"].apply(lambda x: dic[x])
    data_set["event"] = data_set["event"].apply(lambda x: dic[x])
    return data_set


def normalize_data(dst_train):
    '''
    Normalize the data set by the pilots.
    Return:
        the normalized data frame
    '''
    features_n = ["eeg_fp1", "eeg_f7", "eeg_f8", "eeg_t4", "eeg_t6", "eeg_t5", "eeg_t3", "eeg_fp2", "eeg_o1", "eeg_p3",
                  "eeg_pz", "eeg_f3", "eeg_fz", "eeg_f4", "eeg_c4", "eeg_p4", "eeg_poz", "eeg_c3", "eeg_cz", "eeg_o2", "ecg", "r", "gsr"]
    dst_train['pilot'] = 100 * dst_train['seat'] + dst_train['crew']
    pilots = dst_train["pilot"].unique()
    print("Normalizing the data by pilots....")
    for pilot in pilots:
        ids = dst_train[dst_train["pilot"] == pilot].index
        scaler = MinMaxScaler()
        dst_train.loc[ids, features_n] = scaler.fit_transform(
            dst_train.loc[ids, features_n])
    return normalized_data


def make_interactions(dataframe):
    poly = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False)
    result = poly.fit_transform(dataframe)[..., len(dataframe.columns):]
    return pd.DataFrame(result,
                        columns=poly.get_feature_names(
                            dataframe.columns)[-result.shape[1]:],
                        index=dataframe.index)


def feature_engg(df):
    '''
    Input:
        DataFrame which contains the data set
    return:
        DataFrame which did feature engineer
    '''
    # make copy
    dataframe = pd.DataFrame(df)
    # drop experiment, time and seat and id(in test)
    for i in {'experiment', 'time', 'seat', 'id'}:
        if i in dataframe.columns:
            dataframe = dataframe.drop([i], axis=1)

    phi_df = dataframe[['crew', 'ecg', 'r', 'gsr']].copy()
    if 'event' in dataframe.columns:
        phi_df = phi_df.join(dataframe['event'])
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_fp1','eeg_fp2','eeg_fz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_f3','eeg_f4','eeg_f7','eeg_f8', 'eeg_fz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_t3','eeg_t4','eeg_t5','eeg_t6']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_c3','eeg_c4','eeg_cz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_p3','eeg_p4','eeg_poz','eeg_pz']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_o1','eeg_o2']]))
    # phi_df = phi_df.join(make_interactions(dataframe[['eeg_pz','eeg_poz','eeg_cz','eeg_fz']]))
    phi_df = phi_df.join(make_interactions(dataframe[['crew', 'ecg']]))
    phi_df = phi_df.join(make_interactions(dataframe[['crew', 'r']]))
    phi_df = phi_df.join(make_interactions(dataframe[['crew', 'gsr']]))
    phi_df = phi_df.join(make_interactions(dataframe[['ecg', 'gsr']]))
    return phi_df
