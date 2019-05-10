

def get_rcaf_data_set(input_path, under_sample=False, normalize=False):
    '''
    To read the data frame.
    Input parameter:
        input_path: path to open the csv file 
        downsample: if True, load the data set which is downsampled
        normalize: if True, load the data set which is normalized
    return:
        data frame which contains the data set.

    Downsample:
        the downsampled data set contains that 5w data set for each event
    '''
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
    if under_sample:
        pass
    else:
        df = pd.read_csv(input, dtype=dtypes)
    return df


# feature engineering
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
