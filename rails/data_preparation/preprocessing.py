import pandas as pd
import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pickle
import os

encoders_path = 'rails/data_preparation/'


def to_datetime_days(days_timestamp):
    return datetime.datetime.fromtimestamp(days_timestamp * 86400)


def to_timestamp_days(date):
    return int(datetime.datetime.timestamp(date) / 86400)


def coding_update(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result['update'] = pd.to_datetime(result['update']).apply(to_timestamp_days)
    return result


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    result = coding_update(df)
    return result.drop(columns=['num', 'route', 'route_id', 'station', 'station_id', 'start_date'])


def get_no_leak(df_features: pd.DataFrame, update_cut: str, **kwargs) -> tuple[pd.DataFrame, pd.DataFrame]:
    predict = kwargs.get('predict', False)
    shuffle = kwargs.get('shuffle', False)
    df_no_leak = df_features.copy()
    if predict:
        update = df_no_leak[(df_no_leak['update'] >= update_cut)].drop(columns=['arrival'])
        return update
    else:
        df_no_leak.rename(columns={'time_to_home': 'target'}, inplace=True)

        df_no_leak.drop(df_no_leak[(df_no_leak['target'] < 0)].index, inplace=True)

        df_no_leak.reset_index(drop=True)
        train = df_no_leak[(df_no_leak['update'] <= update_cut) &
                           (df_no_leak['arrival'] <= update_cut)].drop(columns=['arrival'])
        test = df_no_leak[(df_no_leak['update'] > update_cut)].drop(columns=['arrival'])

        if shuffle:
            train = train.sample(frac=1)
            test = test.sample(frac=1)
        return train, test


def get_scaler(df: pd.DataFrame, load: bool = False) -> StandardScaler:
    filename = f'{encoders_path}scaler.pkl'
    if load and os.path.isfile(filename):
        with open(filename, 'rb') as file:
            scaler = pickle.load(file)
    else:
        scaler = StandardScaler()
        scaler.fit(df)
        with open(filename, 'wb') as file:
            pickle.dump(scaler, file)
    return scaler


def get_PCA(df: pd.DataFrame, load: bool = False) -> tuple[StandardScaler, PCA]:
    scaler = get_scaler(df, load)
    scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
    filename = f'{encoders_path}pca.pkl'

    if load and os.path.isfile(filename):
        with open(filename, 'rb') as file:
            pca = pickle.load(file)
    else:
        pca = PCA()
        pca.fit(scaled_df)
        with open(filename, 'wb') as file:
            pickle.dump(pca, file)
    return scaler, pca


def PCA_encoding(df: pd.DataFrame, training: bool = True, only_PCA: bool = False) -> pd.DataFrame:
    if training:
        features = df.drop('target', axis=1)
        target = df['target']
    else:
        features = df

    scaler, pca = get_PCA(features, load=not training)

    scaled_features = pd.DataFrame(scaler.transform(features), columns=features.columns, index=features.index)
    df_PCA = pca.transform(scaled_features)
    component_names = [f"PC{i+1}" for i in range(df_PCA.shape[1])]
    result = pd.DataFrame(df_PCA, columns=component_names, index=features.index)

    if not only_PCA:
        result = pd.concat([features, result], axis=1)

    if training:
        result['target'] = target

    return result


def get_data(data: pd.DataFrame, update_cut: str, **kwargs) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]] | dict[str, pd.DataFrame]:

    training = kwargs.get('training', True)
    history_cut = kwargs.get('history_cut', False)
    only_PCA = kwargs.get('only_PCA', False)

    if history_cut:
        df = data[data['update'] >= history_cut].copy()
    else:
        df = data.copy()

    if training:

        train, test = get_no_leak(df, update_cut, shuffle=True)

        coded_train = preprocessing(train.copy())
        coded_test = preprocessing(test.copy())

        no_PCA_train = coded_train.copy()
        no_PCA_test = coded_test.copy()

        PCA_train = PCA_encoding(coded_train.copy(), training=True, only_PCA=only_PCA)
        PCA_test = PCA_encoding(coded_test.copy(), training=True, only_PCA=only_PCA)

        return {'no_PCA': (no_PCA_train, no_PCA_test), 'PCA': (PCA_train, PCA_test)}

    else:

        update = get_no_leak(df.copy(), update_cut, predict=True)

        update = preprocessing(update.copy()).drop(columns=['target'])

        no_PCA_update = update.copy()

        PCA_update = PCA_encoding(update.copy(), training=False, only_PCA=only_PCA)

        return {'no_PCA': no_PCA_update, 'PCA': PCA_update}


def predict_arrival(data: dict[str, pd.DataFrame], model, **kwargs):
    PCA_status = kwargs.get('PCA', True)

    if PCA_status:
        dframe = data['PCA'].copy()
    else:
        dframe = data['no_PCA'].copy()

    forecast = dframe.copy()

    forecast['duration'] = pd.DataFrame(model.predict(forecast, batch_size=512), index=forecast.index)

    # forecast['duration'] = forecast['duration'].apply(lambda x: x + 1 if x >= 0 else 1)

    forecast['update'] = to_datetime_days(forecast['update'])

    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['duration'] = pd.to_timedelta(forecast['duration'], unit='D')

    forecast['arrival'] = forecast['update'] + forecast['duration']
    forecast.drop(columns=dframe.columns, inplace=True)

    forecast['_num'], forecast['update'] = zip(*forecast.index.str.split('_'))

    forecast['update'] = pd.to_datetime(forecast['update'])

    forecast['arrival'] = forecast['arrival'].dt.date

    return forecast.drop(columns=['duration'])
