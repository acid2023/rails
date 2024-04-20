import pandas as pd
import random
import pickle
import os
from tabulate import tabulate
from contextlib import contextmanager
import sys
from functools import partial

import keras
import tensorflow as tf
import sklearn.metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from asgiref.sync import async_to_sync

import rails.m_learning.modeling_settings as mds
from rails.data_preparation.preprocessing import get_data, predict_arrival


@contextmanager
def capture_output(callback):
    old_stdout = sys.stdout

    class CustomStdout:
        def __init__(self, callback):
            self.callback = callback
            self.buffer = ''

        def write(self, message):
            self.buffer += message

        def flush(self):
            self.callback(self.buffer)
            self.buffer = ''

    try:
        sys.stdout = CustomStdout(callback)
        yield
    finally:
        sys.stdout = old_stdout


def announce(text, channel_layer) -> None:
    #text = '\n'.join(text)
    async_to_sync(channel_layer.group_send)('view_status', {'type': 'view_status_update', 'message': text})


def get_model_metrics(model: keras.Model | object, X_test: pd.DataFrame, y_test: pd.DataFrame) -> tuple[float, float, float]:

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # FutureWarning: 'squared' is deprecated in version 1.4 and will be removed in 1.6.
    # To calculate the root mean squared error, use the function'root_mean_squared_error'.
    if 'root_mean_squared_error' in dir(sklearn.metrics):
        rmse = sklearn.metrics.root_mean_squared_error(y_test, y_pred)
    else:
        rmse = mean_squared_error(y_test, y_pred, squared=False)
    return mae, mse, rmse


def load_models(channel_layer) -> dict[str, keras.Model]:
    models = {}
    path = mds.MODELS_FOLDER

    for filename in os.listdir(path):
        file = os.path.join(path, filename)
        if filename.endswith('pkl'):
            continue
        if not filename.endswith('h5') and not filename.endswith('keras'):
            continue
        try:
            models[filename] = keras.saving.load_model(file)
        except Exception as e:
            announce(f'Errors when loading model {filename}, {e}', channel_layer)
            continue
    announce(f'total {len(models)} models loaded', channel_layer)
    return models


def create_models(df: pd.DataFrame, channel_layer, **kwargs) -> dict[str, keras.Model]:
    history_cut = kwargs.get('history_cut', '2023-12-01')
    update_cut = mds.DEFAULT_TRAINING_DATE_CUT
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

    data = get_data(df.copy(), update_cut=update_cut, history_cut=history_cut, training=True, update=True)

    PCA_train, PCA_test = data['PCA']
    train, test = data['no_PCA']

    epochs = mds.TF_number_of_epochs
    batch_size = mds.TF_batch_size

    no_PCA_num_features = len(train.columns) - 1
    PCA_num_features = len(PCA_train.columns) - 1

    y_no_PCA = train.pop('target')
    y_PCA = PCA_train.pop('target')
    y_PCA_test = PCA_test.pop('target')
    y_no_PCA_test = test.pop('target')

    PCA_val, PCA_test, y_PCA_val, y_PCA_test = train_test_split(PCA_test, y_PCA_test, train_size=0.5)
    val, test, y_no_PCA_val, y_no_PCA_test = train_test_split(test, y_no_PCA_test, train_size=0.5)

    PCA_models = mds.declare_keras_models(PCA_num_features)
    no_PCA_models = mds.declare_keras_models(no_PCA_num_features)

    keras_list = list(PCA_models.keys())
    path = mds.MODELS_FOLDER
    for name in keras_list:
        announce(f'PCA , {name}', channel_layer)
        X_PCA, _, y_pca, _ = train_test_split(PCA_train, y_PCA, train_size=0.75)
        with capture_output(lambda text: announce(text, channel_layer)):
            PCA_models[name].fit(X_PCA, y_pca, validation_data=(PCA_val, y_PCA_val),
                                 batch_size=batch_size, epochs=epochs, callbacks=[early_stop])
        metrics = get_model_metrics(PCA_models[name], PCA_test, y_PCA_test)
        announce(f' PCA model {name} metrics - MAE: {metrics[0]}, MSE: {metrics[1]}, RMSE: {metrics[2]}', channel_layer)
        PCA_models[name].save(f'{path}/update_PCA_{name}.keras')
        with open(f'{path}/update_PCA_{name}_history.pkl', 'wb') as file:
            pickle.dump(PCA_models[name].history.history, file)
        announce(f'no_PCA , {name}', channel_layer)
        X, _, y, _ = train_test_split(train, y_no_PCA, train_size=0.75)
        with capture_output(lambda text: announce(text, channel_layer)):
            no_PCA_models[name].fit(X, y, validation_data=(val, y_no_PCA_val),
                                    epochs=epochs, batch_size=batch_size, callbacks=[early_stop])
        metrics = get_model_metrics(no_PCA_models[name], test, y_no_PCA_test)
        announce(f' no PCA model {name} metrics - MAE: {metrics[0]}, MSE: {metrics[1]}, RMSE: {metrics[2]}', channel_layer)
        no_PCA_models[name].save(f'{path}/update_no_PCA_{name}.keras')
        with open(f'{path}/update_no_PCA_{name}_history.pkl', 'wb') as file:
            pickle.dump(no_PCA_models[name].history.history, file)


def update_models(df: pd.DataFrame, channel_layer, **kwargs) -> dict[str, keras.Model]:
    models = load_models(channel_layer)
    n_models = len(models)
    history_cut = kwargs.get('history_cut', '2023-06-01')

    update_cut = mds.DEFAULT_TRAINING_DATE_CUT
    early_stop_1 = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    early_stop_2 = keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True, verbose=1)

    data = get_data(df.copy(), update_cut=update_cut, history_cut=history_cut,
                    training=True, diff=False, just_PCA=False, update=True)

    PCA_train, PCA_test = data['PCA']
    train, test = data['no_PCA']

    y_no_PCA = train.pop('target')
    y_PCA = PCA_train.pop('target')
    y_PCA_test = PCA_test.pop('target')
    y_no_PCA_test = test.pop('target')

    PCA_val, PCA_test, y_PCA_val, y_PCA_test = train_test_split(PCA_test, y_PCA_test, train_size=0.5, random_state=42)
    val, test, y_no_PCA_val, y_no_PCA_test = train_test_split(test, y_no_PCA_test, train_size=0.5, random_state=42)

    data_no_update = get_data(df.copy(), update_cut=update_cut, history_cut=history_cut,
                              training=True, diff=False, just_PCA=False, update=False)

    PCA_train_no_update, PCA_test_no_update = data_no_update['PCA']
    train_no_update, test_no_update = data_no_update['no_PCA']

    y_no_PCA_no_update = train_no_update.pop('target')
    y_PCA_no_update = PCA_train_no_update.pop('target')
    y_PCA_test_no_update = PCA_test_no_update.pop('target')
    y_no_PCA_test_no_update = test_no_update.pop('target')

    PCA_val_no_update, PCA_test_no_update, \
        y_PCA_val_no_update, y_PCA_test_no_update = train_test_split(PCA_test_no_update, y_PCA_test_no_update,
                                                                     train_size=0.5, random_state=42)
    val_no_update, test_no_update, \
        y_no_PCA_val_no_update, y_no_PCA_test_no_update = train_test_split(test_no_update, y_no_PCA_test_no_update,
                                                                           train_size=0.5, random_state=42)

    epochs = 75
    batch_size = 256

    path = mds.MODELS_FOLDER

    num_of_updated = 0
    num_of_not_updated = 0
    idx = 1

    for name, model in sorted(models.items(), key=lambda x: random.random()):
        announce(name, channel_layer)
        if 'update' in name:
            if 'no_PCA' in name:
                X, _, y, _ = train_test_split(train, y_no_PCA, train_size=0.75)
                X_val, X_test, y_val, y_test = val, test, y_no_PCA_val, y_no_PCA_test
            else:
                X, _, y, _ = train_test_split(PCA_train, y_PCA, train_size=0.75)
                X_val, X_test, y_val, y_test = PCA_val, PCA_test, y_PCA_val, y_PCA_test
        else:
            if 'no_PCA' in name:
                X, _, y, _ = train_test_split(train_no_update, y_no_PCA_no_update, train_size=0.75)
                X_val, X_test, y_val, y_test = val_no_update, test_no_update, y_no_PCA_val_no_update, y_no_PCA_test_no_update
            else:
                X, _, y, _ = train_test_split(PCA_train_no_update, y_PCA_no_update, train_size=0.75)
                X_val, X_test, y_val, y_test = PCA_val_no_update, PCA_test_no_update, y_PCA_val_no_update, y_PCA_test_no_update

        loss = model.evaluate(X_test, y_test)[0]
        with capture_output(lambda text: announce(text, channel_layer)):
            model.fit(X, y, validation_data=(X_val, y_val), batch_size=batch_size, epochs=epochs,  callbacks=[early_stop_1, early_stop_2])

        new_loss = model.evaluate(X_test, y_test)[0]
        if new_loss < loss:
            model.save(f'{path}/{name}')
            announce(f'{idx} out of {n_models}, model {name}, initial loss: {loss}, new loss: {new_loss} - model saved', channel_layer)
            num_of_updated += 1
        else:
            announce(f'{idx} out of {n_models}, model {name}, initial loss: {loss}, new loss: {new_loss} - model NOT saved', channel_layer)
            num_of_not_updated += 1
        idx += 1
    announce(f'updated {num_of_updated} models, not updated {num_of_not_updated}', channel_layer)


def get_models_list_for_prediction(models: list[keras.Model], df: pd.DataFrame) -> list[str]:
    
    update_cut = mds.DEFAULT_TRAINING_DATE_CUT
    data = get_data(df.copy(), update_cut=update_cut, training=True, diff=False, just_PCA=False, update=True)

    _, PCA_test = data['PCA']
    _, test = data['no_PCA']
    data_no_update = get_data(df.copy(), update_cut=update_cut, training=True, diff=False, just_PCA=False, update=False)
    _, PCA_test_no_update = data_no_update['PCA']
    _, test_no_update = data_no_update['no_PCA']

    y_PCA_test = PCA_test.pop('target')
    y_no_PCA_test = test.pop('target')
    y_PCA_test_no_update = PCA_test_no_update.pop('target')
    y_no_PCA_test_no_update = test_no_update.pop('target')

    metrics = {}
    for name, model in models.items():
        if 'update' in name:
            if 'no_PCA' in name:
                X_test, y_test = test, y_no_PCA_test
            else:
                X_test, y_test = PCA_test, y_PCA_test
        else:
            if 'no_PCA' in name:
                X_test, y_test = test_no_update, y_no_PCA_test_no_update
            else:
                X_test, y_test = PCA_test_no_update, y_PCA_test_no_update

        metrics[name] = model.evaluate(X_test, y_test, batch_size=1024)

    metrics = pd.DataFrame(metrics, index=['LOSS', 'MAE', 'MSE']).T
    list_for_prediction = list(metrics.sort_values('LOSS').head(49).index)
    return list_for_prediction


def prediction(df: pd.DataFrame, channel_layer) -> pd.DataFrame:

    announce('Loading models', channel_layer)
    models = load_models(channel_layer)
    announce('Models loaded', channel_layer)
    announce('Selecting best performing models', channel_layer)
    models_list = get_models_list_for_prediction(models, df.copy())
    prediction_models = {name: models[name] for name in models_list}
    announce('Models selected',channel_layer)

    announce('Making predictions', channel_layer)
    update_cut = mds.DEFAULT_TRAINING_DATE_CUT
    
    data = get_data(df.copy(), update_cut=update_cut, training=False, update=False)
    update_data = get_data(df.copy(), update_cut=update_cut, training=False, diff=False, just_PCA=False, update=True)

    forecasts = {}

    for name, model in prediction_models.items():
        PCA_status = 'no_PCA' not in name
        with capture_output(lambda text: announce(text, channel_layer)):
            if 'update' in name:
                forecasts[name] = predict_arrival(update_data, model, PCA=PCA_status)
            else:
                forecasts[name] = predict_arrival(data, model, PCA=PCA_status)
    announce('Predictions made', channel_layer)

    announce('Compiling forecasts', channel_layer)
    forecast = pd.concat(forecasts.values(), axis=0)
    forecast['update'] = pd.to_datetime(forecast['update'])
    forecast['arrival'] = pd.to_datetime(forecast['arrival'])
    forecast['duration'] = forecast['arrival'] - forecast['update']
    stats = forecast.groupby(['update', '_num'])['duration'].agg('describe')
    stats['update'] = stats.index.get_level_values('update')
    stats['_num'] = stats.index.get_level_values('_num')
    stats['cattles'] = stats.apply(lambda x: 2 if x['_num'].startswith('7790') else 1, axis=1)
    pd_dt = {}
    for idx, row in stats.iterrows():
        pd_dt[idx] = row
    prediction = pd.DataFrame(pd_dt).T.reset_index(drop=True)
    announce('Forecasts compiled', channel_layer)
    return prediction


def show_models_metrics(df: pd.DataFrame, channel_layer) -> None:

    models = load_models(channel_layer)
    update_cut = mds.DEFAULT_TRAINING_DATE_CUT
    history_cut = '2023-06-01'
    data = get_data(df.copy(), update_cut=update_cut, training=True, diff=False, just_PCA=False, update=True)

    _, PCA_test = data['PCA']
    _, test = data['no_PCA']

    y_PCA_test = PCA_test.pop('target')
    y_no_PCA_test = test.pop('target')
    data_no_update = get_data(df.copy(), update_cut=update_cut, training=True, diff=False, just_PCA=False, update=False)
    _, PCA_test_no_update = data_no_update['PCA']
    _, test_no_update = data_no_update['no_PCA']
    
    y_PCA_test_no_update = PCA_test_no_update.pop('target')
    y_no_PCA_test_no_update = test_no_update.pop('target')
    metrics = {}
    i = 1
    for name, model in models.items():
        if 'update' in name:
            if 'no_PCA' in name:
                X_test, y_test = test, y_no_PCA_test
            else:
                X_test, y_test = PCA_test, y_PCA_test
        else:
            if 'no_PCA' in name:
                X_test, y_test = test_no_update, y_no_PCA_test_no_update
            else:
                X_test, y_test = PCA_test_no_update, y_PCA_test_no_update

        # with tf.device('/CPU:0'):
        with capture_output(lambda text: announce(text, channel_layer)):
            metrics[name] = model.evaluate(X_test, y_test, batch_size=4096)
        announce(f'{i} {name} metrics: {metrics[name]}', channel_layer)
        i += 1
    metrica = pd.DataFrame(metrics, index=['LOSS', 'MAE', 'MSE']).T.sort_values('LOSS')
    table = tabulate(metrica, headers='keys', tablefmt='pipe')
    with capture_output(lambda text: announce(text, channel_layer)):
        print(f'resulting scores:\n{table}')
    return metrica


if __name__ == "__main__":
    pass
