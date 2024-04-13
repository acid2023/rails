from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, Lars, LassoLars, OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge, ARDRegression, PassiveAggressiveRegressor, RANSACRegressor
from sklearn.tree import DecisionTreeRegressor
# from sklearn.base import BaseEstimator, RegressorMixin
# from sklearn.model_selection import train_test_split

import keras
from keras import layers
import tensorflow as tf
import keras.backend as K

MODELS_FOLDER = 'rails/m_learning/models'

DEFAULT_TRAINING_DATE_CUT = '2024-01-15'

DEFAULT_COLUMNS = ['d_Left', 'lat', 'lon', 'update']

TF_DEFAULT_COLUMNS = ['d_Left', 'lat', 'lon', 'update']

TF_number_of_epochs = 100
TF_batch_size = 256


def declare_keras_models(num_features):
    TF_neurons = 512
    TF_learning_rate = 0.001

    class Expand(keras.layers.Layer):
        def call(self, x: tf.Tensor, axis: int) -> tf.Tensor:
            return tf.expand_dims(x, axis=axis)

    class Squeeze(keras.layers.Layer):
        def call(self, x: tf.Tensor, axis: int) -> tf.Tensor:
            return tf.squeeze(x, axis=axis)

    class Squash(keras.layers.Layer):
        def call(self, inputs):
            squared_norm = tf.reduce_sum(tf.square(inputs), axis=-1, keepdims=True)
            scale = squared_norm / (1 + squared_norm)
            result = scale * inputs / tf.sqrt(squared_norm + K.epsilon())
            return result

    def TensorFlow_Relu_Elu_Selu_Nadam():
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='relu', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='elu'),
                                  layers.Dropout(0.2),
                                  layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='selu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(1, activation='relu')])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def TensorFlow_Softplus_Nadam():
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='softplus', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dropout(0.2),
                                  layers.Dense(TF_neurons, activation='softplus'),
                                  layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='softplus'),
                                  layers.Dense(1, activation='relu')])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def TensorFlow_Synthetic():
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(TF_neurons, activation='relu', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dropout(0.2),
                                  layers.Dense(TF_neurons * 2, activation='softplus'),
                                  layers.BatchNormalization(),
                                  layers.Dropout(0.2),
                                  layers.Dense(TF_neurons * 2, activation='relu'),
                                  layers.Dense(1, activation='relu')])
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_1():
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(192, activation='selu', input_shape=(None, num_features)),
                                  layers.BatchNormalization(),
                                  layers.Dense(160, activation='selu'),
                                  layers.Dropout(0.2),
                                  layers.Dense(224, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(96, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(96, activation='softplus'),
                                  layers.Dense(1, activation='relu')])
        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_2():
        model = keras.Sequential([layers.BatchNormalization(),
                                  layers.Dense(288, activation='relu', input_shape=(None, num_features)),
                                  layers.Dropout(0.2),
                                  layers.Dense(416, activation='relu'),
                                  layers.Dropout(0.2),
                                  layers.Dense(160, activation='selu'),
                                  layers.Dense(256, activation='softplus'),
                                  layers.BatchNormalization(),
                                  layers.Dense(256, activation='softplus'),
                                  layers.Dense(320, activation='softplus'),
                                  layers.Dropout(0.2),
                                  layers.Dense(320, activation='relu'),
                                  layers.BatchNormalization(),
                                  layers.Dense(320, activation='softplus'),
                                  layers.Dense(1, activation='relu')])
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_1():
        model = keras.Sequential()
        input_shape = (num_features,)
        model.add(keras.layers.Input(shape=input_shape))
        model.add(keras.layers.Dense(units=34, activation='relu'))
        model.add(keras.layers.Reshape((34, 1)))

        query = keras.layers.Input(shape=(None, 34))
        value = keras.layers.Input(shape=(None, 34))
        attention = keras.layers.Attention()
        query_transformed = keras.layers.Dense(48, activation='sigmoid')(query)
        value_transformed = keras.layers.Dense(48, activation='sigmoid')(value)
        attended_values = attention([query_transformed, value_transformed])
        attended_values = keras.layers.Input(tensor=attended_values, name='attended_values')
        model.add(attended_values)
        model.add(layers.BatchNormalization(),)
        model.add(keras.layers.Conv1D(filters=18, kernel_size=3, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Conv1D(filters=36, kernel_size=5, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.MaxPooling1D(pool_size=3))
        model.add(layers.BatchNormalization(),)
        model.add(keras.layers.Conv1D(filters=54, kernel_size=3, activation='tanh'))
        model.add(keras.layers.BatchNormalization())

        model.add(keras.layers.MaxPooling1D(pool_size=2))

        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(units=96, activation='relu'))
        model.add(keras.layers.Dropout(0.1))

        model.add(keras.layers.Dense(1, activation='relu'))

        optimizer = keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_2():
        input = keras.layers.Input(shape=(None, num_features))
        x = keras.layers.BatchNormalization()(input)
        x = Expand()(x, axis=2)
        x = Expand()(x, axis=1)
        x = keras.layers.Conv1D(filters=12, kernel_size=1, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=12, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Dropout(0.34849147081255716)(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.Dense(units=96, activation='tanh')(x)
        output = keras.layers.Dense(1, activation='relu')(x)

        optimizer = keras.optimizers.RMSprop(learning_rate=TF_learning_rate)
        model = keras.models.Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_3():
        inputs = keras.layers.Input(shape=(None, num_features))
        x1 = keras.layers.Dense(32, activation='relu')(inputs)
        reshaped_inputs = keras.layers.Reshape((2, 2, 8))(x1)
        conv_layer = keras.layers.Conv1D(filters=32, kernel_size=2, activation='relu')(reshaped_inputs)
        conv_layer = keras.layers.Flatten()(conv_layer)
        conv_layer = keras.layers.BatchNormalization()(conv_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(conv_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        dense_layer = keras.layers.Flatten()(dense_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        dense_layer = keras.layers.Dense(units=288, activation='relu')(dense_layer)
        outputs = keras.layers.Dense(units=1, activation='relu')(dense_layer)
        optimizer = keras.optimizers.RMSprop(learning_rate=TF_learning_rate)
        model = keras.models.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_4():
        model = keras.Sequential()
        model.add(keras.layers.Conv1D(64, kernel_size=2, activation='tanh', input_shape=(num_features, 1)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv1D(64, kernel_size=2, activation='tanh'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv1D(64, kernel_size=2, activation='tanh'))
        model.add(keras.layers.MaxPooling1D(pool_size=2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(units=128, activation='sigmoid'))
        model.add(keras.layers.Dense(units=1, activation='relu'))
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_1Flat():
        input = keras.layers.Input(shape=(None, num_features))
        x = keras.layers.BatchNormalization()(input)
        x = Expand()(x, axis=2)
        x = Expand()(x, axis=1)
        x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='elu')(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=1, activation='selu')(x)
        x = keras.layers.Dropout(0.44417634318940014)(x)
        x = Squeeze()(x, axis=1)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='elu')(x)
        x = keras.layers.Conv1DTranspose(filters=32, kernel_size=1, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='softplus')(x)
        x = keras.layers.Dense(units=128, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='selu')(x)
        output = keras.layers.Dense(1, activation='relu')(x)

        model = keras.models.Model(inputs=input, outputs=output)
        optimizer = keras.optimizers.Adam(learning_rate=TF_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_KeraTune_Conv_2Flat():
        input = keras.layers.Input(shape=(None, num_features))
        x = keras.layers.BatchNormalization()(input)
        x = Expand()(x, axis=2)
        x = Expand()(x, axis=1)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Dropout(0.34849147081255716)(x)
        x = keras.layers.Conv1D(filters=32, kernel_size=1, activation='tanh')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.Dense(units=64, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=96, activation='tanh')(x)
        output = keras.layers.Dense(1, activation='relu')(x)
        optimizer = keras.optimizers.RMSprop(learning_rate=TF_learning_rate)

        model = keras.models.Model(inputs=input, outputs=output)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_3Flat():
        num_conv_layers = 4
        num_dropouts = 1
        dropout_rate = 0.1671256121594838
        num_dense_layers = 2
        l1_regularization = 0.1671256121594838
        l2_regularization = 0.29732598490943557

        input = keras.layers.Input(shape=(num_features))
        x = keras.layers.BatchNormalization()(input)
        x = Expand()(x, axis=2)
        x = Expand()(x, axis=1)

        # Add convolutional layers
        for _ in range(num_conv_layers):
            x = keras.layers.Conv1D(filters=32, kernel_size=1)(x)
            x = keras.layers.Activation('selu')(x)
            x = keras.layers.Dropout(0.30713858568974556)(x)
            x = keras.layers.ActivityRegularization(l1=l1_regularization, l2=l2_regularization)(x)

        # Add dropout layers
        for _ in range(num_dropouts):
            x = keras.layers.Dropout(dropout_rate)(x)

        x = Squeeze()(x, axis=1)

        # Add transpose convolutional layers
        for _ in range(num_conv_layers):
            x = keras.layers.Conv1DTranspose(96, kernel_size=1)(x)
            x = keras.layers.Activation('relu')(x)

        x = keras.layers.Flatten()(x)

        for _ in range(num_dense_layers):
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dense(128)(x)
            x = keras.layers.Activation('selu')(x)
            x = keras.layers.Dropout(0.30582557931042254)(x)
            x = keras.layers.ActivityRegularization(l1=l1_regularization, l2=l2_regularization)(x)

        # Add remaining layers
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(128)(x)
        x = keras.layers.Activation('selu')(x)
        output = keras.layers.Dense(1, activation='relu')(x)

        model = keras.models.Model(inputs=input, outputs=output)

        optimizer = keras.optimizers.Nadam(learning_rate=TF_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_4Flat():
        input = keras.layers.Input(shape=(num_features))
        x = Expand()(input, axis=2)
        x = keras.layers.SeparableConv1D(filters=32, kernel_size=1, activation='softplus')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.SeparableConv1D(filters=8, kernel_size=2, activation='tanh')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=2, activation='selu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=2, activation='softplus')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='selu')(x)
        x = keras.layers.Dropout(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=1, activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='selu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='selu')(x)
        output = keras.layers.Dense(1, activation='relu')(x)
        model = keras.models.Model(inputs=input, outputs=output)
        optimizer = keras.optimizers.Nadam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_5Flat():
        input = keras.layers.Input(shape=(None, num_features))
        x = keras.layers.BatchNormalization()(input)
        x = Expand()(x, axis=2)
        x = keras.layers.SeparableConv1D(filters=8, kernel_size=3, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv1D(filters=32, kernel_size=2, activation='selu')(x)
        x = keras.layers.SpatialDropout1D(0.2)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=24, kernel_size=2, activation='tanh')(x)
        x = keras.layers.SpatialDropout1D(0.4)(x)
        x = keras.layers.SimpleRNN(units=64, return_sequences=True)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(units=64, return_sequences=True)(x)
        x = keras.layers.SpatialDropout1D(0.2)(x)
        x = keras.layers.Conv1D(filters=16, kernel_size=2, activation='relu')(x)
        x = keras.layers.SpatialDropout1D(0.4)(x)
        x = keras.layers.SpatialDropout1D(0.4)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=2, activation='tanh')(x)
        x = keras.layers.Dropout(0.4)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=256, activation='tanh')(x)
        x = keras.layers.Dense(units=128, activation='elu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=128, activation='relu')(x)
        output = keras.layers.Dense(1, activation='relu')(x)
        model = keras.models.Model(inputs=input, outputs=output)
        optimizer = keras.optimizers.RMSprop(learning_rate=TF_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_KeraTune_Conv_6Flat():
        input = keras.layers.Input(shape=(num_features))

        x = keras.layers.BatchNormalization()(input)
        x = Expand()(x, axis=2)
        x = Expand()(x, axis=2)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ConvLSTM1D(filters=64, kernel_size=1)(x)
        x = keras.layers.SpatialDropout1D(rate=0.39476)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.SeparableConv1D(filters=64, kernel_size=1, activation='tanh')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')(x)
        x = keras.layers.SpatialDropout1D(rate=0.20179)(x)
        x = keras.layers.SimpleRNN(units=32, return_sequences=True)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.LSTM(units=16, return_sequences=True)(x)
        x = keras.layers.SpatialDropout1D(0.068672)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv1DTranspose(filters=24, kernel_size=5, activation='softplus')(x)
        x = keras.layers.Dropout(0.43681)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units=64, activation='sigmoid')(x)
        x = keras.layers.Dense(units=512, activation='sigmoid')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(units=64, activation='softplus')(x)

        output = keras.layers.Dense(1, activation='relu')(x)

        model = keras.models.Model(inputs=input, outputs=output)

        optimizer = keras.optimizers.RMSprop(learning_rate=TF_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def TensorFlow_resnet():
        input_shape = (None, num_features)
        input = keras.layers.Input(shape=input_shape)
        input = keras.layers.Reshape((num_features, 1))(input)
        filters = 64

        def resnet_block(x, filters, kernel_szie=3, stride=1):
            y = keras.layers.Conv1D(filters, kernel_size=kernel_szie, strides=stride, padding='same')(x)
            y = keras.layers.BatchNormalization()(y)
            y = keras.layers.Activation('relu')(y)
            y = keras.layers.Conv1D(filters, kernel_size=kernel_szie, strides=1, padding='same')(y)
            y = keras.layers.BatchNormalization()(y)

            if stride != 1:
                x = keras.layers.Conv1D(filters, kernel_size=1, strides=stride, padding='same')(x)

            out = keras.layers.Add()([x, y])
            out = keras.layers.Activation('relu')(out)
            return out

        x = keras.layers.Conv1D(filters, kernel_size=7, strides=2, padding='same')(input)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        for _ in range(7):
            x = resnet_block(x, filters=filters)

        x = keras.layers.GlobalAveragePooling1D()(x)

        x = keras.layers.Dense(512, activation='relu')(x)
        x = keras.layers.Dropout(0.15)(x)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dropout(0.15)(x)

        output = keras.layers.Dense(1, activation='relu')(x)
        model = keras.models.Model(inputs=input, outputs=output)

        hp_learning_rate = 0.001
        optimizer = keras.optimizers.Nadam(learning_rate=hp_learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_unet():
        inputs = keras.layers.Input(shape=(num_features, 1))
        conv1 = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(inputs)

        conv1 = keras.layers.BatchNormalization()(conv1)
        pool1 = keras.layers.MaxPooling1D(pool_size=2)(conv1)
        conv2 = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(pool1)

        conv2 = keras.layers.BatchNormalization()(conv2)
        pool2 = keras.layers.MaxPooling1D(pool_size=2)(conv2)
        conv3 = keras.layers.Conv1D(128, 3, activation='relu', padding='same')(pool2)
        conv3 = keras.layers.BatchNormalization()(conv3)

        up1 = keras.layers.UpSampling1D(size=2)(conv3)
        concat1 = keras.layers.Concatenate(axis=1)([conv2, up1])
        conv4 = keras.layers.Conv1D(64, 3, activation='relu', padding='same')(concat1)
        conv4 = keras.layers.BatchNormalization()(conv4)

        up2 = keras.layers.UpSampling1D(size=2)(conv4)
        concat2 = keras.layers.Concatenate(axis=1)([conv1, up2])

        conv5 = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(concat2)
        conv5 = keras.layers.BatchNormalization()(conv5)

        x = conv5
        x = keras.layers.Attention()([x, x])

        x = keras.layers.Flatten()(x)

        x = keras.layers.Dense(500, activation='relu')(conv5)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(250, activation='relu')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Dense(100, activation='relu')(x)

        x = keras.layers.Attention()([x, x])

        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.5)(x)
        x = keras.layers.BatchNormalization()(x)
        outputs = keras.layers.Dense(1, activation='relu')(x)

        model = keras.models.Model(inputs=inputs, outputs=outputs)

        optimizer = keras.optimizers.Nadam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])

        return model

    def TensorFlow_Transformer():
        def transformer_model(input_shape, num_blocks, d_model, num_heads, dff, dropout_rate):

            inputs = keras.layers.Input(shape=input_shape)

            x = inputs

            position = keras.layers.Embedding(input_dim=input_shape[0], output_dim=d_model)(tf.range(input_shape[0], dtype=tf.float32))
            x += position

            # Transformer Encoder
            for _ in range(num_blocks):
                x = transformer_encoder_block(x, num_heads, d_model, dff, dropout_rate)
            # Final Dense layer for regression output

            x = keras.layers.Flatten()(x)
            x = keras.layers.Dense(units=1, activation='relu')(x)

            # Create the model
            model = keras.models.Model(inputs=inputs, outputs=x)

            return model

        def transformer_encoder_block(inputs, num_heads, d_model, dff, dropout_rate):
            x = inputs

            # Multi-head self-attention layer
            x = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)

            # Feed-forward neural network layer
            ffn = tf.keras.Sequential([
                keras.layers.Dense(dff, activation='relu'),
                keras.layers.Dense(d_model)
            ])
            x = ffn(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            x = keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)

            return x

        # Specify the input shape and model parameters
        input_shape = (num_features, 1)  # Input shape: (sequence_length, input_dim)
        num_blocks = 2
        d_model = 64
        num_heads = 4
        dff = 128
        dropout_rate = 0.1

        # Create the Transformer model
        model = transformer_model(input_shape, num_blocks, d_model, num_heads, dff, dropout_rate)

        # Compile the model
        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

        return model

    def TensorFlow_capsule():
        def primary_capsule(inputs, capsules_dim, capsules_num):
            # Apply Conv1D to extract features from input
            conv = keras.layers.Conv1D(filters=256, kernel_size=1, strides=1, padding='valid', activation='relu')(inputs)

            # Reshape the output to match the capsules structure
            reshaped = keras.layers.Reshape(target_shape=(-1, capsules_dim))(conv)

            # Apply Squash activation to normalize the output
            squashed = Squash()(reshaped)

            # Convert the output to capsules
            capsules = keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1), output_shape=(None, 256, 8))(squashed)

            return capsules

        def digit_capsule(capsules, iterations):
            # Transform the input capsule dimensions
            transformed = keras.layers.Dense(units=256, activation='relu')(capsules)

            # Apply routing algorithm
            for _ in range(iterations):
                route = keras.layers.Softmax(axis=2)(transformed)
                weighted_sum = keras.layers.Dot(axes=(2, 2))([route, transformed])
                squashed = Squash()(weighted_sum)
                transformed = squashed

            return squashed

        # Define the Capsule Network model
        def capsule_network(input_shape, capsules_dim=8, capsules_num=32, iterations=3):
            inputs = keras.layers.Input(shape=input_shape)

            # Create Primary Capsule layer
            primary_caps = primary_capsule(inputs, capsules_dim, capsules_num)

            # Create Digit Capsule layer
            digit_caps = digit_capsule(primary_caps, iterations)

            # Classify the digit capsules

            digit_caps = keras.layers.Flatten()(digit_caps)

            classes = keras.layers.Dense(units=1, activation='relu', input_shape=(capsules_num, capsules_dim))(digit_caps)

            # Create the model
            model = keras.models.Model(inputs=inputs, outputs=classes)

            return model

        # Specify the input shape and number of classes
        input_shape = (num_features, 1)  # Input shape: (sequence_length, input_dim)

        # Create the Capsule Network model
        model = capsule_network(input_shape)

        optimizer = keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

        return model

    def TensorFlow_wavenet():
        inputs = keras.layers.Input(shape=(num_features, 1))

        num_blocks = 3
        num_layers_per_block = 10
        num_filters = 64

        x = keras.layers.Conv1D(num_filters, 1, padding='causal', activation='relu')(inputs)

        for _ in range(num_blocks):
            for _ in range(num_layers_per_block):
                # Dilated Convolution with exponentially increasing dilation rate
                x = keras.layers.Conv1D(num_filters, 2, dilation_rate=2**_, padding='causal', activation='relu')(x)

        outputs = keras.layers.Flatten()(x)

        outputs = keras.layers.Dense(1, activation='relu')(outputs)

        model = keras.models.Model(inputs, outputs)

        optimizer = keras.optimizers.Adam(learning_rate=0.001)

        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae', 'mse'])

        return model

    keras_models = {
        'TensorFlow_Synthetic': TensorFlow_Synthetic(),
        'TensorFlow_Softplus_Nadam': TensorFlow_Softplus_Nadam(),
        'TensorFlow_Relu_Elu_Selu_Nadam': TensorFlow_Relu_Elu_Selu_Nadam(),
        'TensorFlow_KeraTune_KeraTune_1': TensorFlow_KeraTune_1(),
        'TensorFlow_KeraTune_KeraTune_2': TensorFlow_KeraTune_2(),
        # 'TensorFlow_KeraTune_Conv_1': TensorFlow_KeraTune_Conv_1(),
        # 'TensorFlow_KeraTune_Conv_2': TensorFlow_KeraTune_Conv_2(),
        # 'TensorFlow_KeraTune_Conv_3': TensorFlow_KeraTune_Conv_3(),
        # 'TensorFlow_KeraTune_Conv_4': TensorFlow_KeraTune_Conv_4(),
        # 'TensorFlow_KeraTune_Conv_1Flat': TensorFlow_KeraTune_Conv_1Flat(),
        # 'TensorFlow_KeraTune_Conv_2Flat': TensorFlow_KeraTune_Conv_2Flat(),
        # 'TensorFlow_KeraTune_Conv_3Flat': TensorFlow_KeraTune_Conv_3Flat(),
        # 'TensorFlow_KeraTune_Conv_4Flat': TensorFlow_KeraTune_Conv_4Flat(),
        'TensorFlow_resnet': TensorFlow_resnet(),
        'TensorFlow_unet': TensorFlow_unet(),
        'TensorFlow_Transformer': TensorFlow_Transformer(),
        # 'TensorFlow_KeraTune_Conv_5Flat': TensorFlow_KeraTune_Conv_5Flat(),
        # 'TensorFlow_KeraTune_Conv_6Flat': TensorFlow_KeraTune_Conv_6Flat(),
        # 'TensorFlow_capsule': TensorFlow_capsule(),
        'TensorFlow_wavenet': TensorFlow_wavenet(),
                    }

    return keras_models


PCA_models = {
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=1000),
        'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=300, max_depth=300),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Lars': Lars(n_nonzero_coefs=10),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'RANSACRegressor': RANSACRegressor(),
        'ElasticNet': ElasticNet(),
        'LassoLars': LassoLars(),
        'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=500)}

no_PCA_models = {
        'RandomForest': RandomForestRegressor(n_estimators=300, random_state=42, max_depth=1000),
        'DecisionTree': DecisionTreeRegressor(max_depth=1000, random_state=42),
        'KNeighbors': KNeighborsRegressor(n_neighbors=5),
        'ExtraTrees': ExtraTreesRegressor(random_state=42, n_estimators=300, max_depth=300),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=500, learning_rate=0.01, random_state=42),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=1.0),
        'Lars': Lars(n_nonzero_coefs=10),
        'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit(),
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),
        'PassiveAggressiveRegressor': PassiveAggressiveRegressor(),
        'RANSACRegressor': RANSACRegressor(),
        'ElasticNet': ElasticNet(),
        'LassoLars': LassoLars(),
        'AdaBoost': AdaBoostRegressor(random_state=42, n_estimators=500)}


models = list(PCA_models.keys())
sklearn_list = list(PCA_models.keys())
