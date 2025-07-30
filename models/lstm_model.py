from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

def build_multi_input_lstm1(input_shapes, unit=40, loss_function='mse'):
    inputs = []
    lstm_outputs = []

    for shape in input_shapes:
        input_layer = Input(shape=shape)
        lstm = LSTM(units=unit, activation='tanh')(input_layer)
        inputs.append(input_layer)
        lstm_outputs.append(lstm)

    merged = Concatenate()(lstm_outputs)
    output = Dense(1)(merged)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])
    return model

def build_lstm(input_shape, loss_function='mse'):
    """
    Builds a stacked LSTM model.
    """
    input_layer = Input(shape=input_shape)
    lstm1 = LSTM(units=50, activation='tanh', return_sequences=True)(input_layer)
    lstm2 = LSTM(units=50, activation='tanh')(lstm1)
    output = Dense(1)(lstm2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])
    return model
