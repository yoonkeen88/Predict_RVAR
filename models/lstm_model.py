from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate, Dropout

def build_multi_input_lstm1(input_shapes, units=[40, 40, 40, 16], loss_function='mse'):
    inputs = []
    lstm_outputs = []

    for i, shape in enumerate(input_shapes):
        input_layer = Input(shape=shape)
        # Use the corresponding unit size for each input group
        unit = units[i] if isinstance(units, list) and len(units) == len(input_shapes) else units
        lstm = LSTM(units=unit, activation='tanh')(input_layer)
        dropout = Dropout(0.3)(lstm)
        inputs.append(input_layer)
        lstm_outputs.append(dropout)

    merged = Concatenate()(lstm_outputs)
    output = Dense(1)(merged)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])
    return model

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])
    return model

def build_lstm(input_shape, loss_function='mse'):
    """
    Builds a stacked LSTM model.
    """
    input_layer = Input(shape=input_shape)
    lstm1 = LSTM(units=50, activation='tanh', return_sequences=True)(input_layer)
    dropout1 = Dropout(0.2)(lstm1)
    lstm2 = LSTM(units=50, activation='tanh')(dropout1)
    dropout2 = Dropout(0.2)(lstm2)
    output = Dense(1)(dropout2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss=loss_function, metrics=['mae'])
    return model
