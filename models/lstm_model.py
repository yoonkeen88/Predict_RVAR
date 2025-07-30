from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

def build_multi_input_lstm(input_shapes):
    inputs = []
    lstm_outputs = []

    for shape in input_shapes:
        input_layer = Input(shape=shape)
        lstm = LSTM(32, activation='tanh')(input_layer)
        inputs.append(input_layer)
        lstm_outputs.append(lstm)

    merged = Concatenate()(lstm_outputs)
    output = Dense(1)(merged)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model