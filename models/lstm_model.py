from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate

def build_multi_input_lstm1(input_shapes, unit =40):
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
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

    return model

def build_lstm(input_shape):
    """
    Builds a stacked LSTM model.
    """
    input_layer = Input(shape=input_shape)

    # 첫 번째 LSTM 레이어
    # return_sequences=True는 다음 LSTM 레이어로 출력을 전달하기 위해 필요합니다.
    lstm1 = LSTM(units=50, activation='tanh', return_sequences=True)(input_layer)

    # 두 번째 LSTM 레이어
    lstm2 = LSTM(units=50, activation='tanh')(lstm1)

    # 출력 레이어
    output = Dense(1)(lstm2)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model