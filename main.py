import streamlit as st
import tensorflow as tf
import pandas as pd

# Load the pre-trained model

# Define the NBeatsBlock layer
class NBeatsBlock(tf.keras.layers.Layer):
    def __init__(self, input_size: int, theta_size: int, horizon: int, n_neurons: int, n_layers: int, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.theta_size = theta_size
        self.horizon = horizon
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
        self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

    def call(self, inputs):
        x = inputs
        for layer in self.hidden:
            x = layer(x)
        theta = self.theta_layer(x)
        backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
        return backcast, forecast
reconstructed_model = load_model(
    "forecaste.keras",
    custom_objects={"NBeatsBlock": NBeatsBlock}
)
# Streamlit app code
def main():
    st.title("Forecasting App")

    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        data = data.rename(columns={'Dates': 'Date', 'ACTUAL (mm) ': 'Rainfall'})

        input_data = data["Rainfall"].tolist()
        input = input_data[-7:]

        forecast = reconstructed_model.predict(tf.expand_dims(input, axis=0))
        value = tf.squeeze(forecast).numpy()

        if value < 0:
            value = 0

        st.write("Forecasted Value:", value)

if __name__ == "__main__":
    main()
