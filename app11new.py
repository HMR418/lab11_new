import streamlit as st
import tensorflow as tf
import numpy as np

# If your model references 'mse' as a metric or loss, define it here:
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# Load the saved model, supplying a custom_objects dict 
# so Keras can properly resolve 'mse' if necessary.
model = tf.keras.models.load_model(
    "tf_bridge_model.h5",
    custom_objects={"mse": custom_mse}  # remove if not needed
)

# Set up the Streamlit app
st.title("Bridge Maximum Load Capacity Predictor")

st.write(
    """
    Enter the parameters for your bridge design below.
    Click 'Predict' to estimate the **Maximum Load Capacity**.
    """
)

# Collect user inputs (these are just placeholders—replace with 
# whatever input features your model actually needs, e.g. span length, 
# deck width, material strength, etc.).
span_length = st.number_input("Span Length (meters)", value=30.0)
deck_width = st.number_input("Deck Width (meters)", value=10.0)
material_strength = st.number_input("Material Strength (MPa)", value=30.0)
number_of_girders = st.number_input("Number of Girders", value=4)

# Button to trigger prediction
if st.button("Predict"):
    # Prepare input data. Adjust shape/dtype according to your model's needs.
    # For example, if your model expects 4 features in a single row:
    input_data = np.array([[span_length, deck_width, material_strength, number_of_girders]], dtype=np.float32)

    # Get prediction from the model
    prediction = model.predict(input_data)

    # The model may output a 1D array or single value for capacity.
    # We'll assume it's a single value for maximum load capacity.
    max_load_capacity = prediction[0][0]

    # Display the result
    st.write("### Predicted Maximum Load Capacity:")
    st.write(f"{max_load_capacity:.2f} kN")  # Adjust units as appropriate
