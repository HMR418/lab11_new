import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Define a custom MSE function 
#    (this will work in virtually any tf.keras version).
@tf.keras.utils.register_keras_serializable()
def custom_mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

# 2. Load your model using the custom MSE function
model = tf.keras.models.load_model(
    "tf_bridge_model.h5",
    custom_objects={"mse": custom_mse}
)

# 3. Set up the application title
st.title("Simple Bridge Model Demo")

st.write(
    """
    This is a simple interactive interface for the loaded Keras model.
    Enter some input values in the fields below and click 'Predict'
    to see the model's output.
    """
)

# 4. Collect user inputs (adjust as needed for your model’s input shape).
input_1 = st.number_input("Input Feature 1", value=0.0)
input_2 = st.number_input("Input Feature 2", value=0.0)

# 5. Create a 'Predict' button. When clicked, it will run the model prediction.
if st.button("Predict"):
    # Prepare the inputs for the model.
    input_data = np.array([[input_1, input_2]], dtype=np.float32)

    # 6. Make a prediction
    prediction = model.predict(input_data)

    # 7. Display the result
    st.write("**Model Output**:", prediction)

