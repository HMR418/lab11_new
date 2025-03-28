import streamlit as st
import tensorflow as tf
import numpy as np

# 1. Load your saved model (ensure tf_bridge_model.h5 is in the same folder or provide the correct path)
model = tf.keras.models.load_model("tf_bridge_model.h5")

# 2. Set up the application title
st.title("Simple Bridge Model Demo")

st.write(
    """
    This is a simple interactive interface for the loaded Keras model.
    Enter some input values in the fields below and click 'Predict'
    to see the model's output.
    """
)

# 3. Collect user inputs. 
#    (Adjust these input widgets according to the real inputs your model expects.)
input_1 = st.number_input("Input Feature 1", value=0.0)
input_2 = st.number_input("Input Feature 2", value=0.0)

# 4. Create a 'Predict' button. When clicked, it will run the model prediction.
if st.button("Predict"):
    # Prepare the inputs for the model. This may vary depending on your model’s input shape.
    # Here, we assume the model expects a 2D array with 2 features per sample.
    input_data = np.array([[input_1, input_2]], dtype=np.float32)

    # 5. Make a prediction
    prediction = model.predict(input_data)

    # 6. Display the result
    st.write("**Model Output**:", prediction)

