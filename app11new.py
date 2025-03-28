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

# Title
st.title("Bridge Maximum Load Capacity Predictor")

st.write(
    """
    Enter the bridge parameters below.  
    Click 'Predict' to estimate the **Maximum Load Capacity**.
    """
)

# 1) Span_ft: length of the bridge span in feet
span_ft = st.number_input("Span (feet)", value=100.0, min_value=0.0)

# 2) Deck_Width_ft: width of the bridge deck in feet
deck_width_ft = st.number_input("Deck Width (feet)", value=40.0, min_value=0.0)

# 3) Age_Years: age of the bridge in years
age_years = st.number_input("Age (years)", value=25, min_value=0)

# 4) Num_Lanes: number of lanes on the bridge
num_lanes = st.number_input("Number of Lanes", value=2, min_value=1)

# 5) Material: categorical variable indicating primary construction material
material_options = ["Steel", "Concrete", "Composite"]
material_choice = st.selectbox("Material", material_options, index=0)

# Simple numeric encoding (0=Steel, 1=Concrete, 2=Composite).
material_encoded = material_options.index(material_choice)

# 6) Condition_Rating: 1 to 5, with 5 being excellent
condition_rating = st.slider("Condition Rating (1 = Worst, 5 = Best)", 1, 5, 3)

# Button to trigger the prediction
if st.button("Predict"):
    # Prepare the input in the order your model expects:
    # [Span_ft, Deck_Width_ft, Age_Years, Num_Lanes, MaterialEncoded, Condition_Rating]
    input_data = np.array([[
        span_ft,
        deck_width_ft,
        float(age_years),
        float(num_lanes),
        float(material_encoded),
        float(condition_rating)
    ]], dtype=np.float32)

    # Make a prediction
    prediction = model.predict(input_data)

    # Assume the model returns a single numeric value for maximum load capacity
    max_load_capacity = float(prediction[0][0])

    # Display the prediction
    st.write("### Predicted Maximum Load Capacity:")
    st.write(f"{max_load_capacity:.2f} (units depend on your training)")  # e.g. kN, lbs, etc.
