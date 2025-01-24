import time
import pandas as pd
import streamlit as st
import joblib

# Load all binary models (MLP removed)
binary_models = {
    "Logistic Regression": joblib.load('./binary_model_logistic_regression.pkl'),
    "Decision Tree": joblib.load('./binary_model_decision_tree.pkl'),
    "Random Forest": joblib.load('./binary_model_random_forest.pkl'),
    "k-NN": joblib.load('./binary_model_knn.pkl'),
    "Gradient Boosting": joblib.load('./binary_model_gradient_boosting.pkl')
}

# Load all multi-class models (MLP removed)
multi_class_models = {
    "Logistic Regression": joblib.load('./multi_class_model_logistic_regression.pkl'),
    "Decision Tree": joblib.load('./multi_class_model_decision_tree.pkl'),
    "Random Forest": joblib.load('./multi_class_model_random_forest.pkl'),
    "k-NN": joblib.load('./multi_class_model_knn.pkl'),
    "Gradient Boosting": joblib.load('./multi_class_model_gradient_boosting.pkl')
}

# Load preprocessor and feature selector
preprocessor = joblib.load('./preprocessor.pkl')
feature_selector = joblib.load('./feature_selector.pkl')

# Load label encoder
label_encoder = joblib.load('./label_encoder.pkl')

# Streamlit App
st.title("CAN Bus Real-Time Classification")

# Dropdown menus for model selection
st.sidebar.title("Model Selection")
binary_model_choice = st.sidebar.selectbox(
    "Choose a Binary Classification Model:",
    list(binary_models.keys())
)

multi_class_model_choice = st.sidebar.selectbox(
    "Choose a Multi-Class Classification Model:",
    list(multi_class_models.keys())
)

# Map dropdown choices to models
binary_model = binary_models[binary_model_choice]
multi_class_model = multi_class_models[multi_class_model_choice]

uploaded_file = st.file_uploader("Upload a Test Dataset (CSV)", type="csv")

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)
    st.write("Uploaded Dataset:")
    st.dataframe(dataset.head())

    # Process records every 5 seconds
    record_index = 0
    st.write("Processing records in real-time...")

    while record_index < len(dataset):
        record = dataset.iloc[record_index]
        record_df = pd.DataFrame([record])

        # Preprocessing
        start_time = time.time()
        preprocessed_record = preprocessor.transform(record_df)
        selected_features = feature_selector.transform(preprocessed_record)

        # Binary Classification
        binary_prediction = binary_model.predict(selected_features)[0]

        if binary_prediction == 0:  # Normal
            st.markdown(
                "<div style='background-color: green; color: white; padding: 10px;'>Normal</div>",
                unsafe_allow_html=True,
            )
            attack_type = "N/A"  # No attack
        else:  # Attack
            st.markdown(
                "<div style='background-color: red; color: white; padding: 10px;'>Attack</div>",
                unsafe_allow_html=True,
            )
            # Multi-class Classification
            multi_class_prediction = multi_class_model.predict(selected_features)[0]
            attack_type = label_encoder.inverse_transform([multi_class_prediction])[0]

        end_time = time.time()
        execution_time = end_time - start_time

        # Display results
        st.write(f"Record {record_index + 1}:")
        st.write(record)
        st.write(f"Prediction: {'Normal' if binary_prediction == 0 else 'Attack'}")
        st.write(f"Type of Attack: {attack_type}")
        st.write(f"Execution Time: {execution_time:.2f} seconds")
        st.write("---")

        record_index += 1
        time.sleep(5)  # Simulate real-time processing every 5 seconds

    st.success("Classification complete!")

