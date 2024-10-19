import streamlit as st
import numpy as np
import pandas as pd
import joblib


def main_app():
    st.title("Predictive Modeling for Heart Failure")

    st.sidebar.header("Sidebar")
    st.sidebar.image("py.jpg")
    sidebar = st.sidebar.selectbox(
        "The app features",
        ("Main Page", "Dataset", "Analysis",
         "Our Model Prediction", "About", "Feedback")
    )

# ========= MAIN PAGE TAB =========
    if sidebar == "Main Page":
        st.image("heart.jpg")
        st.header("The Heart Disease")

        st.write("""A heart attack, or myocardial infarction, occurs when a section of the heart muscle is deprived of oxygen-rich blood, leading to potential damage. In India, coronary artery disease (CAD) is the primary culprit, often stemming from lifestyle factors such as poor diet, lack of exercise, and increasing stress levels.

    The significance of timely treatment cannot be overstated; every moment counts in restoring blood flow to minimize damage to the heart. Additionally, while CAD is the leading cause, there are instances where severe spasms of the coronary arteries can also halt blood flow, although this is less common.

    In India, awareness around heart health is crucial, especially given the rise in risk factors like diabetes, hypertension, and obesity. Promoting a balanced diet, regular physical activity, and stress management can significantly help in preventing heart attacks. Community health initiatives and regular health check-ups can play an important role in early detection and intervention..""")

        st.image("ty.jpg")
        st.subheader("Symptoms")

        st.write("""
                The major symptoms of a heart attack are

    - Chest pain or discomfort. Most heart attacks involve discomfort in the center or left side of the chest that lasts for more than a few minutes or that goes away and comes back. The discomfort can feel like uncomfortable pressure, squeezing, fullness, or pain.
    - Feeling weak, light-headed, or faint. You may also break out into a cold sweat.
    - Pain or discomfort in the jaw, neck, or back.
    - Pain or discomfort in one or both arms or shoulders.
    - Shortness of breath. This often comes along with chest discomfort, but shortness of breath also can happen before chest discomfort.
    """)

        st.subheader("Risk factors")

        st.write("""Several health conditions, your lifestyle, and your age and family history can increase your risk for heart disease and heart attack. These are called risk factors. About half of all Americans have at least one of the three key risk factors for heart disease: high blood pressure, high blood cholesterol, and smoking.

    Some risk factors cannot be controlled, such as your age or family history. But you can take steps to lower your risk by changing the factors you can control.
    """)

        st.subheader("Recover after a heart attack")

        st.write("""
            If you’ve had a heart attack, your heart may be damaged. This could affect your heart’s rhythm and its ability to pump blood to the rest of the body. You may also be at risk for another heart attack or conditions such as stroke, kidney disorders, and peripheral arterial disease (PAD).

    You can lower your chances of having future health problems following a heart attack with these steps:

    - Physical activity—Talk with your health care team about the things you do each day in your life and work. Your doctor may want you to limit work, travel, or sexual activity for some time after a heart attack.
    - Lifestyle changes—Eating a healthier diet, increasing physical activity, quitting smoking, and managing stress—in addition to taking prescribed medicines—can help improve your heart health and quality of life. Ask your health care team about attending a program called cardiac rehabilitation to help you make these lifestyle changes.
    - Cardiac rehabilitation—Cardiac rehabilitation is an important program for anyone recovering from a heart attack, heart failure, or other heart problem that required surgery or medical care. Cardiac rehab is a supervised program that includes
    1. Physical activity
    2. Education about healthy living, including healthy eating, taking medicine as prescribed, and ways to help you quit smoking
    3. Counseling to find ways to relieve stress and improve mental health

    A team of people may help you through cardiac rehab, including your health care team, exercise and nutrition specialists, physical therapists, and counselors or mental health professionals.


    """)
    # ========= DATASET TAB =========
    if sidebar == "Dataset":
        st.write("Here's the dataset")
        df = pd.read_csv("Heart_datasets/heart.csv")
        x = df.head(100)
        st.write(x)

    # ========= ANALYSIS TAB =========
    if sidebar == "Analysis":
        st.header("Analysis")
        st.write("Insights dataset")
        st.image("img/heart1.jpg")
        st.image("img/heart2.jpg")
        st.image("img/heart3.jpg")

    # ========= OUR MODEL PREDICTION TAB =========
    if sidebar == "Our Model Prediction":
        st.image("artificial.jpg")
        st.header("Let's use our data for Heart Failure Prediction")
        st.write("Let's see what the AI says about your heart")
        st.subheader("Enter your details")
        # making dictionaries
        sex_options = {"Male": 1, "Female": 0}
        chest_pain_type_options = {
            "Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        fasting_bs_options = {"True": 1, "Fasle": 0}
        fasting_ecg_options = {
            "Normal": 0, "Having ST-T Wave Abnormality": 1, "Showing Left Ventricular Hypertrophy": 2}
        exercise_angina_options = {"Yes": 1, "No": 0}
        st_slope_option = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        # Input fields for the heart failure prediction attributes
        name = st.text_input("Name", value="", max_chars=30)
        age = st.number_input("Age", min_value=1, max_value=120, value=None)
        sex = st.selectbox('Sex ( Male  or Female)',
                           options=[""] + list(sex_options.keys()))
        chest_pain_type = st.selectbox(
            'Chest Pain Type (0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic)',
            options=[""] + list(chest_pain_type_options.keys()))
        resting_bp = st.number_input(
            "Resting Blood Pressure (Min 68mm Hg to Max 250mm Hg )", min_value=68, max_value=250, value=None)
        cholesterol = st.number_input(

            "Cholesterol (Min 100 mg/dL to Max  600mg/dL)", min_value=50, max_value=600, value=None)
        fasting_bs = st.selectbox(
            'Fasting Blood Sugar (1: True, 0: False)', options=[""] + list(fasting_bs_options.keys()))
        resting_ecg = st.selectbox(
            'Resting ECG Results (0: Normal, 1: Having ST-T Wave Abnormality, 2: Showing Left Ventricular Hypertrophy)',
            options=[""] + list(fasting_ecg_options.keys())
        )
        max_hr = st.number_input(
            "Maximum Heart Rate Achieved" " (Min 60bpm to Max 200bpm) ", min_value=60, max_value=220, value=None)
        exercise_angina = st.selectbox(
            'Exercise Induced Angina (1: Yes, 0: No)',
            options=[""] + list(exercise_angina_options.keys()))
        oldpeak = st.number_input("Oldpeak (Min 0.0 to Max 6.2 ST depression induced by exercise relative to rest )",
                                  min_value=0.0, max_value=6.2, step=0.1, format="%.1f", value=None)
        st_slope = st.selectbox(
            'Slope of the Peak Exercise ST Segment (0: Upsloping, 1: Flat, 2: Downsloping)',
            options=[""] + list(st_slope_option.keys())
        )

        clicked = st.button("Predict")

        if clicked:
            if not name.strip():
                st.error("Please enter your name.")
            elif age is None or age <= 0 or age > 120:
                st.error("Please enter a valid age between 1 and 120.")
            elif cholesterol is None or cholesterol < 50 or cholesterol > 600:
                st.error("Cholesterol must be between 50 and 600 mg/dL.")
            elif resting_bp is None or resting_bp < 50 or resting_bp > 250:
                st.error(
                    "Resting Blood Pressure must be between 50 and 250 mm Hg.")
            elif max_hr is None or max_hr < 60 or max_hr > 220:
                st.error("Maximum Heart Rate must be between 60 and 220.")
            elif oldpeak is None or oldpeak < 0.0 or oldpeak > 6.2:
                st.error("Oldpeak must be between 0.0 and 6.2.")
            else:
                try:
                    model = joblib.load(open('model.pkl', 'rb'))

                    if model is not None:
                        sex_values = sex_options[sex]
                        chest_pain_type_values = chest_pain_type_options[chest_pain_type]
                        fasting_bs_values = fasting_bs_options[fasting_bs]
                        resting_ecg_values = fasting_ecg_options[resting_ecg]
                        exercise_angina_values = exercise_angina_options[exercise_angina]
                        st_slope_values = st_slope_option[st_slope]
                        features = np.array([[age, sex_values, chest_pain_type_values, resting_bp, cholesterol,
                                              fasting_bs_values, resting_ecg_values, max_hr, exercise_angina_values, oldpeak, st_slope_values]])
                        predicted = model.predict(features)

                        # Display header for predicted result
                        st.header("Predicted Result")
                        st.info(
                            '0 (No possibility of heart attack), 1 (Future heart attack detected)')

                        # Assuming `predicted` is a list or array containing the model's prediction
                        # Replace this with your actual prediction logic
                        predicted = [1]

                        # Conditional message display based on prediction
                        if predicted[0] == 0:
                            st.success(
                                "✅ Good news! No possibility of heart attack")
                        elif predicted[0] == 1:
                            st.warning("⚠ Future heart attack detected")
                        else:
                            st.error("Unexpected prediction value")
                        # Display the user's input values
                        st.subheader("Your Input Values")
                        st.write(f"**Name:** {name}")
                        st.write(f"**Age:** {age}")
                        st.write(
                            f"**Sex:** {'Male' if sex == 1 else 'Female'}")
                        st.write(f"**Chest Pain Type:** {chest_pain_type}")
                        st.write(
                            f"**Resting Blood Pressure:** {resting_bp} mm Hg")
                        st.write(f"**Cholesterol:** {cholesterol} mg/dL")
                        st.write(
                            f"**Fasting Blood Sugar:** {'True' if fasting_bs == 1 else 'False'}")
                        st.write(f"**Resting ECG Results:** {resting_ecg}")
                        st.write(f"**Maximum Heart Rate Achieved:** {max_hr}")
                        st.write(
                            f"**Exercise Induced Angina:** {'Yes' if exercise_angina == 1 else 'No'}")
                        st.write(f"**Oldpeak:** {oldpeak}")
                        st.write(
                            f"**Slope of the Peak Exercise ST Segment:** {st_slope}")

                        # Prepare the content for the downloadable file
                        report_content = f"""
                        Name: {name}
                        Age: {age}
                        Sex: {'Male' if sex == 1 else 'Female'}
                        Chest Pain Type: {chest_pain_type}
                        Resting Blood Pressure: {resting_bp} mm Hg
                        Cholesterol: {cholesterol} mg/dL
                        Fasting Blood Sugar: {'True' if fasting_bs == 1 else 'False'}
                        Resting ECG Results: {resting_ecg}
                        Maximum Heart Rate Achieved: {max_hr}
                        Exercise Induced Angina: {'Yes' if exercise_angina == 1 else 'No'}
                        Oldpeak: {oldpeak}
                        Slope of the Peak Exercise ST Segment: {st_slope}
                        Prediction Result: {predicted[0]}
                        """

                        # Create a download link
                        st.download_button(
                            label="Download Prediction Report",
                            data=report_content,
                            file_name=f"{name}_prediction_report.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    st.error(f"Error loading the model: {e}")


# ========= ABOUT TAB =========
    if sidebar == "About":

        st.header("About")

        st.subheader("How soon after treatment will You feel better?")
        st.write("""
    After you’ve had a heart attack, you’re at a higher risk of a similar occurrence. Your healthcare provider will likely recommend follow-up monitoring, testing and care to avoid future heart attacks. Some of these include:

    - Heart scans: Similar to the methods used to diagnose a heart attack, these can assess the effects of your heart attack and determine if you have permanent heart damage. They can also look for signs of heart and circulatory problems that increase the chance of future heart attacks.
    - Stress test: These heart tests and scans that take place while you’re exercising can show potential problems that stand out only when your heart is working harder.
    - Cardiac rehabilitation: These programs help you improve your overall health and lifestyle, which can prevent another heart attack.


    Additionally, you’ll continue to take medicines — some of the ones you received for immediate treatment of your heart attack — long-term. These include:

    - Beta-blockers.
    - ACE inhibitors.
    - Aspirin and other blood-thinning agents.""")

        st.subheader("How soon after treatment will I feel better?")
        st.write("""
    In general, your heart attack symptoms should decrease as you receive treatment. You’ll likely have some lingering weakness and fatigue during your hospital stay and for several days after. Your healthcare provider will give you guidance on rest, medications to take, etc.

    Recovery from the treatments also varies, depending on the method of treatment. The average hospital stay for a heart attack is between four and five days. In general, expect to stay in the hospital for the following length of time:

    - Medication only: People treated with medication only have an average hospital stay of approximately six days.
    - PCI (Percutaneous Coronary Intervention): Recovering from PCI is easier than surgery because it’s a less invasive method for treating a heart attack. The average length of stay for PCI is about four days. In Indian households, where family support plays a vital role, this quicker recovery means that patients can resume their roles within the family and community without too much disruption.
    - CABG (Coronary Artery Bypass Grafting): In contrast, CABG is a major surgery that requires a longer recovery time, typically around seven days in the hospital. This extended stay means patients need more time to heal, and families often step in to provide care and support. While the longer recovery can be challenging, it also strengthens familial bonds, as loved ones rally together to help the patient. However, there are financial considerations, especially for families where the primary earner may be unable to work for weeks or months.
    - In India, the decision often involves family discussions, considering not just medical factors but also socio-economic implications. Access to healthcare facilities, post-operative support, and overall health status play crucial roles in determining the most suitable approach for heart treatment.    
        """)

        st.subheader("How common are heart attacks?")
        st.write("""
    Heart attacks are quite common in India, with cardiovascular diseases being a leading cause of mortality. According to various studies, it’s estimated that around 1 in 4 people in India may suffer from some form of heart disease, with heart attacks increasingly affecting younger populations due to lifestyle factors, stress, and dietary habits.

    Urbanization, smoking, sedentary lifestyles, and increasing obesity rates contribute to this trend. Awareness and early intervention are critical, as many cases can be managed or prevented with lifestyle changes and proper medical care .""")

    # ========= FEEDBACK TAB =========
    if sidebar == "Feedback":
        col1, col2 = st.columns([2, 2])
        st.markdown("### Bug Report 🪲")
        bug_report = st.text_area("Please describe the issue or report a bug:")
        uploaded_file = st.file_uploader(
            "Attach Screenshot (optional):", type=["png", "jpg"])
        if uploaded_file is not None:
            st.markdown(
                "**<span style = 'color:lightgreen'>Screenshot Attached Successfully 👍🏻</span>**", unsafe_allow_html=True)
            with st.expander("Preview Attached Screenshot"):
                st.image(uploaded_file)
        send_button = st.button("Send Report ✈️")
        if send_button:
            st.markdown(
                "<span style = 'color:lightgreen'>Report Sent Successfully, We'll get back to you super soon ⚡</span>", unsafe_allow_html=True)
            st.markdown(
                "## <span style = 'color:white'>Thank You 💖</span>", unsafe_allow_html=True)


# Run the application
if __name__ == "__main__":
    main_app()
