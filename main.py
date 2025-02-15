import streamlit as st
import joblib
import pandas as pd

# Load model yang telah disimpan
try:
    model = joblib.load('model_regression1.pkl')
    st.success("Model loaded successfully!")  # Model loading success message
except FileNotFoundError:
    st.error("Model file not found. Please check the file path.")
except Exception as e:
    st.error(f"Error loading model: {e}")  # Error message if loading fails

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main Page", "Visualization"])
# Kolom data training (X.columns) - pastikan ini sesuai dengan model yang dilatih
X_columns = [
    'Hours_Studied', 'Attendance', 'Tutoring_Sessions', 'Physical_Activity',
    'Motivation_Level_Medium',  # Jaga hanya satu dari Motivation_Level
    'Family_Income_Medium', 'Family_Income_High',
    'Teacher_Quality_Medium', 'Teacher_Quality_High',
    'Peer_Influence_Neutral', 'Peer_Influence_Positive',
    'Internet_Access_Yes', 'School_Type_Private',
    'Learning_Disabilities_Yes', 'Parental_Education_Level_College',
    'Parental_Education_Level_Postgraduate',
    'Distance_from_Home_Moderate', 'Distance_from_Home_Far',
    'Gender_Male'
]

# Kolom kategoris sesuai dengan training
categorical_cols = ['Motivation_Level', 'Family_Income', 'Teacher_Quality', 'Peer_Influence',
                    'Internet_Access', 'School_Type', 'Learning_Disabilities',
                    'Parental_Education_Level', 'Distance_from_Home', 'Gender']

# Fungsi untuk melakukan prediksi dengan data baru yang diinput pengguna
def predict_new_data(new_data):
    new_df = pd.DataFrame([new_data])

    # Lakukan encoding terhadap data kategoris
    new_df_encoded = pd.get_dummies(new_df, columns=categorical_cols, drop_first=True)

    # Debugging: Tampilkan kolom dari new_df_encoded untuk memeriksa masalah


    # Pastikan fitur sesuai dengan data training (menambahkan kolom yang hilang jika perlu)
    new_df_encoded = new_df_encoded.reindex(columns=X_columns, fill_value=0)

    # Debugging: Tampilkan kolom setelah reindexing
    

    if new_df_encoded.shape[1] == len(X_columns):
        prediction = model.predict(new_df_encoded)
        return prediction[0]
    else:
        st.write("Jumlah kolom tidak sesuai dengan yang diharapkan model. Tidak bisa melakukan prediksi.")
        return None



def main_page():
    st.title("Prediksi Nilai dengan Linear Regression")

    # Input dari pengguna untuk masing-masing fitur
    hours_studied = st.number_input("Hours Studied:", min_value=0)
    attendance = st.number_input("Attendance (%):", min_value=0, max_value=100)
    motivation_level = st.selectbox("Motivation Level:", ['Low', 'Medium', 'High'])
    internet_access = st.selectbox("Internet Access:", ['No', 'Yes'])
    tutoring_sessions = st.number_input("Tutoring Sessions:", min_value=0)
    family_income = st.selectbox("Family Income:", ['Low', 'Medium', 'High'])
    teacher_quality = st.selectbox("Teacher Quality:", ['Low', 'Medium', 'High'])
    school_type = st.selectbox("School Type:", ['Public', 'Private'])
    peer_influence = st.selectbox("Peer Influence:", ['Negative', 'Neutral', 'Positive'])
    physical_activity = st.number_input("Physical Activity (hours/week):", min_value=0)
    learning_disabilities = st.selectbox("Learning Disabilities:", ['No', 'Yes'])
    parental_education_level = st.selectbox("Parental Education Level:", ['High School', 'College', 'Postgraduate'])
    distance_from_home = st.selectbox("Distance from Home:", ['Near', 'Moderate', 'Far'])
    gender = st.selectbox("Gender:", ['Male', 'Female'])

    # Jika tombol prediksi ditekan
    if st.button("Prediksi"):
        new_input = {
            'Hours_Studied': hours_studied,
            'Attendance': attendance,
            'Motivation_Level': motivation_level,
            'Internet_Access': internet_access,
            'Tutoring_Sessions': tutoring_sessions,
            'Family_Income': family_income,
            'Teacher_Quality': teacher_quality,
            'School_Type': school_type,
            'Peer_Influence': peer_influence,
            'Physical_Activity': physical_activity,
            'Learning_Disabilities': learning_disabilities,
            'Parental_Education_Level': parental_education_level,
            'Distance_from_Home': distance_from_home,
            'Gender': gender
        }

        prediction = predict_new_data(new_input)

        if prediction is not None:
            st.write(f"Hasil Prediksi: {prediction}")
        else:
            st.write("Prediksi tidak dapat dilakukan. Periksa kesesuaian jumlah kolom.")

       
def visualization_page():
    import visualization  # Ensure this file is in the same directory
    visualization.show_visualizations()   # Call the function to display visualizations

# Render the selected page
if page == "Main Page":
    main_page()
elif page == "Visualization":
    visualization_page()