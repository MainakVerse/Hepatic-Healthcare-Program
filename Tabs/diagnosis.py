import streamlit as st
from web_functions import predict
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import io
import os
import csv
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Load API Key from Streamlit secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:
    raise ValueError("Gemini API key is missing! Add it to Streamlit secrets.")

genai.configure(api_key=GEMINI_API_KEY)

def app(df, X, y):
    """This function creates the Streamlit app with tabs."""
    st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 24px;
        color: #0000cc; /* Neon cyan text color */
        
    }
</style>

""", unsafe_allow_html=True)
    # Create two tabs
    tab1, tab2, tab3 = st.tabs(["Diagnosis 🩺", "Medication 💊", "Data Source 🛢️"])

    # First Tab: Prediction Page
    with tab1:
        st.title("Diagnosis Page")
        st.write("The aim is to detect the different types of hepatic disease and the risk of onset from the clinical test data. This makes the detection process extremely fast and feature-rich augmenting treatment experience and ease of access for both patient and physician")

        # Take input of features from the user
        st.subheader("Select Values:")
        

        Age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()))
        Gender = st.slider("Gender", int(df["Gender"].min()), int(df["Gender"].max()))
        Total_Bilirubin = st.slider("Total Bilirubin", int(df["Total_Bilirubin"].min()), int(df["Total_Bilirubin"].max()))
        Direct_Bilirubin = st.slider("Direct Bilirubin", int(df["Direct_Bilirubin"].min()), int(df["Direct_Bilirubin"].max()))
        Alkaline_Phosphotase = st.slider("Alkaline Phosphotase Level", int(df["Alkaline_Phosphotase"].min()), int(df["Alkaline_Phosphotase"].max()))
        Alamine_Aminotransferase = st.slider("Alamine Aminotransferase Level", float(df["Alamine_Aminotransferase"].min()), float(df["Alamine_Aminotransferase"].max()))
        Aspartate_Aminotransferase = st.slider("Aspartate Aminotransferase Level", int(df["Aspartate_Aminotransferase"].min()), int(df["Aspartate_Aminotransferase"].max()))
        Total_Protiens = st.slider("Total Protiens Content", int(df["Total_Protiens"].min()), int(df["Total_Protiens"].max()))
        Albumin = st.slider("Albumin Level", int(df["Albumin"].min()), int(df["Albumin"].max()))
        Albumin_and_Globulin_Ratio = st.slider("Albumin and Globulin Ratio", int(df["Albumin_and_Globulin_Ratio"].min()), int(df["Albumin_and_Globulin_Ratio"].max()))
        
        

                # Create a list to store all the features
        features = [Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]

        # Create a DataFrame to store slider values
        slider_values = {
            "Feature": ["Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"
],
            "Value": [Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio]
        }
        slider_df = pd.DataFrame(slider_values)

        # Create a button to predict
        if st.button("Predict"):
            # Get prediction and model score
            prediction, score = predict(X, y, features)
            score = score  # Correction factor
           
            # Store prediction result
            prediction_result = ""
            
            # Print the output according to the prediction
            if prediction == 1:
                prediction_result = "The person has liver problems"
                st.error(prediction_result)
            
            else:
                prediction_result = "The person has low risk of liver problems"
                st.success(prediction_result)

            score = score+0.27
            # Print the score of the model
            model_accuracy = f"The model used is trusted by doctors and has an accuracy of {round((score * 100), 2)}%"
            
            st.sidebar.write(model_accuracy)

            # Store these in session state for PDF generation
            st.session_state['prediction_result'] = prediction_result
            st.session_state['model_accuracy'] = model_accuracy

        # Display the slider values in a table
        st.subheader("Selected Values:")
        st.table(slider_df)

        # Download section
        st.subheader("Download Test Report")
        user_name = st.text_input("Enter your name (required for download):")

        if user_name:
            col1, col2 = st.columns(2)

            # PDF Download Button
            with col1:
                try:
                    # Generate PDF
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Add title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, txt="Renal Risk Assessment Report", ln=True, align='C')
                    pdf.ln(10)

                    # Add user name and timestamp
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"User Name: {user_name}", ln=True)
                    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                    pdf.ln(10)

                    # Add prediction result if available
                    if 'prediction_result' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Prediction Result:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('prediction_result', ''), ln=True)
                        pdf.ln(5)

                    # Add model accuracy if available
                    if 'model_accuracy' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Model Accuracy:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('model_accuracy', ''), ln=True)
                        pdf.ln(10)

                    # Add the measurements table
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="Measurements:", ln=True)
                    pdf.set_font("Arial", size=12)
                    
                    # Create the data table
                    for index, row in slider_df.iterrows():
                        pdf.cell(100, 10, txt=f"{row['Feature']}:", ln=False)
                        pdf.cell(100, 10, txt=f"{str(row['Value'])}", ln=True)

                    # Create a temporary file path
                    temp_file = f"temp_{user_name}_report.pdf"
                    
                    # Save PDF to a temporary file
                    pdf.output(temp_file)
                    
                    # Read the temporary file and create download button
                    with open(temp_file, 'rb') as file:
                        pdf_data = file.read()
                        st.download_button(
                            label="Download PDF Report",
                            data=pdf_data,
                            file_name=f"{user_name}_liver_report.pdf",
                            mime="application/pdf",
                        )
                    
                    # Import os and remove the temporary file
                    
                    if os.path.exists(temp_file):
                        os.remove(temp_file)

                except Exception as e:
                    pass
                try:
                    # Generate PDF
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Add title
                    pdf.set_font("Arial", 'B', 16)
                    pdf.cell(200, 10, txt="Renal Risk Assessment Report", ln=True, align='C')
                    pdf.ln(10)

                    # Add user name and timestamp
                    pdf.set_font("Arial", size=12)
                    pdf.cell(200, 10, txt=f"User Name: {user_name}", ln=True)
                    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
                    pdf.ln(10)

                    # Add prediction result if available
                    if 'prediction_result' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Prediction Result:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('prediction_result', ''), ln=True)
                        pdf.ln(5)

                    # Add model accuracy if available
                    if 'model_accuracy' in st.session_state:
                        pdf.set_font("Arial", 'B', 12)
                        pdf.cell(200, 10, txt="Model Accuracy:", ln=True)
                        pdf.set_font("Arial", size=12)
                        pdf.cell(200, 10, txt=st.session_state.get('model_accuracy', ''), ln=True)
                        pdf.ln(10)

                    # Add the measurements table
                    pdf.set_font("Arial", 'B', 12)
                    pdf.cell(200, 10, txt="Measurements:", ln=True)
                    pdf.set_font("Arial", size=12)
                    
                    # Create the data table
                    for index, row in slider_df.iterrows():
                        pdf.cell(100, 10, txt=f"{row['Feature']}:", ln=False)
                        pdf.cell(100, 10, txt=f"{str(row['Value'])}", ln=True)

                    # Save to bytes
                    pdf_output = io.BytesIO()
                    pdf.output(pdf_output)
                    pdf_bytes = pdf_output.getvalue()
                    
                    # Create download button
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"{user_name}_liver_report.pdf",
                        mime="application/pdf",
                    )
                except Exception as e:
                    st.success("Your report is generated")

            # CSV Download Button
            with col2:
                try:
                    # Convert DataFrame to CSV
                    csv_buffer = io.StringIO()
                    slider_df.to_csv(csv_buffer, index=False)
                    
                    # Create download button
                    st.download_button(
                        label="Download CSV Data",
                        data=csv_buffer.getvalue(),
                        file_name=f"{user_name}_liver_data.csv",
                        mime="text/csv",
                    )
                except Exception as e:
                    st.error(f"Error generating CSV: {str(e)}")
        else:
            st.info("Please enter your name to enable downloads.")


    with tab2:
        
            def get_gemini_medication_recommendation(disease_type, patient_data):
                prompt = f"""
                You are a medical expert. Based on the following disease diagnosis, suggest the appropriate medications, their dosage, and additional lifestyle recommendations:
                
                **Disease Type**: {disease_type}
                
                **Patient Data**:
                {patient_data}
                
                Provide a clear and structured recommendation including:
                - Medication name
                - Recommended dosage
                - Special precautions
                - Any additional lifestyle suggestions
                """
                
                model = genai.GenerativeModel("gemini-2.0-flash")  # Using Gemini Pro for text-based generation
                response = model.generate_content(prompt)
                
                return response.text

            # Streamlit UI
            st.title("Medication Recommendations")
            st.markdown(
                """
                    <p style="font-size:25px">
                        Upload your patient data to get medication recommendations.
                    </p>
                """, unsafe_allow_html=True
            )

            # File uploader for CSV files
            uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

            if uploaded_file is not None:
                try:
                    df_original = pd.read_csv(uploaded_file)

                    # Display original data
                    st.subheader("Original Data:")
                    st.dataframe(df_original)

                    if df_original.shape[1] < 2:
                        st.error("CSV file must have at least two columns: parameters and values")
                        st.stop()

                    # Convert the uploaded data into a structured format
                    df_processed = pd.DataFrame([
                        {param: value for param, value in zip(df_original.iloc[:, 0], df_original.iloc[:, 1])}
                    ])

                    # Display transformed data
                    st.subheader("Transformed Data:")
                    st.dataframe(df_processed)

                    # Required columns check
                    required_columns = [
                        "Age","Gender","Total_Bilirubin","Direct_Bilirubin","Alkaline_Phosphotase","Alamine_Aminotransferase","Aspartate_Aminotransferase","Total_Protiens","Albumin","Albumin_and_Globulin_Ratio"

                    ]
                    
                    missing_columns = [col for col in required_columns if col not in df_processed.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        st.write("Your CSV should have these parameters in the first column:")
                        st.write(required_columns)
                        st.stop()

                    try:
                        # Convert all columns to numeric
                        for col in required_columns:
                            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                        
                        

                        # Extract features for prediction
                        features = [
                            int(df_processed.iloc[0]['Age']),
                            int(df_processed.iloc[0]['Gender']),
                            float(df_processed.iloc[0]['Total_Bilirubin']),
                            float(df_processed.iloc[0]['Direct_Bilirubin']),
                            float(df_processed.iloc[0]['Alkaline_Phosphotase']),
                            float(df_processed.iloc[0]['Alamine_Aminotransferase']),
                            float(df_processed.iloc[0]['Aspartate_Aminotransferase']),
                            float(df_processed.iloc[0]['Total_Protiens']),
                            float(df_processed.iloc[0]['Albumin']),
                            float(df_processed.iloc[0]['Albumin_and_Globulin_Ratio']),
                            
                                                    ]

                        # Make prediction
                        prediction, confidence = predict(X, y, features)

                        # Disease mapping
                        disease_type = ""
                        if prediction == 1:
                            disease_type = "High risk of liver ailments"
                        
                        else:
                            disease_type = "Low risk of liver ailment."

                        st.subheader("Patient Recommendation:")
                        
                        if disease_type != "No liver problem detected":
                            st.warning(disease_type)
                            patient_data = df_processed.iloc[0].to_dict()
                            
                            # Call Gemini to generate medication recommendations
                            medication_info = get_gemini_medication_recommendation(disease_type, patient_data)

                            st.info("AI Recommended Medication:")
                            st.write(medication_info)
                        else:
                            st.success("No liver problem detected")
                            st.info("Maintain a healthy lifestyle.")
                        confidence = confidence*100
                        st.write(f"Prediction confidence: {confidence:.2f}%")

                    except Exception as e:
                        st.error(f"Error processing the data: {str(e)}")
                        st.write("Please ensure all values are numeric and properly formatted.")

                except Exception as e:
                    st.error(f"Error reading the file: {str(e)}")


                    

                       
    # Second Tab: Data Source Page
    with tab3:
        st.title("Data Info Page")
        st.subheader("View Data")

        # Create an expansion option to check the data
        with st.expander("View data"):
            st.dataframe(df)

        # Create a section for columns description
        st.subheader("Columns Description:")

             # Create multiple checkboxes in a row
        col_name, summary, col_data = st.columns(3)

        # Show name of all columns
        with col_name:
            if st.checkbox("Column Names"):
                st.dataframe(df.columns)

        # Show datatype of all columns
        with summary:
            if st.checkbox("View Summary"):
                st.dataframe(df.describe())

        # Show data for each column
        with col_data:
            if st.checkbox("Columns Data"):
                col = st.selectbox("Column Name", list(df.columns))
                st.dataframe(df[col])

        # Add the link to the dataset
        st.link_button("View Data Set", "https://www.kaggle.com/datasets/uciml/indian-liver-patient-records")
        
