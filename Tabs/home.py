import streamlit as st
import PIL

def app():
    st.title('Integrated Hepatic Health Care Program')
    st.image('./images/hepatic.png')

    
    st.markdown(
    """<p style="font-size:20px;">
            
**Hepatic Disease (Liver Disease)** is a broad term that encompasses various conditions affecting liver function, including cirrhosis, fatty liver disease, hepatitis, and liver cancer.  
The liver plays a crucial role in detoxification, metabolism, digestion, and nutrient storage. Liver diseases can result from viral infections, excessive alcohol consumption, obesity, autoimmune disorders, or genetic factors.  

There is no absolute cure for advanced liver diseases, but early detection, lifestyle modifications, proper medication, and managing underlying conditions such as diabetes and high cholesterol can slow disease progression and improve liver health.  

This **Web App** utilizes the **Random Forest Classifier** to analyze key health parameters and predict whether a person has liver disease or is at risk of developing it in the future.  
        </p>
    """, unsafe_allow_html=True)

    