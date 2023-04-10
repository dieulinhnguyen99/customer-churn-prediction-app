import streamlit as st
import pandas as pd
import argparse
import os
import sys
from PIL import Image
from inference import DowngradeInferencePipeline
#load the model from disk

def main(arguments):
    """
    Main function
    """
    #Create parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #Add arguments  
    parser.add_argument(
        "--input-dir", type=str,
        help='location of the input'
    )

    #Parse the arguments
    args = parser.parse_args(arguments)

#Setting Application title
    st.title('Telco Customer Churn Prediction App')

        #Setting Application description
    st.markdown("""
        :dart:  This Streamlit app is made to predict customer churn in a ficitional telecommunication use case.
    The application is functional for both online prediction and batch data prediction. \n
    """)
    st.markdown("<h3></h3>", unsafe_allow_html=True)

    #Setting Application sidebar default
    image = Image.open('Churn.png')
    add_selectbox = st.sidebar.selectbox(
    "How would you like to predict?", ("Online", "Batch"))
    st.sidebar.info('This app is created to predict Customer Churn')
    st.sidebar.image(image)

    log_dir = args.input_dir
    model_path = os.path.join(log_dir, "model.json")
    scaler_path = os.path.join(log_dir, "scaler.joblib")
    pipeline = DowngradeInferencePipeline(model_path, scaler_path)

    #Setting selectbox
    if add_selectbox == "Online":
        st.info("Input data below")

        #Setting selectbox Identification data
        st.subheader("Identification data")
        customerID = st.text_input("Customer ID")

        #Setting selectbox Demographic data
        #Based on our optimal features selection
        st.subheader("Demographic data")
        gender = st.selectbox('Gender:', ('Male', 'Female'))
        seniorcitizen = st.selectbox('Senior Citizen:', ('Yes', 'No'))
        dependents = st.selectbox('Dependent:', ('Yes', 'No'))
        partner = st.selectbox('Partner:', ('Yes', 'No'))

        #Setting selectbox Payment data
        st.subheader("Payment data")
        tenure = st.slider('Number of months the customer has stayed with the company', min_value=0, max_value=72, value=0)
        contract = st.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
        paperlessbilling = st.selectbox('Paperless Billing', ('Yes', 'No'))
        PaymentMethod = st.selectbox('PaymentMethod',('Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'))
        monthlycharges = st.number_input('The amount charged to the customer monthly', min_value=0, max_value=150, value=0)
        totalcharges = st.number_input('The total amount charged to the customer',min_value=0, max_value=10000, value=0)

        #Setting selectbox Services signed up for
        st.subheader("Services signed up for")
        mutliplelines = st.selectbox("Does the customer have multiple lines",('Yes','No','No phone service'))
        phoneservice = st.selectbox('Phone Service:', ('Yes', 'No'))
        internetservice = st.selectbox("Does the customer have internet service", ('DSL', 'Fiber optic', 'No'))
        onlinesecurity = st.selectbox("Does the customer have online security",('Yes','No','No internet service'))
        onlinebackup = st.selectbox("Does the customer have online backup",('Yes','No','No internet service'))
        deviceprotection = st.selectbox("Does the customer have device protection",('Yes','No','No internet service'))
        techsupport = st.selectbox("Does the customer have technology support", ('Yes','No','No internet service'))
        streamingtv = st.selectbox("Does the customer stream TV", ('Yes','No','No internet service'))
        streamingmovies = st.selectbox("Does the customer stream movies", ('Yes','No','No internet service'))

        data = {
                'customerID': customerID,
                'gender': gender,
                'SeniorCitizen': seniorcitizen,
                'Partner': partner,
                'Dependents': dependents,
                'tenure':tenure,
                'PhoneService': phoneservice,
                'MultipleLines': mutliplelines,
                'InternetService': internetservice,
                'OnlineSecurity': onlinesecurity,
                'OnlineBackup': onlinebackup,
                'DeviceProtection': deviceprotection,
                'TechSupport': techsupport,
                'StreamingTV': streamingtv,
                'StreamingMovies': streamingmovies,
                'Contract': contract,
                'PaperlessBilling': paperlessbilling,
                'PaymentMethod':PaymentMethod, 
                'MonthlyCharges': monthlycharges, 
                'TotalCharges': totalcharges
                }

        features_df = pd.DataFrame.from_dict([data])
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.write('Overview of input is shown below')
        st.markdown("<h3></h3>", unsafe_allow_html=True)
        st.dataframe(features_df)

        output, explaination = pipeline.run(features_df)
        prediction = output.loc[0, "Churn"]
        probability = output.loc[0, "Probability"]

        #Setting Predict sidebar 
        if st.button('Predict'):
            if prediction == 1:
                st.warning(f'Yes, the customer terminate the service with the probability of {probability*100: .2f}%.')
            else:
                st.success(f'No, the customer is happy with Telco Services and does not terminate the service with the probability of {100-probability*100: .2f}%.')
            
            st.write(explaination[['Category', 'Value','Contribution']])

    else:
        st.subheader("Dataset upload")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            #Get overview of data
            st.write(data.head())
            st.markdown("<h3></h3>", unsafe_allow_html=True)
            
            if st.button('Predict'):
                #Get batch prediction
                output, explaination = pipeline.run(data)
                
                st.markdown("<h3></h3>", unsafe_allow_html=True)
                st.subheader('Prediction')
                st.write(output)
            
if __name__ == '__main__':
    main(sys.argv[1:])
