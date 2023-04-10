import pandas as pd
import xgboost as xgb
import joblib
from eli5.formatters.as_dataframe import explain_prediction_df
import warnings
warnings.filterwarnings('ignore')

class DowngradeInferencePipeline():
    """
    Inference pipeline
    """
    def __init__(self, model_path, scaler_path):
        """
        Contructor

        Params:
            model_path: file path of the saved model
            scaler_path: file path of the saved scaler
        """
        super().__init__

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = xgb.XGBClassifier()
        self.model.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)


    def run(self, df, threshold=0.5):
        """
        Run the inference pipeline

        Params:
            df: dataframe

        Return:
            output_df: the prediction
            explanation_table: dataframe for explaination the output

        """
        processed_data = self.__clean_df(df.copy())
        output_df = processed_data[['customerID']]
        processed_data = self.__process_features(processed_data)

        # copy processed_data to df
        df = processed_data.copy(deep=True)
        feature_names = processed_data.columns.values.tolist()
        processed_data = self.scaler.transform(processed_data)
        print(processed_data.shape)
        print('[INFO] Predicting...')
        probabilities = self.model.predict_proba(processed_data)[:,1]
        prediction = probabilities > threshold
        output_df.loc[:,'Churn'] = prediction.astype(int)
        output_df.loc[:,'Probability'] = probabilities
        explaination_df = explain_prediction_df(self.model, processed_data[0], feature_names=feature_names)
        explaination_df = explaination_df.rename(columns={'feature': 'Category', 'weight': 'Contribution'})
        
       
        df['<BIAS>'] = '<BIAS>'
        df = df.transpose().reset_index()
        
        df.rename(columns={'index': 'Category', 0:'Value'}, inplace=True)
        df['Value'] = df['Value'].astype(str)

        explanation_table = pd.merge(df,explaination_df, on = 'Category')
        
        # find rows of explanation_table which have 0 or 1 in their values -> convert into 'Yes' or 'No'
        values = ['SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
        'gender_Female', 'gender_Male', 'MultipleLines_No', 'MultipleLines_No phone service', 
        'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',
        'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
        'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service', 
        'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month', 
        'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', 
        'PaymentMethod_Electronic check','PaymentMethod_Mailed check']

        for c in values:
            explanation_table['Value'][explanation_table['Category']==c] = explanation_table['Value'][explanation_table['Category']==c].apply(lambda x: 'Yes' if x == 1 else 'No')

        return output_df, explanation_table

    def __clean_df(self, df):
        """
        Clean dataset
        """
        def convert_to_numeric(df, columns=[]):
            """
            Converting variable to a numerical data type
            """
            if len(columns) > 0:
                for column in columns:
                    df.loc[:, column] = pd.to_numeric(df[column], errors = 'coerce')
            
            return df
              
        def fillna(df, fill_values={}):
            """
            Fill the missing values with the other value
            """
            if len(fill_values) > 0:
                for k,v in fill_values.items():
                    df.fillna({k:v}, inplace=True)
            return df

        #Test converting Total Charges to a numerical data type and replacing missing values with 0
        cleaning_functions = [(lambda x: convert_to_numeric(x, ["TotalCharges"]), 'Converting object column to type float64'),
                   (lambda x: fillna(x,{'TotalCharges': 0}), 'Replacing nan values with 0 from "TotalCharges" column')]

        for cleaning_function, description in cleaning_functions:
            df = cleaning_function(df)
            print(f'{description}: Done!')

        return df

    
    def __process_features(self,df):
        """
        Process features from dataset 
        """
        print('[INFO] Processing features...')
        dropped_features = ["customerID"]
        df = df.drop(dropped_features, axis=1)
        
        #List of features that contain the yes/no variables
        EN_FEATURES = ['Partner', 'Dependents','PhoneService', 'PaperlessBilling']
       
        def process_encoding(df, encoding_columns = EN_FEATURES):
            """
            Convert the yes/no variable to a binary variable, 1 for Yes and 0 for No
            """
            if len(encoding_columns)>0:
                for column in encoding_columns:
                    df.loc[:, [column]] = df[column].map({'No':0, 'Yes':1})
                    
            return df

        #List of features containing the categorical variables
        CAT_FEATURES = ['gender', 'MultipleLines','InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport','StreamingTV','StreamingMovies', 'Contract','PaymentMethod']

        def process_categorical_features(df, cat_features=CAT_FEATURES):
            """
            Convert the categorical variables into dummy variables 
            """
            if len(cat_features) > 0:
                for cat_feature in cat_features:
                    df.loc[:, [cat_feature]] = df[cat_feature].astype(str)

                df = pd.get_dummies(df)
                    
            return df

        #Test encoding the yes/no features and coverting categorical features
        processing_functions = [(process_encoding, "Encoding"),(process_categorical_features, 'Categorical features')]
        for processing_function, description in processing_functions:
            df = processing_function(df)
            print(f'{description}: Done!')
        
        #List of all features
        features = ['SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling', 'MonthlyCharges', \
            'TotalCharges', 'gender_Female', 'gender_Male', 'MultipleLines_No', 'MultipleLines_No phone service', \
                'MultipleLines_Yes', 'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', \
                    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 'OnlineBackup_No internet service',\
                         'OnlineBackup_Yes', 'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes', \
                            'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 'StreamingTV_No internet service', \
                                'StreamingTV_Yes', 'StreamingMovies_No', 'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'Contract_Month-to-month', \
                                    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)', \
                                        'PaymentMethod_Electronic check','PaymentMethod_Mailed check']

        #Replace variables in the feature, which are not in the list, with 0
        for feature in features:
            if feature not in df.columns.values.tolist():
                df.loc[:, feature] = 0
        
        df = df[features]

        return df
