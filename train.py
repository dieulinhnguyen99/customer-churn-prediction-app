import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import recall_score, f1_score, roc_auc_score, accuracy_score, \
    confusion_matrix, classification_report
from xgboost import XGBClassifier
import optuna
import os
import logging
from datetime import datetime
import joblib
import plotly.graph_objects as go
import plotly.figure_factory as ff
import sys
import pandas as pd

class Cleaner():
    """
    A component in the Machine Learning Pipeline
    Clean the dataset with predefined functions
    """

    def __init__(self, cleaning_functions=[]):
        """
        Constructor

        Params:
            cleaning_functions: A list of tuples of a cleaning function and its description
        """
        self.cleaning_functions = cleaning_functions
        self.logger = None
    
    def add(self, cleaning_function, description):
        """
        Add a cleaning function and its description

        Params:
            cleaning_function: A cleaning function
            description: A description
        """
        self.cleaning_functions.append((cleaning_function, description))

    def clean(self, df):
        """
        Clean the dataframe with the cleaning functions

        Params:
            df: A dataframe

        Return:
            A cleaned dataframe
        """
        cleaned_df = df.copy()
        if self.logger is not None:
            self.logger.log_info('Cleaning...')
            self.logger.log_info(f'Before cleaning: {len(cleaned_df)} rows')
        else:
            print('Cleaning...')
            print(f'Before cleaning: {len(cleaned_df)} rows')
        for cleaning_function, description in self.cleaning_functions:
            cleaned_df = cleaning_function(cleaned_df)
            if self.logger is not None:
                self.logger.log_info(f'{description}: {len(cleaned_df)} rows')
            else:
                print(f'{description}: {len(cleaned_df)} rows') 
        
        return cleaned_df
    
    def get_infos(self):
        """
        Print the description of each cleaning function
        """
        for _, description in self.cleaning_functions:
            print(description)
            
    def config(self, logger):
        """
        Configure the logger
        """
        self.logger = logger
            
def convert_to_numeric(df, columns=[]):
    """
    Convert variable to a numerical data type.
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


class FeatureProcessor():
    """
    A component in the Machine Learning Pipeline
    Process featues with predefined processing functions
    """
    def __init__(self, processing_functions=[]):
        """
        Constructor

        Params:
            processing_functions: A list of tuples of a processing function and its description
        """
        self.processing_functions = processing_functions
        self.logger = None
     
    def process(self, df):
        """
        Process features of dataframe with the processing functions

        Params:
            df: A dataframe

        Return:
            A processed dataframe
        """
        if self.logger is not None:
            self.logger.log_info('Processing features...')
        processed_df = df.copy()
        for processing_function, description in self.processing_functions:
            processed_df = processing_function(processed_df)
            if self.logger is not None:
                self.logger.log_info(f'{description}: Done!')
            
        return processed_df
  
    def get_infos(self):
        """
        Print the description of each processing function
        """
        for _, description in self.processing_functions:
            print(description)
         
    def config(self, logger):
        """
        Configure the logger
        """
        self.logger = logger 


def drop_columns(df, columns=['customerID']):
    """
    Drop column customerID from the data set
    Params:
            df: A dataframe

        Return:
            A dropped dataframe
    """
    if len(columns) >0:
        df = df.drop(columns,axis=1)
    return df

#List of features that contain the yes/no variables
EN_FEATURES = ['Churn','Partner', 'Dependents','PhoneService', 'PaperlessBilling']

def process_encoding(df, encoding_columns = EN_FEATURES):
    """
    Convert yes/no variable into a binary variable, 1 for Yes and 0 for No
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



class DataSplitter():
    """
    A component in the Machine Learning Pipeline
    Split dataset into Train, Test, Valid set
    """
    def __init__(self, test_size=0.2):
        """
        Constructor

        Params:
           test_size: size of the test set
        """
        super().__init__()
        self.test_size = test_size
        self.logger = None
    
    def split(self, df):
        """
        Split the dataset into Train, Test, Valid set

        Params:
            df: A dataframe

        Return:
            features and labels of train, valid, test
        """
        if self.logger is not None:
            self.logger.log_info('Splitting...')

        X, y = self.__to_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        stratify=y, 
                                                        test_size=self.test_size, 
                                                        random_state=42)
        
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, 
                                                        stratify=y_train, 
                                                        test_size=self.test_size, 
                                                        random_state=42)
        if self.logger is not None:
            self.logger.log_info(f'X_train shape: {X_train.shape}')
            self.logger.log_info(f'Train distribution: {(y_train == 0).sum()}, {(y_train == 1).sum()}')
            self.logger.log_info(f'X_val shape: {X_val.shape}')
            self.logger.log_info(f'Val distribution: {(y_val == 0).sum()}, {(y_val == 1).sum()}')
            self.logger.log_info(f'X_test shape: {X_test.shape}')
            self.logger.log_info(f'Test distribution: {(y_test == 0).sum()}, {(y_test == 1).sum()}')
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def __to_xy(self, df):
        """
        Convert dataset into features and labels
        """
        y = df[['Churn']].to_numpy().reshape(-1,)
        X = df.drop(['Churn'], axis=1)
        return X, y
    
    def config(self, logger):
        """
        Configure the logger
        """
        self.logger = logger

 
class HyperparameterTuner():
    """
    A component in the Machine Learning Pipeline
    Tune hyperparamter of model
    """
    def __init__(self, model_generator, trainer, metric, n_trials=500):
        """
        Constructor

        Params:
           model_generator: an instance of ModelGenerator
           trainer: an instance of Trainer
           metric: a metric used to evaluate model
        """
        super().__init__()
        self.model_generator = model_generator
        self.trainer = trainer
        self.metric = metric
        self.n_trials = n_trials
        self.logger = None 

    def __objective(self, trial, data):
        """
        Objective function for tuning hyperparameter in Optuna
        Params:
           trial
           data: features label of train, valid set
           
        """
        model = self.model_generator.optimize(trial)
        X_train, y_train, X_val, y_val = data[0], data[1], data[2], data[3]
        weight_ratio = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 1]))
        w_array = np.ones(y_train.shape[0])
        w_array[y_train==1] = weight_ratio
        self.trainer.fit(model, 
                    (X_train, y_train), 
                    (X_val, y_val),
                    sample_weight=w_array)
        p_val = self.trainer.model.predict_proba(X_val)[:, 1]
        y_pred_val = p_val > self.trainer.threshold
        y_pred_val = y_pred_val.astype(int)

        if self.metric == 'f1':
            f1_val = f1_score(y_val, y_pred_val, average="weighted")
            return f1_val
        elif self.metric == 'recall':
            return recall_score(y_val, y_pred_val)
        elif self.metric == 'roc_auc':
            return roc_auc_score(y_val, p_val)
        elif self.metric == 'f1_roc_auc':
            f1 = f1_score(y_val, y_pred_val, average="weighted")
            p_val = model.predict_proba(X_val)[:, 1]
            roc_auc = roc_auc_score(y_val, p_val)
            recall = recall_score(y_val, y_pred_val)
            res = roc_auc*0.2 + f1*0.4 + recall*0.4
            return res
 
    def tune(self, data):
        """
        Tune the hyperparameter 
        """
        if self.logger is not None:
            self.logger.log_info("Tuning...")
        #Propagate logs to the root logger.
        optuna.logging.enable_propagation() 
        #Stop showing logs in sys.stderr 
        optuna.logging.disable_default_handler()  
        if self.metric in ['f1', 'recall', 'roc_auc', 'f1_roc_auc']:
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction="maximize", sampler=sampler)
            study.optimize(lambda x: self.__objective(x, data), n_trials=self.n_trials)
            if self.logger is not None:
                self.logger.log_info(f"Best hyperparameters: {study.best_params}")
            return study.best_params
    
     
    def config(self, logger):
        """
        Configure the logger
        """
        self.logger = logger

class ModelGenerator():
    """
    A component in the Machine Learning Pipeline
    Generate Model
    """
    def __init__(self, create_model, get_param_space):
        """
        Constructor

        Params:
           create_model: function creating model
           get_param_space: function that returns the hyperparamter space

        """
        self.create_model = create_model
        self.get_param_space = get_param_space

    def generate(self, params):
        """
        Generate Model
        """
        model = self.create_model(**params) 
        
        return model
    
    def optimize(self, trial):
        """
        Generate Model for optimization
        """
        params = self.get_param_space(trial)
        model = self.create_model(**params)
        
        return model

class Trainer():
    """
    A component in the Machine Learning Pipeline
    Train Model
    """
    def __init__(self, eval_metric, threshold=0.5, visualizer=None, flags={}):
        """
        Constructor

        Params:
           eval_metric: evaluation metric
           threshold: threshold for defining positive labels
           visualizer: an instance of Visualizer
           flags:

        """
        self.visualizer = visualizer
        self.flags = flags
        self.eval_metric = eval_metric
        self.threshold = threshold
        self.metrics = {'f1': self.__f1_eval, 
                        'roc_auc': 'auc'}
        self.__is_fit = False
        self.logger = None

    def fit(self, model, train_data, val_data, sample_weight=None):
        """
        Fit model  

        Params:
            train_data: features and labels of train set
            val_data: features and labels of valid set
            sample_weight: weights for classes
        """
        self.model = model
        X_train, y_train = train_data[0], train_data[1]
        self.model.fit(X_train, y_train, 
                       eval_set=[train_data, val_data],
                       eval_metric=self.metrics[self.eval_metric],
                       sample_weight=sample_weight,
                       **self.flags)
        self.__is_fit = True
    
    def evaluate(self, X, y, target_names, name=''):
        """
        Evaluate model

        Params:
            X: features
            y: labels
            target_names: names of targets
        """
        if self.__is_fit:
            if self.logger is not None:
                self.logger.log_info(f'Evaluating {name}...')
            p = self.model.predict_proba(X)[:, 1]
            y_pred = p > self.threshold
            y_pred = y_pred.astype(int)
            accuracy = accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="weighted")
            roc_auc = roc_auc_score(y, p)
            cm = confusion_matrix(y, y_pred)[::-1]
            clf_report = classification_report(y, y_pred, target_names=target_names, output_dict=True)
            clf_report_df = pd.DataFrame(clf_report).round(3)
            if self.visualizer is not None:
                self.visualizer.plot_confusion_matrix(z=clf_report_df.iloc[:-1, :-2].T.values.tolist(), 
                                                      x=clf_report_df.iloc[:-1, :-2].index.values.tolist(), 
                                                      y=clf_report_df.iloc[:-1, :-2].columns.values.tolist(),
                                                      plot_name='classification_report_'+name)
                self.visualizer.plot_confusion_matrix(z=cm, 
                                                      x=["predicted "+_ for _ in target_names], 
                                                      y=["true "+_ for _ in target_names[::-1]],
                                                      plot_name='confusion_matrix_'+name)
            res = {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}  
            if self.logger is not None:
                self.logger.log_info(f'{name}: {res}')
            return res
        else:
            if self.logger is not None:
                self.logger.log_info(f'Call fit method first!')
    
    def __f1_eval(self, predt, dtrain):
        """
        Evaluate F1 score

        Params:
            predt: prediction
            dtrain: train set
        """
        y = dtrain.get_label()
        res = f1_score(y, predt > self.threshold, average="weighted")
        
        return 'f1_err', 1-res

    def predict(self, X):
        """
        Predict from the features

        Params:
            X: features

        """
        if self.__is_fit:
            p = self.model.predict_proba(X)[:, 1]
            y_pred = p > self.threshold

            return y_pred.astype(int)
        else:
            if self.logger is not None:
                self.logger.log_info(f'Call fit method first!')

    
    def config(self, logger):
        """
        Configure the logger
        """
        self.logger = logger


class Visualizer():
    """
    A component in the Machine Learning Pipeline
    Plotting

    """
    def __init__(self):
        """
        Constructor
        """
        self.logger = None
        self.output_path = None
    
    def config(self, output_path, logger):
        """
        Configure the output path and logger 
        """
        self.output_path = output_path
        self.logger = logger
    
    def plot_confusion_matrix(self, z, x, y, plot_name='', colorscale='reds', showscale=True):
        """
        plot confusion matrix
        """
        if self.logger is not None:
            self.logger.log_info(f'Plotting {plot_name}...')
        fig = ff.create_annotated_heatmap(z=z, x=x, y=y, 
                                          colorscale=colorscale, 
                                          showscale=True)
        fig = fig.update_layout(autosize=False,
                                width=400, height=250,
                                margin=dict(l=5, r=5, b=10, t=10, pad=4))
        if self.output_path is not None:
            fig.write_image(f"{self.output_path}/{plot_name}.png")

     
    def plot_feature_importances(self, feature_importances, feature_names):
        """
        plot feature importances   
        """
        if self.logger is not None:
            self.logger.log_info(f'Plotting feature importances...')
        inds = np.argsort(feature_importances)
        thresholds = feature_importances[inds]
        feature_names = feature_names[inds]
        fig = go.Figure(go.Bar(x=thresholds,
                               y=feature_names, 
                               marker_color='rgb(255, 12, 122)',
                               orientation='h'))
        fig = fig.update_layout(autosize=False,
                                width=800,
                                height=2000,
                                margin=dict(l=5, r=5, b=10, t=10, pad=4))
        if self.output_path is not None:
            fig.write_image(f"{self.output_path}/feature_importances.png")
        

class Logger():
    """
    A component in the Machine Learning Pipeline
    Logging

    """
    def config(self, log_dir):
        """
        Configure the directory of Logging file
        """
        self.log_dir = log_dir
        log_file = os.path.join(self.log_dir, 'pipeline.log')
        self.reset()
        logging.basicConfig(filename=log_file, 
                            format='%(asctime)s - %(message)s', 
                            level=logging.INFO)
    
    def log_info(self, info):
        """
        Log information
        """
        logging.info(info)

    def log_model(self, model):
        """
        save model 
        """
        model.save_model(os.path.join(self.log_dir, 'model.json'))
        self.log_info("Saved model")
        
    def log_scaler(self, scaler):
        """
        save Scaler
        """
        joblib.dump(scaler, os.path.join(self.log_dir, 'scaler.joblib')) 
        self.log_info("Saved scaler")

    
    def log_csv(self, df, index, name):
        """
        save dataframe as csv file
        """
        df.to_csv(os.path.join(self.log_dir, f'{name}.csv'), index=index)
        self.log_info(f"Saved CSV {name}")
 
    def reset(self):
        """
        reset the logger 
        """
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)


class DowngradeTrainingPipeline():
    """
    Machine Learning Piple for Customer Churn prediction
    """
    def __init__(self, cleaner, processor, splitter, 
                 scaler, model_generator, trainer, 
                 visualizer, tuner, logger, name=''):
        """
        Constructor

        Params:
           cleaner: an instance of Cleaner
           processor: an instance of FeaturesProcessor
           splitter: an instance of Dataplitter
           scaler: an instance of scikit-learn Scaler
           model_generator: an instance of ModelGenerator
           trainer: an instance of Trainer
           visualizer: an instance of Visualizer
           tuner: an instance of HyperparameterTuner
           logger: an instance of Logger

        """
        self.processor = processor
        self.cleaner = cleaner
        self.splitter = splitter
        self.scaler = scaler
        self.tuner = tuner
        self.model_generator = model_generator
        self.trainer = trainer
        self.visualizer = visualizer
        self.logger = logger
        self.__config()
        self.logger.log_info(name)
    
    def run(self, df, dropped_features, target_names):
        """
        Run the Machine Learning Pipeline
        """
        #Clean data
        data = self.cleaner.clean(df.copy())
        #Process data
        data = self.processor.process(data)
        #Split the dataset
        X_train, X_val, X_test, y_train, y_val, y_test = self.splitter.split(data)
        
        if self.logger is not None:
            self.logger.log_info(f'Feature names: {X_train.columns.values}')
            
        feature_names = X_train.columns.values
        self.scaler = self.scaler.fit(X_train)
        X_train = self.scaler.transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        best_params = self.tuner.tune((X_train, y_train, X_val, y_val))
        
        weight_ratio = float(len(y_train[y_train == 0]))/float(len(y_train[y_train == 1]))
        w_array = np.ones(y_train.shape[0])
        w_array[y_train==1] = weight_ratio
        
        model = self.model_generator.generate(best_params)
        self.trainer.fit(model, (X_train, y_train), (X_val, y_val), sample_weight=w_array)
        
        if self.logger is not None:
            self.logger.log_model(model)
            self.logger.log_scaler(self.scaler)
            
        train_res = self.trainer.evaluate(X_train, y_train, 
                                     target_names=target_names, 
                                     name='train')
        val_res = self.trainer.evaluate(X_val, y_val, 
                                   target_names=target_names, 
                                   name='val')
        test_res = self.trainer.evaluate(X_test, y_test, 
                                    target_names=target_names, 
                                    name='test')
        
        self.visualizer.plot_feature_importances(model.feature_importances_, feature_names)
        
        self.logger.reset()
        
        return best_params
    
    def __config(self):
        """
        Configure all components in Machine Learning Pipeline
        """
        if self.logger is not None:
            cwd = os.getcwd()
            self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.log_dir = os.path.join(cwd, self.timestamp)
            plot_dir = os.path.join(self.log_dir, 'plots')
            os.makedirs(self.log_dir)
            os.makedirs(plot_dir)
            self.logger.config(self.log_dir)
            self.visualizer.config(plot_dir, self.logger)
            self.cleaner.config(self.logger)
            self.splitter.config(self.logger)
            self.processor.config(self.logger)
            self.trainer.config(self.logger)
            self.tuner.config(self.logger)

def get_param_space(trial):
    """
    Get parameter space
    """
    params = {
        'early_stopping_rounds': trial.suggest_categorical('early_stopping_rounds', [50]),
    #         'tree_method':'gpu_hist',  # Use GPU acceleration
        'silent': trial.suggest_categorical('silent', [1]),
        # 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
        'booster': trial.suggest_categorical('booster', ['gbtree']),
        'lambda': trial.suggest_loguniform('lambda', 1e-6, 1),
        'alpha': trial.suggest_loguniform('alpha', 1e-6, 1),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.7, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.7, 1.0),
        'learning_rate': trial.suggest_uniform('learning_rate', 1e-6, 1),
        'n_estimators': trial.suggest_int("n_estimators", 1, 300),
        'max_depth': trial.suggest_int("max_depth", 1, 20),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        'gamma': trial.suggest_int('gamma', 0, 300),
        'random_state': trial.suggest_categorical('random_state', [42]), 
        'verbosity': trial.suggest_categorical('verbosity', [0]),
        'use_label_encoder': trial.suggest_categorical('use_label_encoder', [False])
    }
    return params

processing_functions = [(process_encoding, "Encoding"),(process_categorical_features, 'Categorical features')]
cleaning_functions = [(lambda x: convert_to_numeric(x, ["TotalCharges"]), 'Converting object column to type float64'),
                   (lambda x: fillna(x,{'TotalCharges': 0}), 'Replacing nan values with 0 from "TotalCharges" column'),
                   (drop_columns, 'Dropping column')]
flags = {'verbose': False}
dropped_features = []

def train(input, metric, n_trials, name):
    """
    Initiate components of pipeline and run the training pipeline

    Params:
        n_trials: number of trials
        name: name of the pipeline    
    Return:
        best_params: the best parameters
    """
    cleaner = Cleaner(cleaning_functions)
    processor = FeatureProcessor(processing_functions)
    splitter = DataSplitter()
    scaler = RobustScaler()
    model_generator = ModelGenerator(XGBClassifier, get_param_space)
    visualizer = Visualizer()
    trainer = Trainer(metric, threshold=0.5, visualizer=visualizer, flags=flags)
    tuner = HyperparameterTuner(model_generator=model_generator, 
                                trainer=trainer,
                                metric=metric, 
                                n_trials=n_trials)
    logger = Logger()
    df = pd.read_csv(input)
    training_pipeline = DowngradeTrainingPipeline(cleaner=cleaner,
                                                processor=processor,
                                                splitter=splitter,
                                                scaler=scaler,
                                                model_generator=model_generator,
                                                tuner=tuner, 
                                                trainer=trainer, 
                                                visualizer=visualizer,
                                                logger=logger,
                                                name=name)
    best_params = training_pipeline.run(df=df, 
                                        dropped_features=dropped_features,
                                        target_names=['No Churn','Churn'])

    return best_params


def main(arguments):
    """
    Main function
    """
    #Create parser
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    #Add arguments
    parser.add_argument(
        "--input", type=str,
        help='location of the input file'
    )
    parser.add_argument(
        "--name", type=str,
        help='name of the experiment'
    )

    #training specific parameters
    parser.add_argument('--n-trials', type=int, default=100,
                        help='number of trials for tuning hyperparameters')
    parser.add_argument('--metric', type=str, default='f1',
                        help='evaluation metric')

    #parse the arguments
    args = parser.parse_args(arguments)
    train(args.input, args.metric, args.n_trials, args.name)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))