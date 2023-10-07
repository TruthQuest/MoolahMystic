#########################
"""
This code and its contents are the property of the author, Eric Brattin, alone 
and do not represent any group or organization, whether affiliated or not.
 
Any views or opinions expressed in this code are solely those of the author and do not reflect 
the views or opinions of any group or organization. 

This code is provided as-is, without warranty of any kind, express or implied.
"""
#########################


import importlib
import time
import datetime
import hashlib
import subprocess
import sys
import os
import pkg_resources
import cProfile
import pstats
from multiprocessing import Pool, cpu_count
from functools import lru_cache
import joblib
from joblib import Memory, dump, load
from tqdm import tqdm
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import getpass


packages = [
    ("aiohttp", "aiohttp", []),
    ("asyncio", "asyncio", []),
    ("warnings", "warnings", []),
    ("datetime", None, ["datetime"]),
    ("datetime", None, ["timedelta"]),
    ("concurrent.futures", None, ["ThreadPoolExecutor"]),
    ("tabulate", None, ["tabulate"]),
    ("sklearn.utils", None, ["murmurhash3_32"]),
    ("sklearn.ensemble", None, ["RandomForestRegressor"]),
    ("sklearn.linear_model", None, ["RidgeCV"]),
    ("sklearn.svm", None, ["LinearSVR"]),
    ("sklearn.impute", None, ["SimpleImputer"]),
    ("sklearn.pipeline", None, ["Pipeline"]),
    ("sklearn.pipeline", None, ["make_pipeline"]),
    ("sklearn.compose", None, ["ColumnTransformer"]),
    ("sklearn.ensemble", None, ["GradientBoostingRegressor"]),
    ("sklearn.ensemble", None, ["StackingRegressor"]),
    ("sklearn.base", None, ["BaseEstimator"]),
    ("sklearn.base", None, ["TransformerMixin"]),
    ("sklearn.model_selection", None, ["GridSearchCV"]),
    ("sklearn.exceptions", None, ["ConvergenceWarning"]),
    ("sklearn.linear_model", "LR", ["LinearRegression"]),
    ("sklearn.model_selection", None, ["train_test_split"]),
    ("gzip", "gzip", []),
    ("hashlib", "hashlib", []),
    ("nest_asyncio", "nest_asyncio", []),
    ("multiprocessing", "multiprocessing", []),
    ("numpy", "np", []),
    ("pandas", "pd", []),
    ("xgboost", "xgb", []),
    ("xgboost", None, ["XGBRegressor"]),
    ("shutil", "shutil", []),
]

def import_and_install(package):
    if isinstance(package, tuple):
        package = str(package[0])
    try:
        return importlib.import_module(package)
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
        return importlib.import_module(package)


def update_globals(packages):
    for package in packages:
        pkg, alias, submodules = (
            package + (None,))[:3] if isinstance(package, tuple) else (package, None, None)

        try:
            module = importlib.import_module(pkg)

            if alias:
                if isinstance(alias, tuple):
                    for a in alias:
                        globals()[a] = module
                else:
                    globals()[alias] = module

            if submodules:
                for submodule in submodules:
                    if submodule is not None:
                        globals()[submodule] = getattr(module, submodule)

            if pkg == "sklearn.linear_model" and alias == "LR":
                globals()["LR"] = getattr(module, submodules[0])

        except AttributeError:
            print(
                f"Error: The '{pkg}' package does not have the specified submodule(s).")

        except Exception as e:
            print(f"Error: {e}. Failed to import the '{pkg}' package.")


update_globals(packages)


url = 'https://www.federalreserve.gov/releases/h10/hist/dat00_ja.htm'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, self.columns]

async def fetch_data(url, headers):
    try:
        print("Fetching data...")
        start_time = time.time()  # Start timing

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                html = await response.text()
                print("Data fetched successfully.")

        duration = time.time() - start_time 
        print(f"Time taken fetch_data: {int(duration // 3600)} hours, {int((duration % 3600) // 60)} minutes, {int(duration % 60)} seconds")

        return pd.read_html(html)[0]
    except Exception as e:
        print(f"Error during data fetching: {e}")
        return None

def preprocess_and_load_data(url, headers):
    try:
        print("Loading and preprocessing data...")
        start_time = time.time()  

        loop = asyncio.get_event_loop()
        df = loop.run_until_complete(fetch_data(url, headers))
        if df is None:
            print("Failed to load data.")
            return None
        df = df[df['Rate'] != 'ND']
        df.loc[:, 'Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%B, %d, %Y")
        print("Data loaded and preprocessed successfully.")

        duration = time.time() - start_time  
        print(f"Time taken preprocess_and_load_data: {int(duration // 3600)} hours, {int((duration % 3600) // 60)} minutes, {int(duration % 60)} seconds")
        # Print the head of df
        print("Head of df:")
        print(tabulate(df.head(10), headers='keys', tablefmt='orgtbl'))
        return df
    except Exception as e:
        print(f"Error during data loading and preprocessing: {e}")
        return None

async def fetch_new_data(url, headers):
    try:
        print("Fetching new data...")
        start_time = time.time()  
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                html = await response.text()
                print("Data fetched successfully.")
                end_time = time.time()  
                elapsed_time = end_time - start_time
                hours = int(elapsed_time // 3600)
                minutes = int((elapsed_time % 3600) // 60)
                seconds = int(elapsed_time % 60)
                print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")  # Print time taken
                return pd.read_html(html)[0]
    except Exception as e:
        print(f"Error during data fetching: {e}")
        return None

def new_preprocess(new_df):
    try:
        print("Performing new data preprocessing...")
        start_time = time.time()  
        if isinstance(new_df, pd.DataFrame):
            mask = new_df.apply(lambda row: row.astype(str).str.contains(
                'table|update', case=False)).any(axis=1)
            new_df = new_df[~mask]
            new_df = new_df.iloc[:, :2].head(10).reset_index(drop=True)
            new_df.columns = ['Date', 'Rate']
            new_df = new_df[~new_df['Rate'].astype(
                str).str.contains('dollar', case=False)]
            new_df['Rate'] = new_df['Rate'].str.extract(
                '(\d+\.\d+)', expand=False).astype(float)
            new_df['Date'] = pd.to_datetime(
                new_df['Date'], format="%A %d %B %Y").dt.strftime('%Y-%m-%d')
            new_df['Date'] = pd.to_datetime(
                new_df['Date']).dt.strftime('%B, %d, %Y')
            print("New data preprocessing completed successfully.")
        else:
            print("Invalid data format. Data preprocessing skipped.")
        end_time = time.time()  
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")  
        print("\nHead of new_df:")
        print(tabulate(new_df.head(10), headers='keys', tablefmt='orgtbl'))
    except Exception as e:
        print(f"Error during new data preprocessing: {e}")
        return None
    return new_df

def merge_and_process_data(df, new_df):
    try:
        print("Merging and processing data...")
        merged_df = pd.concat([df, new_df], ignore_index=True)
        merged_df = merged_df.assign(Date=pd.to_datetime(merged_df['Date']).dt.date).sort_values('Date')

        if len(merged_df) <= len(df):
            print("Merging and processing data failed: No new data added.")
            return None

        merged_df['Date String'] = pd.to_datetime(merged_df['Date']).dt.strftime('%B, %d, %Y')
        merged_df = merged_df.drop_duplicates(subset='Date')

        cols = ['Date String', 'Date'] + [col for col in new_df.columns if col not in ['Date String', 'Date']]
        processed_df = merged_df[cols]
        print("Data merging and processing completed successfully.")
        return processed_df
    except Exception as e:
        print(f"Error during data merging and processing: {e}")
        return None


def sequential_predictions(model, X, y, future_days):
    try:
        print("Performing sequential predictions...")
        predictions = []
        for i in range(future_days):
            model.fit(X, y)
            next_day_features = X[-1:].copy()
            next_day_prediction = model.predict(next_day_features)
            predictions.append(next_day_prediction)

            X = np.vstack((X, next_day_features))
            y = np.append(y, next_day_prediction)

        print("Sequential predictions completed successfully.")
        return predictions
    except Exception as e:
        print(f"Error during sequential predictions: {e}")
        return None

def get_purchase_rate(df, purchase_date):
    try:
        print("Retrieving the exchange rate at the purchase date...")
        purchase_row = df[df['Date String'] == purchase_date]
        if purchase_row.empty:
            print(f"No data available for date: {purchase_date}")
            return None
        else:
            print("Exchange rate retrieval successful.")
            return purchase_row['Rate'].values[0]
    except Exception as e:
        print(f"Error during exchange rate retrieval: {e}")
        return None

def calculate_earnings(prediction, purchase_rate, amount=100):
    try:
        print("Calculating earnings...")
        if purchase_rate is not None:
            earnings = (prediction * amount) - (purchase_rate * amount)
            return np.round(earnings, 2)
        else:
            return None
    except Exception as e:
        print(f"Error during earnings calculation: {e}")
        return None

def send_predictions_table_email():
    dates = [(datetime.now() + timedelta(days=i)).strftime("%B %d, %Y") for i in range(1, 6)]
    rf_predictions = sequential_predictions(pipeline_rf, X_train, y_train, 5)
    gbm_predictions = sequential_predictions(pipeline_gbm, X_train, y_train, 5)
    xgb_predictions = sequential_predictions(pipeline_xgb, X_train, y_train, 5)
    stacking_predictions = sequential_predictions(pipeline_stacking, X_train, y_train, 5)
    
    table = list(zip(dates, rf_predictions, gbm_predictions, xgb_predictions, stacking_predictions))
    df = pd.DataFrame(table, columns=['Date', 'RF_Prediction', 'GBM_Prediction', 'XGB_Prediction', 'Stacking_Prediction'])

    print(tabulate(table, headers=['Date', 'Random Forest', 'GBM', 'XGBoost', 'Stacking'], tablefmt='orgtbl'))

    current_date = datetime.datetime.now().strftime('%B_%d_%Y').lower()

    filename = f"{current_date}_predictions.xlsx"
    df.to_excel(filename, index=False)

    sender_email = getpass.getpass("Enter your email address: ")
    sender_password = getpass.getpass("Enter your email password: ")
    receiver_email = 'ebrattin@gmail.com'

    subject = 'Predictions Table'
    body = 'Please find the predictions table attached.'

    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject

    message.attach(MIMEText(body, 'plain'))

    with open(filename, 'rb') as attachment:
   p
        p = MIMEBase('application', 'octet-stream')
        p.set_payload(attachment.read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', f"attachment; filename= {filename}")
        message.attach(p)

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(sender_email, sender_password)

    text = message.as_string()
    server.sendmail(sender_email, receiver_email, text)

    server.quit()    


    
def feature_engineering_batch(df_batch, model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, transformer):
    batch_size = len(df_batch)
    if batch_size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    X_today = df_batch[['SMA_20', 'SMA_50']].values
    start_time = time.time()  
    predictions_lr, predictions_rf, predictions_gbm, predictions_xgb, predictions_stacking = train_and_predict(
        model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, X_today, transformer
    )
    duration = time.time() - start_time  
    print(f"Time taken for feature_engineering_batch: {duration} seconds")

    return predictions_lr, predictions_rf, predictions_gbm, predictions_xgb, predictions_stacking


def feature_engineering(df, model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, transformer, batch_size=100):
    try:
        print("Performing feature engineering...")

        n = len(df)
        batch_count = n // batch_size
        remaining_rows = n % batch_size

        predictions_lr = np.empty(n)
        predictions_rf = np.empty(n)
        predictions_gbm = np.empty(n)
        predictions_xgb = np.empty(n)
        predictions_stacking = np.empty(n)

        def hash_row(row):
            row_bytes = row.values.tobytes()
            return hashlib.sha256(row_bytes).hexdigest()

        processed_rows = np.zeros(n, dtype=bool)
        start_time = time.time()

        for i in range(batch_count + 1):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n)
            df_batch = df.iloc[start_idx:end_idx]

            print(f"Processing batch {i + 1} of {batch_count + 1}...")

            df_batch['Hash'] = df_batch.apply(hash_row, axis=1)
            mask = ~df_batch['Hash'].duplicated()
            df_batch_filtered = df_batch[mask]

            batch_lr, batch_rf, batch_gbm, batch_xgb, batch_stacking = feature_engineering_batch(
                df_batch_filtered, model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, transformer
            )

            start_idx_filtered = start_idx + len(df_batch_filtered)
            end_idx_filtered = end_idx + len(df_batch_filtered)

            predictions_lr[start_idx_filtered:end_idx_filtered] = batch_lr
            predictions_rf[start_idx_filtered:end_idx_filtered] = batch_rf
            predictions_gbm[start_idx_filtered:end_idx_filtered] = batch_gbm
            predictions_xgb[start_idx_filtered:end_idx_filtered] = batch_xgb
            predictions_stacking[start_idx_filtered:end_idx_filtered] = batch_stacking

            processed_rows[start_idx:end_idx] = True

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        df['LR_Prediction'] = predictions_lr
        df['RF_Prediction'] = predictions_rf
        df['GBM_Prediction'] = predictions_gbm
        df['XGB_Prediction'] = predictions_xgb
        df['Stacking_Prediction'] = predictions_stacking

        print("Feature engineering completed.")
        print(f"Time taken for Feature engineering: {hours} hours, {minutes} minutes, {seconds} seconds")  # Print time taken

        return df

    except Exception as e:
        print(f"Error during feature engineering: {e}")
        return None


def prepare_models_and_data(df):
    try:
        print("Preparing models and data...")

        def initialize_models():

            model_lr = LR()
            model_rf = RandomForestRegressor(random_state=42)
            model_gbm = GradientBoostingRegressor(random_state=42)
            model_xgb = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            model_stacking = StackingRegressor(
                estimators=[('lr', model_lr), ('rf', model_rf), ('gbm', model_gbm), ('xgb', model_xgb)],
                final_estimator=LinearSVR(max_iter=10000)
            )
    
            return model_lr, model_rf, model_gbm, model_xgb, model_stacking

        print("Initializing models...")
        start_time = time.time() 
        model_lr, model_rf, model_gbm, model_xgb, model_stacking = initialize_models()
        end_time = time.time()  
        elapsed_time = end_time - start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)

        print("Models initialized.")

        df['SMA_20'] = df['Rate'].rolling(window=20).mean()
        df['SMA_50'] = df['Rate'].rolling(window=50).mean()

        X_train = df[['SMA_20', 'SMA_50']].values
        y_train = df['Rate']
        imputer = SimpleImputer()
        X_train_imputed = imputer.fit_transform(X_train)

        today_features = df[['SMA_20', 'SMA_50']].tail(1)
        today_features_imputed = imputer.transform(today_features)
        transformer = ColumnSelector(columns=['SMA_20', 'SMA_50'])
    
        print("Models and data preparation completed successfully.")
        print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")
      

    except Exception as e:
        print(f"Error during models and data preparation: {e}")
        return None
    
    return model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, today_features_imputed, transformer

def train_and_predict_single(args):
    model, model_name, X, y, today_features = args
    model_path = f"{model_name}.joblib"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        print(f"Fitting {model_name} model...")
        
        if model_name == 'model_xgb':
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X, y)
            
        print(f"{model_name} model fitting completed.")
        joblib.dump(model, model_path)
    return model.predict(today_features)

if __name__ == '__main__':


    def train_and_predict(model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train, y_train, today_features, transformer):
        try:
            today_features_df = pd.DataFrame(today_features, columns=['SMA_20', 'SMA_50'])
            today_features_transformed = transformer.transform(today_features_df)
    
            models = [model_lr, model_rf, model_gbm, model_xgb, model_stacking]
            model_names = ['model_lr', 'model_rf', 'model_gbm', 'model_xgb', 'model_stacking']
            X_train_list = [X_train] * 5
            y_train_list = [y_train] * 5
            today_features_list = [today_features_transformed] * 5
    
            start_time = time.time()
            with Pool(processes=min(5, cpu_count())) as pool:
                results = pool.map(train_and_predict_single, zip(models, model_names, X_train_list, y_train_list, today_features_list))
            end_time = time.time()  
            elapsed_time = end_time - start_time
            hours = int(elapsed_time // 3600)
            minutes = int((elapsed_time % 3600) // 60)
            seconds = int(elapsed_time % 60)
            print(f"Time taken: {hours} hours, {minutes} minutes, {seconds} seconds")
    
            return tuple(results)
    
        except Exception as e:
            print(f"Error during training and prediction: {e}")
            return None, None, None, None, None

    def train_model(df):
        X_train = df[['SMA_20', 'SMA_50']].values
        y_train = df['Rate']

        pipeline_rf = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestRegressor(random_state=42))
        ])
        
        pipeline_gbm = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', GradientBoostingRegressor(random_state=42))
        ])
        
        pipeline_xgb = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
        ])
        
        pipeline_stacking = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', StackingRegressor(
                estimators=[('lr', model_lr), ('rf', model_rf), ('gbm', model_gbm), ('xgb', model_xgb)],
                final_estimator=LinearSVR(max_iter=10000)
            ))
        ])
        pipeline_rf.fit(X_train, y_train)
        pipeline_gbm.fit(X_train, y_train)
        pipeline_xgb.fit(X_train, y_train)
        pipeline_stacking.fit(X_train, y_train)
    
        return pipeline_rf, pipeline_gbm, pipeline_xgb, pipeline_stacking

if __name__ == '__main__':
    multiprocessing.freeze_support()
   
    

    profiler = cProfile.Profile()
    profiler.enable()
    
    nest_asyncio.apply()
    
    df = preprocess_and_load_data(url, headers)
    
    new_url = 'https://www.exchangerates.org.uk/USD-JPY-exchange-rate-history.html'
    loop = asyncio.get_event_loop()
    new_df = loop.run_until_complete(fetch_new_data(new_url, headers))
    new_df = new_preprocess(new_df)
    
 
    df = merge_and_process_data(df, new_df)

    model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, today_features_imputed, transformer = prepare_models_and_data(df)
    
    df=train_model(df)

    today_prediction_lr, today_prediction_rf, today_prediction_gbm, today_prediction_xgb, today_prediction_stacking = train_and_predict(
        model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, today_features_imputed, transformer
    )

    df = feature_engineering(df, model_lr, model_rf, model_gbm, model_xgb, model_stacking, X_train_imputed, y_train, transformer)
    

    purchase_date = "May, 10, 2023"
    amount_bought = 100 

    purchase_rate = get_purchase_rate(df, purchase_date)
    
    earnings_lr = calculate_earnings(
        np.round(today_prediction_lr, 2), purchase_rate, amount_bought)
    earnings_rf = calculate_earnings(
        np.round(today_prediction_rf, 2), purchase_rate, amount_bought)
    earnings_gm = calculate_earnings(
        np.round(today_prediction_gbm, 2), purchase_rate, amount_bought)
    earnings_stk = calculate_earnings(
        np.round(today_prediction_stacking, 2), purchase_rate, amount_bought)
    
    if earnings_lr is not None:
        print(
            f"If you bought ${amount_bought} of currency on {purchase_date},")
        print(
            f"Based on the Linear Regression Prediction, you would now have: ${amount_bought + earnings_lr}")
    if earnings_rf is not None:
        print(
            f"Based on the Random Forest Prediction, you would now have: ${amount_bought + earnings_rf}")
    if earnings_gm is not None:
        print(
            f"Based on XGBoost, you would now have: ${amount_bought + earnings_gm}")
    if earnings_stk is not None:
        print(
            f"Based on Stacking, you would now have: ${amount_bought + earnings_stk}")
    
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    profiler.print_stats(sort='cumulative')

    send_predictions_table_email()
