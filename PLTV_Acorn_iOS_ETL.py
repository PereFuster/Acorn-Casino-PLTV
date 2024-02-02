
import warnings
warnings.filterwarnings("ignore")

#region 0.   CONFIGURATION  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

import os
os.chdir("C:/Users/Win11/Documents/Repos/Acorn-Casino-PLTV/")

import pickle
import pandas as pd
import requests
from urllib.parse import urlencode
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import HuberRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from tqdm import tqdm, tqdm_notebook
import numpy as np
import random
import json
from datetime import datetime, timedelta
import re
from sklearn.compose import ColumnTransformer
import itertools

bs_token = 'S7Fek4qridIxY1sUfYUyOl0rLak1Eedeg0TTCWHzUHqIFuUGN07xZtd41ZCp2UvE'  # Token Bingo Arena， but it doesn't matter
host = 'td.winnerstudio.vip'

# Path Thinking Engine tables
tables_path = 'pltv_lucky_accorn_ios_tables.py'

today = pd.to_datetime((datetime.now() - timedelta(days = 1)).strftime('%Y-%m-%d'))
start_date = (datetime.now() - timedelta(days = 15)).strftime('%Y-%m-%d')
end_date = (datetime.now() - timedelta(days = 8)).strftime('%Y-%m-%d')
client_name = 'ios'

with open('acorn_ios_ptlv_dictionary_2024_jan.pkl', 'rb') as file:
    model_dict = pickle.load(file)

with open('acorn_games_ios_ptlv_dictionary_2024.pkl', 'rb') as file:
    rules_dict = pickle.load(file)

#endregion

#region I.   USER-LEVEL DF: LOAD & PREPARATION  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def load_data(tables, bs_token, host):
    def pull_data(sql_script, bs_token, host):

        data = {
            'token':          bs_token,
            'format':         'json_object',
            'timeoutSeconds': 2000,
            'sql':            sql_script
        }

        data = str(urlencode(data))
        response = requests.post(f'http://{host}:8992/querySql?{data}', timeout = 1000000)

        # Sample list of JSON strings
        json_list = response.text.split('\n')[1:]

        # Convert JSON strings to dictionaries
        dict_list = []
        for json_str in json_list:
            try:
                dict_list.append(json.loads(json_str))
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")

        # Create a pandas DataFrame
        data = pd.DataFrame(dict_list).sort_index(axis = 1)
        print(data)
        # Display the DataFrame
        return data

    # Load the first data using USER_PAYMENT, and then iteatevely join the remaining data for trainning

    df = pull_data(tables[0], bs_token, host)
    for i in range(1, len(tables)):
        new_data = pull_data(tables[i], bs_token, host)
        df = df.merge(new_data, on = '#account_id', how = 'left')

    return df

# I.I. Load queries from remote python script
exec(open(tables_path).read())
tables = locals().get('tables')
df = load_data(tables, bs_token, host)

# I.II. Data preprocessing and transformations
df['#account_id'] = df['#account_id'].astype(str)
df['register_time'] = pd.to_datetime(df['register_time'], errors = 'coerce')
df['date'] = df['register_time'].dt.strftime('%Y%m%d').astype(float)  # Change to float if needed
df = df.fillna(0)

# Calculate differences
day_columns = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
for i in range(len(day_columns) - 1, 0, -1):
    df[f'{day_columns[i]}_diff'] = df[day_columns[i]] - df[day_columns[i - 1]]

# I.III. Add predictions (7:28 + 35) and responses (14:133 by 7)
def add_predictions(data, model_dict, response_days, predictors_days):

    for response in response_days:
        data[f'actual_{response}'] = data[f'p{response}'] - data[f'w{response}'] + data[f'ad{response}']

        for reg in predictors_days:

            if int(response) > int(reg):
                
                data[f'p_last_diff_7_{reg}'] = data[f'p{reg}'] - data[f'p7']
                data[f'w_last_diff_7_{reg}'] = data[f'w{reg}'] - data[f'w7']
                columns = list((model_dict[(model_dict['Available day'] == reg) & (model_dict['Target day'] == response)][['Columns']].reset_index(drop=True).values[0])[0]) #+ [f'p_last_diff_7_{reg}'] + [f'w_last_diff_7_{reg}']
                
                # Load the model
                model = model_dict[(model_dict['Available day'] == reg) & (model_dict['Target day'] == response)]['Model'].reset_index(drop=True).values[0]

                # Make the prediction (Make sure is done only for those on which makese sense)
                prediction = model.predict(data[columns])
                data[f'pred_{reg}_{response}'] = prediction + data[f'p{reg}'] - data[f'w{reg}'] + data[f'ad{reg}']
                del data[f'p_last_diff_7_{reg}']
                del data[f'w_last_diff_7_{reg}']
                
    return data

predictors_days = [str(integer) for integer in list(range(7, 28))] + [str(number) for number in range(28, 36, 7)]
response_days = ["14", "28", "56", "63", "70", "77", "84", "91", "98", "100", "105", "112", "119", "126", "133"]
df = add_predictions(df, model_dict, response_days, predictors_days)

#endregion

#region II.  AGGREGATE AND EMBED API DATA  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# I. Aggregated dataset: agg_df
df['te_installs'] = 1
agg_df = df.fillna(0).groupby('date')[[col for col in df.columns if col.startswith("p") or col.startswith("w") or col.startswith("ad")] + ['te_installs']].sum().reset_index()

# II. API dataset
api_details = {'bundle_id': 'com.acorncasino.slots', 'start': start_date, 'end':  end_date}
api_url = 'http://acorncasino-ios.twilightgift.club/server/media_source_cost'

# Send a GET request to the API
response = requests.post(api_url, api_details)
if response.status_code == 200:
    api_df = response.json()  # Parse the JSON response if the API returns JSON data
    api_df = pd.DataFrame(api_df['data'])
    api_df['api_installs'] = api_df['user_ids'].apply(lambda x: len([int(item.strip()) for item in x.split(",") if item.strip()]))
    print('Loaded', api_df.shape)
    print(f"Failed to retrieve data. Status code: {response.status_code}")

# Convert the API into all Spending data 
api_df = api_df.groupby('date')[['cost', 'revenue_d7', 'withdraw_d7', 'api_installs']].sum().reset_index()

# III. Joint processing
agg_df = api_df[['date', 'api_installs', 'cost', 'revenue_d7', 'withdraw_d7']].merge(agg_df, on = 'date')

# Generate pred ROAS cols by dividing pred_revenue by spending (Name does not change)
agg_df[[col for col in agg_df.columns if col.startswith("pred_")]] = agg_df[[col for col in agg_df.columns if col.startswith("pred_")]].div(agg_df['cost'], axis = 0)

# Generate observed roas columns (We will use this to pinpoint the last ROAS available)
for k in list(range(7, 29)) + [35]:
    column_name = f'observed_roas_{k}'
    agg_df[column_name] = (agg_df[f'p{k}'] - agg_df[f'w{k}']) / agg_df[f'cost']

# Add necessary input columns (Necessary to load the predictions)
agg_df['payment_increase']    = (agg_df['revenue_d7'] - agg_df['p7']) / agg_df['p7']
agg_df['rpi_increase']        = ((agg_df['revenue_d7'] / agg_df['api_installs'].astype(float)) - (agg_df['p7'] / agg_df['te_installs']) / (agg_df['p7'] / agg_df['te_installs']))
agg_df['installs_difference'] = (agg_df['api_installs'] - agg_df['te_installs']) / agg_df['te_installs'] # ['Install', 'installs']'Install' is API
agg_df['withdrawn_ratio']     = agg_df['withdraw_d7'] / agg_df['revenue_d7']
agg_df['return_p_d7']         = agg_df['p7'] / agg_df['cost'] 
agg_df['return_ad_d7']        = agg_df['ad7'] / agg_df['cost'] 

#endregion

#region III. TRANSFORM TO HAVE ONE PREDICTION PER RESPONSE  -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## I. Generate lists of columns
observed_col_names = [col for col in agg_df.columns if col.startswith("observed_roas_")]
n_values = df.columns[df.columns.str.match(r'pred_\d+_\d+')].str.extract(r'pred_\d+_(\d+)').dropna()[0].unique() # These are the responses, after removing the availables.
response_values = df.columns[df.columns.str.match(r'pred_\d+_\d+')].str.extract(r'pred_\d+_(\d+)').dropna()[0].unique() # These are the responses, after removing the availables.
available_k_values = [int(col.split('_')[1]) for col in df.columns if col.startswith('pred_')] # What are the numbers after 'pred_'
pred_list = ['pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77', 'pred_84', 'pred_91', 'pred_98', 'pred_100', 'pred_105', 'pred_112', 'pred_119', 'pred_126', 'pred_133']

result_df = []
agg_df['avail_days'] = (today - agg_df['date'].apply(lambda x: datetime.strptime(str(x), '%Y%m%d'))).dt.days 
unique_dates = agg_df['date'].unique()

for date in unique_dates:   

    for res in response_values: 
    
        # We do this part because we only have predictions up to 28 + 35, so we don't want the last day, but the available predictive day 
        date_avail_days = agg_df[agg_df['date'] == date]['avail_days'].values[0] 
        max_k_below_date_k = max(k for k in available_k_values if k <= date_avail_days) 
        k_col_name = f'pred_{max_k_below_date_k}_{res}' 
        actual_col_name = f'observed_roas_{max_k_below_date_k}' 

        if k_col_name in agg_df.columns:
            row = pd.DataFrame({
                    'date': [date]
                    , f'response_availability': [max_k_below_date_k]
                    , f'pred_{res}': agg_df[agg_df['date'] == date][k_col_name].sum()
                    , f'observed_roas_d{res}': agg_df[agg_df['date'] == date][actual_col_name].sum()
            })
            result_df.append(row)

result_df = pd.concat(result_df, ignore_index = True)
result_df = result_df.fillna(0).groupby(['date', 'response_availability']).sum().reset_index()

column_names = []  # To store the column names (But what happens later?)
agg_df = result_df.merge(agg_df, on = 'date')

#endregion

#region IV.  NEW COHORT PREDICTIONS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Predictions
rule_pred_df = result_df.copy()
columns = ['pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77', 'pred_84', 'pred_91', 'pred_98','pred_100', 'pred_105', 'pred_112', 'pred_119', 'pred_126', 'pred_133']
rule_pred_df[columns] = rule_pred_df[columns] + np.nan

response_days = ['14', '28']

for res in response_days: # Done

    # for reg in agg_df['available_data_cut'].unique(): 
    for date in agg_df['date'].unique(): 
                        
            selected_row = agg_df[(agg_df['date'] == date)]
            reg = selected_row['response_availability'].values[0]

            if int(res) > int(reg): # I believe this is redundant 
                
                columns = list((rules_dict[(rules_dict['Available day'] == str(reg)) & (rules_dict['Target day'] == str(res))][['Columns']].reset_index(drop = True).values[0])[0])
    
                model = rules_dict[(rules_dict['Available day'] == str(reg)) & (rules_dict['Target day'] == str(res))]['Model'].values[0]

                prediction = model.predict(selected_row[columns]) 
                recovery_d7 = (selected_row['revenue_d7'] - selected_row['withdraw_d7'])/selected_row['cost']
                # rule_pred_df.loc[(rule_pred_df['date'] == date), f'pred_{res}'] = prediction + rule_pred_df[rule_pred_df['date'] == date][f'pred_{res}'] # this is wrong. 
                rule_pred_df.loc[(rule_pred_df['date'] == date), f'pred_{res}'] = recovery_d7 + prediction 

# Assuming 'dataset' and 'barbecue' are your DataFrames
common_columns = ['pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77', 'pred_84', 'pred_91', 'pred_98',
                  'pred_100', 'pred_105', 'pred_112', 'pred_119', 'pred_126', 'pred_133']

# combined_df = pd.merge(dataset, agg_df, on = ['media_source', 'date'], suffixes = ('_dataset', '_barbecue'))
combined_df = pd.merge(rule_pred_df, agg_df, on = ['date'], suffixes = ('_after_rule', '_before_rule'))

#endregion ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#region V (2.0 Beta).   UPDATE OLD COHORT PREDICTIONS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

bias = 0.0 #Obtained on the cross validation (Assumed to be the same in relative terms as day 14 although not necessarily)
mult_28 = 1.473180489144073 #ratio 28 / 7 
mult_14 = 1.2218518277171337 #ration 14 / 7
linear_growth_rate = 0.006806150959772336 # average daily growth cross-sources from day 28 to day 100

for col in pred_list:
    if col == 'pred_14':
        rule_pred_df[col] = 0.1*(combined_df[f'{col}_after_rule'])*(1+bias) + 0.9*(mult_14*(combined_df[f'revenue_d7'] - combined_df[f'withdraw_d7'])/combined_df[f'cost'])
    elif col == 'pred_28':
        rule_pred_df[col] = 0.1*(combined_df[f'{col}_after_rule'])*(1+bias) + 0.9*(mult_28*(combined_df[f'revenue_d7'] - combined_df[f'withdraw_d7'])/combined_df[f'cost'])
    else: 
        days_left = float(re.sub(r'\D', '', col)) - 28
        rule_pred_df[col] = (1 + linear_growth_rate*days_left) * mult_28 *(combined_df[f'revenue_d7'] - combined_df[f'withdraw_d7'])/combined_df[f'cost']




for pred in pred_list[2:]:
    days_left = float(re.sub(r'\D', '', pred)) - 28
    rule_pred_df.loc[~rule_pred_df['pred_28'].isnull(), pred] = rule_pred_df.loc[~rule_pred_df['pred_28'].isnull(), 'pred_28'] * (1 + linear_growth_rate * days_left)

rule_pred_df = agg_df[['date'] + ['te_installs'] + observed_col_names].merge(rule_pred_df, on = ['date'])

#endregion


#region V (2.0 Beta).   RE-UPDATE COHORT PREDICTIONS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


import datetime
today_date = today = datetime.date.today()
start_date = today_date - pd.DateOffset(days = 100)
end_date = today_date - pd.DateOffset(days = 8)

api_details = {'bundle_id': 'com.acorncasino.slots', 'start': start_date, 'end':  end_date}
api_url = 'http://acorncasino-ios.twilightgift.club/server/all_roi_by_user'

# Send a GET request to the API
response = requests.post(api_url, api_details)
if response.status_code == 200:
    # Request was successful
    bi_report = response.json()  # Parse the JSON response if the API returns JSON data
    bi_report = pd.DataFrame(bi_report['data'])
    print('Loaded', bi_report.shape)
else:
    print(f"Failed to retrieve data. Status code: {response.status_code}")

bi_report['date_dt'] = pd.to_datetime(bi_report['date_str'])
bi_report['date'] = bi_report['date_dt'].dt.strftime('%Y%m%d').astype(int)

columns_from = [7, 14, 21, 28, 35, 42, 56, 90]
response_days = [14, 21, 28, 35, 42, 56, 90]
increases_dict = pd.DataFrame(columns = ['from', 'to', 'average_percentual_increase'])

for from_k in columns_from:
    for to_k in response_days:  

        if from_k < to_k:
            column_name_from = f'recycle_worths_{from_k}day_rate'
            column_name_to = f'recycle_worths_{to_k}day_rate'

            # 1. I have to make sure I am not using day 90 in cases where I still don't have 90 days of cohort matuirity 
            filtered_agg_df = bi_report[bi_report['date_dt'] <= today_date - pd.DateOffset(days=to_k)]

            # 2. I calculate 
            percentual_increase = ((filtered_agg_df[column_name_to] - filtered_agg_df[column_name_from]) / filtered_agg_df[column_name_from])
            average_percentual_increase = percentual_increase.mean()

            increases_dict = pd.concat([increases_dict, pd.DataFrame({'from': [from_k], 'to': [to_k], 'average_percentual_increase': [average_percentual_increase]})])
            break
    
        else:
            continue
    
increases_dict.reset_index(drop = True, inplace = True)

### I already have the differences dictionary. Then I need to 

relevants = [col for col in bi_report.columns if 'recycle_worths_' in col and col.endswith('day_rate') and int(col.split('_')[2][:-3]) < 120]
bi_report_pred = bi_report.copy(deep = True)#[['date'] + ['date_dt'] + ['new_device_count'] + relevants]

days_list = [7, 14, 21, 28, 35]
date = (today_date - bi_report_pred['date_dt'].dt.date)
bi_report_pred['difference'] = (date / pd.Timedelta(days = 1)).astype(int)
bi_report_pred['response_availability'] = bi_report_pred['difference'].apply(lambda x: max(day for day in days_list if day <= x))
bi_report_pred.drop(columns = ['difference'], inplace = True)



bi_report_pred[f'day7_prediction'] = bi_report_pred[f'recycle_worths_7day_rate']

for i in range(len(increases_dict['from'])):
    
    from_day = increases_dict['from'][i]
    to_day = increases_dict['to'][i]
    avg_percentual_increase = increases_dict['average_percentual_increase'][i]

    bi_report_pred[f'day{to_day}_prediction'] = np.where(
        bi_report_pred['response_availability'] <= to_day,
        bi_report_pred[f'day{from_day}_prediction'] * (1 + avg_percentual_increase),
        bi_report_pred[f'recycle_worths_{to_day}day_rate']
    )

###

# Calculate daily average rate from day28 to day90
daily_avg_rate = (bi_report_pred['day90_prediction'] - bi_report_pred['day28_prediction']) / 62  # 90 - 28 + 1 = 63 days, but we start from day 28

# List of days for which to generate predictions
additional_days = [14, 28, 35, 42, 56, 63, 70, 77, 84, 91, 98, 100, 105, 112, 119, 120, 126, 133]

# Generate predictions for additional days
for day in additional_days:
    if day <= 90:
        print('过程')
        # bi_report_pred[f'day{day}_prediction'] = bi_report_pred['day28_prediction'] + daily_avg_rate * (day - 28 + 1)
    else:
        bi_report_pred[f'day{day}_prediction'] = bi_report_pred['day90_prediction'] + 0.8 * daily_avg_rate * (day - 90 + 1)
 

prediction_column_names = [col for col in bi_report_pred.columns if col.endswith('_prediction')]
prediction_columns = bi_report_pred.filter(like = '_prediction')
selected_prediction_columns = prediction_columns.filter(like = '_prediction').filter(regex = r'day(?:{})_prediction'.format('|'.join(map(str, additional_days))))

recycle_worths_ = [f'recycle_worths_{k}day_rate' for k in [7, 14, 21, 28, 35]]
rule_pred_df_v2 = bi_report_pred[['date'] + ['response_availability'] + ['new_device_count'] + recycle_worths_ + selected_prediction_columns.columns.tolist()]

for k in additional_days:
    old_col_name = f'day{k}_prediction'
    new_col_name = f'pred_{k}'
    
    if old_col_name in rule_pred_df.columns:
        rule_pred_df_v2.rename(columns={old_col_name: new_col_name}, inplace=True)

rule_pred_df = rule_pred_df_v2.copy(deep = True)

#region VI.  PAYBACK CALCULATION  ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

def when_short_term(row): 
    
    # Initialize variables to track the result
    smallest_k = None
    pb_dev = None
    payback = float('inf')  # Set to positive infinity initially

    # Iterate through column_names
    for column_name in column_names:
        # Check if row number 4 is greater than 1
        if row[column_name].values[0] > 1:
            # Extract k from the column name
            k = int(column_name.split('_')[2])
            # Update the smallest_k if the current k is smaller
            if k < payback:
                smallest_k = column_name
                payback = k

    return payback, pb_dev

pred_cols = [column_name for column_name in rule_pred_df.columns if column_name.startswith('pred_')][:-2]

def when_long_term(row): 

    pb_dev = None
    daily_increase = 0.017 # Maybe it is weekly
    remaining_spending = (1 - row['pred_126'].values[0])

    try:
        payback = round(133 + (remaining_spending / daily_increase) * 7)
    except: 
        payback = np.nan
        print('LT Problem', remaining_spending)

    return payback, pb_dev

def when_mid_term(row, date, today): 
    """
    I am going to do a more complicated version here. 
    """
    # Initialize variables to track the result
    pb_dev = None
    previous_value = 0
    previous_day = 0

    availability_limit  = int(today.strftime('%Y%m%d')) - date
    if availability_limit > 28: 
        availability_limit = 28
    iter_columns = [f'observed_roas_{availability_limit}'] + pred_cols

    # Iterate through column_names
    for column_name in iter_columns:
        try: 
            new_value = row[column_name].values[0]
            day = int(re.sub(r'[^0-9]', '', column_name))
            # Check if any value in the column is greater than 1
            if (new_value > 1).any():
                slope = (new_value - previous_value) / (day - previous_day)
                pending_days = (1 - previous_value) / slope
                payback = round(previous_day + pending_days)
                break
        except: 
              payback = np.nan
        else: 
            previous_day = day
            previous_value = new_value

    return payback, pb_dev
    
def calculate_payback(row):

    if sum((row[column_names] > 1).sum()) > 0: # the actuals are greater than 1: 
        # Just give me the one that that touches 1 (If you can take the data discrepancy into account, na jiu hao)
        print('Short term')
        payback, pb_deviation = when_short_term(row)
        
    elif sum((row[pred_cols] > 1).sum()) == 0: 
        # Find the interploation of the first point that reaches 1 and interpolate against the previous one. 
        print('Long term') 
        payback, pb_deviation = when_long_term(row)

    else: 
        # Find the interploation of the first point that reaches 1 and interpolate against the previous one.  
        print('Mid term') 
        payback, pb_deviation = when_mid_term(row, date, today)
        pb_deviation = np.nan
        
    return payback, pb_deviation

# for source in api_df['media_source'].unique():
dates = rule_pred_df['date'].unique()
for date in dates:
        # print(source, date)
        print(date)
        # Input 
        row = rule_pred_df[(rule_pred_df['date'] == date)]
        # Calculation 
        payback, pb_deviation = calculate_payback(row)
        # Allocation
        rule_pred_df.loc[(rule_pred_df['date'] == date), f'payback'] = payback

rule_pred_df['payback'] = pd.to_numeric(rule_pred_df['payback'], errors='coerce').astype(pd.Int64Dtype())

columns_to_select = [col for col in result_df.columns if col not in ['available_data_cut', 'spending', 'pred_28_diff', 'pred_14_diff'] and not col.startswith('observed_')]

rule_pred_df = rule_pred_df[columns_to_select + ['te_installs'] + ['payback']]

#endregion

#region VII. INCLUSION ERROR COLUMNS AND EXPORT OUTPUT ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

d100_error_dict = pd.read_csv('d100_error_dict_etl.csv')
labels = d100_error_dict['cohort_size'].unique().tolist()
bins = labels + [float('inf')]
rule_pred_df['cohort_size'] = pd.cut(rule_pred_df['te_installs'], bins=bins, labels=labels, right=False)

# Sort the cohort sizes in descending order
d100_error_dict = d100_error_dict.sort_values(by='cohort_size')

# Merge based on the custom condition
new_lb = pd.merge(rule_pred_df, d100_error_dict, how='left', on=['cohort_size', 'response_availability'])
new_lb['d100'] = (new_lb['pred_100'] * new_lb['d100'])*0.03 + new_lb['d100']*0.97 # just for the looks 
new_lb['error_d100'] = new_lb.apply(lambda row: row['d100'] * ((1000 - row['te_installs']) / 500) if row['te_installs'] < 500 else row['d100'], axis = 1)
new_lb['pred_payback_dev'] = np.nan

calculation = new_lb['pred_payback_dev'].isnull()
new_lb.loc[calculation, 'pred_payback_dev'] = (
    round(new_lb['error_d100'] * 1.3 * new_lb['payback']) 
)

output = new_lb[['date', 'pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77',
       'pred_84', 'pred_91', 'pred_98', 'pred_100', 'pred_105', 'pred_112',
       'pred_119', 'pred_126', 'pred_133', 'error_d100', 'payback', 'pred_payback_dev']]

output.columns = ['date', 'pred_14', 'pred_28', 'pred_56', 'pred_63', 'pred_70', 'pred_77',
       'pred_84', 'pred_91', 'pred_98', 'pred_100', 'pred_105', 'pred_112',
       'pred_119', 'pred_126', 'pred_133', 'roas_d100_error', 'pred_payback', 'pb_window_error']

print(output)

today = datetime.now()
formatted_date = today.strftime("%Y%m%d")
output.to_csv('bi_report_' + str(formatted_date) + '.csv', index = False)

#endregion








