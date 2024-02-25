import pandas as pd
from statsforecast import StatsForecast
from utilsforecast.losses import mse
from utilsforecast.evaluation import evaluate
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import os
import math


# this makes it so that the outputs of the predict methods have the id as a column 
# instead of as the index
os.environ['NIXTLA_ID_AS_COL'] = '1'

def evaluate_cross_validation(df, metric):
    models = df.drop(columns=['unique_id', 'ds', 'cutoff', 'y']).columns.tolist()
    evals = []
    # Calculate loss for every unique_id and cutoff.    
    for cutoff in df['cutoff'].unique():
        eval_ = evaluate(df[df['cutoff'] == cutoff], metrics=[metric], models=models)
        evals.append(eval_)
    evals = pd.concat(evals)
    evals = evals.groupby('unique_id').mean(numeric_only=True) # Averages the error metrics for all cutoffs for every combination of model and unique_id
    evals['best_model'] = evals.idxmin(axis=1)
    return evals

# Output file to append the results
output_file_path = 'models_kpi_memory_dataset_rnd_one_month.csv'

# Load the dataset
file_path = 'unique_dataset_rnd.csv'
data = pd.read_csv(file_path, parse_dates=True)



# Creating the two univariate datasets with the corrected timestamp
memory_usage_dataset_with_corrected_timestamp = data[['timestamp', 'Memory usage [KB]', 'unique_id']].copy()
memory_usage_dataset_with_corrected_timestamp  = memory_usage_dataset_with_corrected_timestamp.rename(columns={'Memory usage [KB]': 'y', 'timestamp': 'ds'})


#Resample to hours
memory_usage_dataset_with_corrected_timestamp['ds'] = pd.to_datetime(memory_usage_dataset_with_corrected_timestamp['ds'])
memory_usage_dataset_with_corrected_timestamp.set_index('ds', inplace=True)
memory_usage_dataset_with_corrected_timestamp = memory_usage_dataset_with_corrected_timestamp.groupby('unique_id').resample('H').mean()
memory_usage_dataset_with_corrected_timestamp = memory_usage_dataset_with_corrected_timestamp.fillna(0)  # Fills NaN with 0

memory_usage_dataset_with_corrected_timestamp = memory_usage_dataset_with_corrected_timestamp.reset_index()

memory_usage_dataset_with_corrected_timestamp = memory_usage_dataset_with_corrected_timestamp.sort_values(by='ds').groupby('unique_id')

from statsforecast.models import (
    AutoARIMA,
    AutoTheta,
    AutoETS,
    AutoCES,
    MSTL,
    SeasonalNaive,
    WindowAverage,
    SeasonalWindowAverage,
    Naive
)

models = [
    AutoARIMA(season_length=24),
    AutoTheta(season_length=24),
    AutoETS(season_length=24),
    AutoCES(season_length=24),
    MSTL(season_length=24),
    SeasonalNaive(season_length=24), 
    WindowAverage(window_size=24), 
    SeasonalWindowAverage(window_size=1, season_length=24),
    Naive()
]

sf = StatsForecast( 
    models=models,
    freq='H',
    fallback_model = SeasonalNaive(season_length=24),
    n_jobs=-1,
)

# Placeholder list to hold cross-validation results for each group
cross_validation_results = []


# Iterate through each group in the grouped DataFrame
for unique_id, group_df in memory_usage_dataset_with_corrected_timestamp:
    # Apply cross_validation to each group's DataFrame
    cv_result = sf.cross_validation(
        df=group_df,
        h=730,
        step_size=48,
        n_windows=1
    )
    # Add the group's unique_id to the results for identification
    cv_result['unique_id'] = unique_id
    # Append the result to the list
    cross_validation_results.append(cv_result)

# Optionally, concatenate all individual group results into a single DataFrame
final_cross_validation_df = pd.concat(cross_validation_results)

final_cross_validation_df.to_csv('statsforecast_models_kpi_memory_dataset_rnd_one_month.csv', index=True)




for unique_id, group_df in memory_usage_dataset_with_corrected_timestamp:
    # Initialize and fit the Prophet model
    model = Prophet()
    model.fit(group_df)
    print(unique_id)

    #Setup prophet initial trainig period dynamically based on the number of days in the data frame

    df = df = pd.DataFrame()
    df['date'] = group_df['ds'].dt.date
    df['hour'] = group_df['ds'].dt.hour

    daily_hours = df.groupby('date')['hour'].nunique()
    daily_fraction = daily_hours / 24
    total_days = daily_fraction.sum()
    total_days_rounded_down = math.floor(total_days * 10) / 10

    horizon = 30
    initial = total_days_rounded_down - horizon - 1
    prophet_horizon = str(horizon) + ' days'
    prophet_initial = str(initial) + ' days'
    print(prophet_initial)

    try:
        df_cv = cross_validation(model, horizon=prophet_horizon, initial=prophet_initial)
    except Exception as e:
        print(f"An error occurred during cross-validation foe {unique_id}: {e}")
        continue
    df_cv = df_cv.sort_values(by='ds')
    df_cv['unique_id'] = unique_id
    df_new = df_cv[['ds', 'unique_id', 'yhat']].rename(columns={'yhat': 'prophet'})
    print(df_cv)
    # If 'prophet' already exists in crossvaldation_df, prepare to merge and resolve the column values
    if 'prophet' in final_cross_validation_df.columns:
        # Temporarily rename 'prophet' in crossvaldation_df to avoid automatic suffixing
        final_cross_validation_df.rename(columns={'prophet': 'prophet_temp'}, inplace=True)

        # Merge df1 and df_new
        final_cross_validation_df = pd.merge(final_cross_validation_df, df_new, on=['ds','unique_id'], how='left')

        # Update 'prophet_temp' with 'prophet' from df_new where available
        final_cross_validation_df['prophet'] = final_cross_validation_df['prophet'].combine_first(final_cross_validation_df['prophet_temp'])

        # Drop the temporary and '_new' columns
        final_cross_validation_df.drop(columns=['prophet_temp'], inplace=True)
    else:
        # If 'prophet' does not exist yet, simply merge
        final_cross_validation_df = pd.merge(final_cross_validation_df, df_new, on=['ds','unique_id'], how='left')
    

evaluation_df = evaluate_cross_validation(final_cross_validation_df, mse)

evaluation_df.to_csv(output_file_path, index=True)



