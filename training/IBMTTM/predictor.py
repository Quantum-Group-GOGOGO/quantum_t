import math
import os
import tempfile

import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from tsfm_public import (
    TimeSeriesPreprocessor,
    TrackingCallback,
    count_parameters,
    get_datasets,
)
from tsfm_public.toolkit.get_model import get_model
from tsfm_public.toolkit.lr_finder import optimal_lr_finder
from tsfm_public.toolkit.visualization import plot_predictions

data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
#data_base = 'D:/quantum/quantum_t_data/quantum_t_data'


# Results dir
# Forecasting parameters
context_length = 512
forecast_length = 96
timestamp_column = 'datetime'
id_columns = []
column_specifiers = {
        "timestamp_column": timestamp_column,
        "id_columns": id_columns,
        "target_columns": ["close_10", 'volume_10'],
        "conditional_columns": [
        'sinT',
        'cosT',
        #'event',
        'pre_event',
        'post_event',
        #'time_break_flag',
        'pre_break',
        'post_break',
        'week_fraction_sin',
        'week_fraction_cos',
        'absolute_time'
        ],
    }
tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )
model_save_path = "finetune_model"
finetuned_model = get_model(
        model_save_path,
        context_length=context_length,
        prediction_length=forecast_length,
        prediction_channel_indices=tsp.prediction_channel_indices,
        num_input_channels=tsp.num_input_channels,
)

DATA_ROOT_PATH = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'

#time_column_series = pd.read_pickle(data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated.pkl')['datetime']
#raw_data = pd.read_pickle(DATA_ROOT_PATH)


#raw_data = pd.concat([time_column_series, raw_data], axis=1)

#raw_data[timestamp_column] = pd.to_datetime(raw_data[timestamp_column])

# Set the 'datetime' column as the index for resampling
#raw_data.set_index('datetime', inplace=True)

# Resample the data to get hourly intervals using the 'first' value of each hour
#data = raw_data.resample('10min').first()

# Reset the index to convert the datetime index back to a column
#data.reset_index(inplace=True)
#raw_data.reset_index(inplace=True)
#data.to_pickle(data_base + '/type6/10minprocessed.pkl')
data=pd.read_pickle(data_base + '/type6/10minprocessed.pkl')
split_params = {"train": [0, 0.7], "valid": [0.7, 0.8], "test": [0.8, 1.0]}

train_dataset, valid_dataset, test_dataset = get_datasets(
    tsp,
    raw_data,
    split_params,
)