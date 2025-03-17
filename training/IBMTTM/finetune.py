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

if __name__ == "__main__":
    # Set seed for reproducibility
    SEED = 42
    set_seed(SEED)

    #data_base = '/Users/wentianwang/Library/CloudStorage/GoogleDrive-littlenova223@gmail.com/My Drive/quantum_t_data'
    data_base = 'D:/quantum/quantum_t_data/quantum_t_data'

    DATA_ROOT_PATH = data_base + '/type6/Nasdaq_qqq_align_labeled_base_evaluated_normST1.pkl'
    #DATA_ROOT_PATH = data_base + '/QQQ_full_1min_adjsplitdiv.csv'

    # Results dir
    OUT_DIR = "ttm_finetuned_models_10min/"




    # Forecasting parameters
    context_length = 512
    forecast_length = 30

    timestamp_column = 'datetime'
    id_columns = []

    # column_names = [
    #     'datetime',
    #     'close',
    #     'volume',
    #     'sinT',
    #     'cosT',
    #     'event',
    #     'pre_event',
    #     'post_event',
    #     'time_break_flag',
    #     'pre_break',
    #     'post_break',
    #     #'absolute_time'
    # ]
    #pd.read_pickle(DATA_ROOT_PATH)
    time_column_series = pd.read_pickle(
        data_base + '/type4/Nasdaq_qqq_align_labeled_base_evaluated.pkl')['datetime']
    raw_data = pd.read_pickle(DATA_ROOT_PATH)


    raw_data = pd.concat([time_column_series, raw_data], axis=1)

    #raw_data = raw_data.iloc[-2000000:]

    #把Type4的datetime列左侧合并到type6(raw_data)的数据上，同时截取最后2百万行数据

    raw_data[timestamp_column] = pd.to_datetime(raw_data[timestamp_column])

    # Set the 'datetime' column as the index for resampling
    raw_data.set_index('datetime', inplace=True)

    # Resample the data to get hourly intervals using the 'first' value of each hour
    data = raw_data.resample('10T').first()

    # Reset the index to convert the datetime index back to a column
    data.reset_index(inplace=True)
    raw_data.reset_index(inplace=True)

    import matplotlib.pyplot as plt
    import random

    # Select a random starting index for a continuous batch of 100 rows
    start_index = random.randint(0, len(raw_data) - context_length)

    # Get a continuous batch of 100 rows
    continuous_batch = raw_data.iloc[start_index:start_index + context_length]

    # Create a figure with two subplots: one for price data and one for volume
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot price data on the first subplot
    ax1.plot(continuous_batch[timestamp_column], continuous_batch["close_10"], label="Close", marker="o")
    ax1.set_ylabel("Price")
    ax1.set_title("Continuous Batch of Hour-Level Stock Data")
    ax1.legend()
    ax1.grid()

    # Plot volume data on the second subplot
    ax2.plot(continuous_batch[timestamp_column], continuous_batch["volume_10"], color='gray', alpha=0.7)
    ax2.set_xlabel("Datetime")
    ax2.set_ylabel("Volume")
    ax2.set_title("Volume Data")
    ax2.grid()

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Display the plot
    #plt.show()

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

    split_params = {"train": [0, 0.7], "valid": [0.7, 0.8], "test": [0.8, 1.0]}

    tsp = TimeSeriesPreprocessor(
        **column_specifiers,
        context_length=context_length,
        prediction_length=forecast_length,
        scaling=True,
        encode_categorical=False,
        scaler_type="standard",
    )

    train_dataset, valid_dataset, test_dataset = get_datasets(
        tsp,
        data,
        split_params,
    )

    TTM_MODEL_PATH = "ibm-granite/granite-timeseries-ttm-r1"

    zeroshot_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        prediction_channel_indices=tsp.prediction_channel_indices,
        num_input_channels=tsp.num_input_channels,
    )
    zeroshot_model

    temp_dir = tempfile.mkdtemp()
    # zeroshot_trainer
    zeroshot_trainer = Trainer(
        model=zeroshot_model,
        args=TrainingArguments(
            output_dir=temp_dir,
            per_device_eval_batch_size=256,
        ),
    )
    zeroshot_trainer.save_model("original_model")
    #zeroshot_trainer.evaluate(test_dataset)
    # plot
    #plot_predictions(
    #    model=zeroshot_trainer.model,
    #    dset=test_dataset,
    #    plot_dir=os.path.join(OUT_DIR, "close_10min"),
    #    plot_prefix="test_zeroshot",
    #    channel=0,
    #    timestamp_column=timestamp_column,
    #    indices=range(0, len(test_dataset), 20000)
    #)
    #finetune begin
    finetune_forecast_model = get_model(
        TTM_MODEL_PATH,
        context_length=context_length,
        prediction_length=forecast_length,
        num_input_channels=tsp.num_input_channels,
        decoder_mode="mix_channel",  # ch_mix:  set to mix_channel for mixing channels in history
        prediction_channel_indices=tsp.prediction_channel_indices,
    )

    finetune_forecast_model

    #frezze backbone

    print(
        "Number of params before freezing backbone",
        count_parameters(finetune_forecast_model),
    )

    # Freeze the backbone of the model
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False

    # Count params
    print(
        "Number of params after freezing the backbone",
        count_parameters(finetune_forecast_model),
    )

    # Important parameters
    num_epochs = 19  # Ideally, we need more epochs (try offline preferably in a gpu for faster computation)
    batch_size = 128

    learning_rate, finetune_forecast_model = optimal_lr_finder(
        finetune_forecast_model,
        train_dataset,
        batch_size=batch_size,
        enable_prefix_tuning=False,
    )
    print("OPTIMAL SUGGESTED LEARNING RATE =", learning_rate)

    #learning_rate = 0.002
    # finetuning

    finetune_forecast_args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "output"),
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_num_workers=2,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(OUT_DIR, "logs"),  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Number of epochs with no improvement after which to stop
        early_stopping_threshold=0.0,  # Minimum improvement required to consider as improvement
    )
    tracking_callback = TrackingCallback()

    # Optimizer and scheduler
    optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    scheduler = OneCycleLR(
        optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=math.ceil(len(train_dataset) / (batch_size)),
    )

    finetune_forecast_trainer = Trainer(
        model=finetune_forecast_model,
        args=finetune_forecast_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        callbacks=[early_stopping_callback, tracking_callback],
        optimizers=(optimizer, scheduler),
    )

    # Fine tune
    finetune_forecast_trainer.train()
    finetune_forecast_trainer.save_model("finetune_10_model")
    #finetune_forecast_trainer.evaluate(test_dataset)