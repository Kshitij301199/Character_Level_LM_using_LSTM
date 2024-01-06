"""
Train and evaluate an LSTM language model with various options.

Options include:
- `--default_train`: Train LSTM with default hyperparameters.
- `--custom_train`: Train LSTM while tuning hyperparameters.
- `--plot_loss`: Plot losses chart with different learning rates.
- `--diff_temp`: Generate strings using different temperature values.

This script serves as a tool for training, evaluating, and experimenting with LSTM language models.
The main function, `main()`, acts as the entry point and parses command-line arguments to execute
different tasks, such as default training, custom training with hyperparameter tuning, loss
plotting, and string generation with different temperatures.

Command-line Arguments:
    --default_train: Train LSTM with default hyperparameters.
    --custom_train: Train LSTM while tuning hyperparameters.
    --plot_loss: Plot losses chart with different learning rates.
    --diff_temp: Generate strings using different temperature values.

Returns:
    None
"""
import argparse
import csv
import string
import time
import datetime
import torch
import numpy as np
# import torch.nn as nn
# import unidecode

from utils import random_training_set, time_since
from language_model import plot_loss, diff_temp, custom_train, train, generate
from model.model import LSTM


def main():
    """
    Main function for training and evaluating an LSTM model with various options.

    Options include:
    - `--default_train`: Train LSTM with default hyperparameters.
    - `--custom_train`: Train LSTM while tuning hyperparameters.
    - `--plot_loss`: Plot losses chart with different learning rates.
    - `--diff_temp`: Generate strings using different temperature values.

    Returns:
        None

    This function serves as the entry point for the script. It parses command-line arguments,
    performs the specified tasks, and prints the results. The training and evaluation options
    include default training, custom training with hyperparameter tuning, plotting loss curves,
    and generating strings with different temperatures.
    """
    parser = argparse.ArgumentParser(
        description='Train LSTM'
    )

    parser.add_argument(
        '--default_train', dest='default_train',
        help='Train LSTM with default hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--custom_train', dest='custom_train',
        help='Train LSTM while tuning hyperparameter',
        action='store_true'
    )

    parser.add_argument(
        '--plot_loss', dest='plot_loss',
        help='Plot losses chart with different learning rates',
        action='store_true'
    )

    parser.add_argument(
        '--diff_temp', dest='diff_temp',
        help='Generate strings by using different temperature',
        action='store_true'
    )

    args = parser.parse_args()

    all_characters = string.printable
    n_characters = len(all_characters)

    if args.default_train:
        print(f"Start Time : {datetime.datetime.now()}")
        n_epochs = 2000
        print_every = 100
        plot_every = 10
        hidden_size = 128
        n_layers = 2
        lr = 0.005
        
        decoder = LSTM(
            input_size = n_characters, 
            hidden_size = hidden_size,
            num_layers = n_layers, 
            output_size = n_characters)
        
        decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
        print("Parameters of LSTM : \n")
        print(decoder)
        print("Parameters of optimizer : \n")
        print(f"Learning Rate : {lr}")
        print(decoder_optimizer)
        start = time.time()
        all_losses = []
        loss_avg = 0

        print(" -------------------------- STARTING TRAINING -------------------------- ")
        
        for epoch in range(1, n_epochs+1):
            loss = train(decoder, decoder_optimizer, *random_training_set())
            loss_avg += loss

            if epoch % print_every == 0:
                print(f'[{time_since(start)} ({epoch} {np.round(epoch/n_epochs * 100,4)} %) {np.round(loss,4)}]')
                print(generate(decoder, 'A', 100), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0

    if args.custom_train:
        print(f"Start Time : {datetime.datetime.now()}")
        hyperparam_list = [
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.005, "opt" : "Adam"},
            {"n_epochs" : 3000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.005, "opt" : "Adam"}, # tune n_epochs
            {"n_epochs" : 2000, "hidden_size" : 64, "n_layers" : 2, "lr" : 0.005, "opt" : "Adam"},
            {"n_epochs" : 2000, "hidden_size" : 256, "n_layers" : 2, "lr" : 0.005, "opt" : "Adam"}, # tune hidden_size
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 1, "lr" : 0.005, "opt" : "Adam"},
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 3, "lr" : 0.005, "opt" : "Adam"}, # tune n_layers
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.1, "opt" : "Adam"},
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.01, "opt" : "Adam"},
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.0001, "opt" : "Adam"}, # tune lr
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.005, "opt" : "AdamW"},
            {"n_epochs" : 2000, "hidden_size" : 128, "n_layers" : 2, "lr" : 0.005, "opt" : "RMSprop"} # tune opt
            # {"n_epochs" : 2000, "hidden_size" : 256, "n_layers" : 2, "lr" : 0.005, "opt" : "RMSprop"}
            ]
        
        bpc_list, loss_list = custom_train(hyperparam_list)
        file_name = "custom_train_output.txt"
        
        with open(file_name, 'w', newline='') as file:
            headers = list(hyperparam_list[0].keys())
            headers.append("BPC")
            headers.append("Loss")
            writer = csv.DictWriter(file, fieldnames = headers, delimiter="\t")
            writer.writeheader()
        
            for hyperparams, bpc, loss in zip(hyperparam_list, bpc_list, loss_list):
                print(f"CONFIGURATION : {hyperparams}\n BPC = {bpc} \t LOSS = {loss}")
                writer.writerow(hyperparams | {"BPC" : np.round(bpc,5), "Loss" : np.round(loss,4)})
            
            
    if args.plot_loss:
        print(f"Start Time : {datetime.datetime.now()}")
        lr_list = [0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
        lr_list = [0.001,0.0025,0.005,0.0075,0.01]
        plot_loss(no_of_epochs = 2000,plot_every = 5,lr_list = lr_list)

    if args.diff_temp:
        print(f"Start Time : {datetime.datetime.now}")
        # YOUR CODE HERE
        #         1) Fill in `temp_list` with temperatures that you want to try.
        ########################### STUDENT SOLUTION ###########################
        temp_list = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        ########################################################################
        diff_temp(temp_list)


if __name__ == "__main__":
    main()
