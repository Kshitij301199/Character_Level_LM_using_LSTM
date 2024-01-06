import os
import random
import time
import string
import torch
import torch.nn as nn
import numpy as np
import unidecode
import matplotlib.pyplot as plt

from utils import char_tensor, random_training_set, time_since, CHUNK_LEN
from evaluation import compute_bpc
from model.model import LSTM


def generate(decoder, prime_str='A', predict_len=100, temperature=0.8):
    hidden, cell = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str
    all_characters = string.printable

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, (hidden, cell) = decoder(prime_input[p].view(1), hidden, cell) 
    inp = prime_input[-1]

    for p in range(predict_len):
        output, (hidden, cell) = decoder(inp.view(1), hidden, cell)

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted

def train(decoder : object, decoder_optimizer : object,
          inp : torch.TensorType, target : torch.TensorType) -> float:
    hidden, cell = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0
    criterion = nn.CrossEntropyLoss()

    for c in range(CHUNK_LEN):
        output, (hidden, cell) = decoder(inp[:, c], hidden, cell)
        loss += criterion(output, target[:, c].view(1))

    loss.backward()
    decoder_optimizer.step()

    return loss.item() / CHUNK_LEN

def tuner(n_epochs : int = 3000,
          print_every : int = 100,
          plot_every  : int = 10,
          hidden_size : int = 128,
          n_layers : int = 2,
          lr :float = 0.005,
          opt : str = "Adam") -> (object,list) :
    
    # import string
    
    all_characters = string.printable
    n_characters = len(all_characters)
    
    decoder = LSTM(
        input_size = n_characters,
        hidden_size = hidden_size,
        num_layers = n_layers,
        output_size = n_characters
                   )
    
    if opt.lower() == 'adam':
        optimizer = torch.optim.Adam(params = decoder.parameters(), lr = lr)
    elif opt.lower() == 'adamw':
        optimizer = torch.optim.AdamW(params = decoder.parameters(), lr = lr)
    elif opt.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(params = decoder.parameters(), lr = lr)
        
    print("Parameters of LSTM : \n")
    print(decoder)
    print("Parameters of optimizer : \n")
    print(optimizer)
    print(f"Number of Epochs : {n_epochs}")
    print(f"Learning Rate : {lr}")
    
    start = time.time()
    all_losses = []
    loss_avg = 0
   
    print(" -------------------------- STARTING TRAINING -------------------------- ")
    
    for epoch in range(1, n_epochs+1):
            loss = train(decoder, optimizer, *random_training_set())
            loss_avg += loss

            if epoch % print_every == 0:
                print(f'[{time_since(start)} ({epoch} {np.round(epoch/n_epochs * 100,2)}%) {np.round(loss,4)}]')
                #print(generate(decoder, start_string , prediction_length), '\n')

            if epoch % plot_every == 0:
                all_losses.append(loss_avg / plot_every)
                loss_avg = 0
                
    return decoder, all_losses

def plot_loss(no_of_epochs:int = 1000, plot_every : int = 10, lr_list : list = [0.005]) -> None:
    plt.style.use('seaborn')
    plt.figure(figsize=(10,10))
    plt.xticks(fontweight='bold', size=12)
    plt.yticks(fontweight='bold', size=12)
    plt.tick_params(axis='both', direction='in')
    all_colours = ["red","blue","green","yellow","purple","black","violet","grey","cyan","magenta"]
    colours = random.sample(all_colours,len(lr_list))
    # no_of_epochs = 2000
    # plot_every = 5
    x = np.arange(0,no_of_epochs,plot_every)
    for index,lr in enumerate(lr_list):
        print(f" -------------------------- LEARNING RATE OPTION {index+1}/{len(lr_list)} -------------------------- ")
        _ , loss = tuner(n_epochs = no_of_epochs, plot_every = plot_every, lr = lr)
        plt.plot(x,loss,colours[index],label = f"{lr}")
        
    plt.xlabel("Number of iterations",fontsize=15,fontweight='bold')
    plt.ylabel("CrossEntropy Loss",fontsize=15,fontweight='bold')
    plt.legend(prop = {"size":12,"weight":"bold"},loc='best',labelcolor='linecolor',
            frameon = True, fancybox = True,framealpha=1,
            facecolor = "white", edgecolor="black")
    
    plt.tight_layout()
    print("Saving Image ...")
    if os.path.exists("./plot_loss.png"):
        plt.savefig("./plot_loss_2.png",dpi = 300)
    else:
        plt.savefig("./plot_loss.png",dpi = 300)
    print("Image Saved !")

def diff_temp(temp_list):
    # YOUR CODE HERE
    #         1) Using `tuner()` function, try to generate strings by using different temperature
    #         from `temp_list`.
    #         2) In order to do this, create chunks from the test set (with 200 characters length)
    #         and take first 10 characters of a randomly chosen chunk as a priming string.
    #         3) What happen with the output when you increase or decrease the temperature?
    ################################ STUDENT SOLUTION ################################
    model, _ = tuner(n_epochs = 2000, opt = "RMSprop")
    test_path = './data/dickens_test.txt'
    input_string = unidecode.unidecode(open(test_path, 'r').read())
    predict_len = 200
    file_name = "./diff_temp.txt"
    with open(file_name, 'w', newline='') as file:
        for temperature in temp_list:
            # random_chunk
            start_index = random.randint(0, len(input_string) - CHUNK_LEN - 1)
            end_index = start_index + CHUNK_LEN + 1
            chunk = input_string[start_index: end_index]
            predicted_text = generate(model, prime_str=chunk[:10], predict_len=predict_len, temperature=temperature)    
            print(f"TEMPERATURE : {temperature}")
            file.write(f"TEMPERATURE : {temperature}\n")
            print(f"PREDICTION : {predicted_text}")
            file.write(f"PREDICTION : {predicted_text}\n")
            print(f"ORIGINAL TEXT : {chunk}")
            print("---------------------------------------------------------------\n")
    ##################################################################################

def custom_train(hyperparam_list):
    """
    Train model with X different set of hyperparameters, where X is 
    len(hyperparam_list).

    Args:
        hyperparam_list: list of dict of hyperparameter settings

    Returns:
        bpc_dict: dict of bpc score for each set of hyperparameters.
    """
    i = 0
    test_path = './data/dickens_test.txt'
    bpc_list = []
    loss_list = []
    test_string = unidecode.unidecode(open(test_path, 'r').read())
    for hyperparams in hyperparam_list:
        i += 1
        print(f" -------------------------- HYPERPARAMETER OPTION {i}/{len(hyperparam_list)} -------------------------- ")
        decoder, loss = tuner(**hyperparams)
        bpc_list.append(compute_bpc(decoder, test_string))
        loss_list.append(loss[-1])
        
    return bpc_list, loss_list