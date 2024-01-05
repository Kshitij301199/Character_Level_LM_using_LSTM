import math
import torch

from utils import char_tensor, CHUNK_LEN


def compute_bpc(model : object, string : str) -> float:
    """
    Given a model and a string of characters, compute bits per character
    (BPC) using that model.

    Args:
        model: RNN-based model (RNN, LSTM, GRU, etc.)
        string: string of characters

    Returns:
        BPC for that set of string.
    """
    criterion = torch.nn.CrossEntropyLoss()
    avg_bpc = 0
    bpc_losses = []
    num_iters = 0
    for i in range(0,len(string)-1,CHUNK_LEN):
        hidden, cell = model.init_hidden()
        chunk = string[i:i+CHUNK_LEN+1]
        inp : torch.TensorType = char_tensor(chunk[:-1]).unsqueeze(0) #adds a dimension in the 0th index
        # print(inp.size()) = 200
        target : torch.TensorType = char_tensor(chunk[1:]).unsqueeze(0)
        # print(target.size()) = 200
        if len(target.squeeze(0)) != 200:
            continue
        
        loss = 0
        for c in range(CHUNK_LEN):
            with torch.no_grad():
                output, (hidden,cell) = model(inp[:, c],hidden,cell)
            loss += criterion(output, target[:, c].view(1))
            
        loss = loss.item()/CHUNK_LEN #gets element in the tensor
        
        # Bits per character = CrossEntropyLoss / log2
        bpc = loss / math.log(2)
        avg_bpc += bpc
        bpc_losses.append(bpc)
        num_iters += 1
        if num_iters % 1500 == 0:
            print(f"Number of iterations run for BPC calc : {num_iters}")
        print(f"Total number of iterations : {num_iters}")
    return avg_bpc / num_iters

