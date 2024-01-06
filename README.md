# Character-Level Language Modeling using LSTM

This repository contains code for building a Character-Level Language Model using Long Short-Term Memory (LSTM) neural networks. The model is trained on a given text corpus to generate coherent and contextually relevant sequences of characters.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Repository Structure](#repository-structure)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Overview
The primary goal of this project is to demonstrate the implementation of an LSTM-based language model for character-level sequence generation. The model is trained on a provided text dataset, and the trained model can be used to generate new sequences of characters.

## Dependencies
- Python 3
- PyTorch
- Other dependencies listed in `requirements.txt`

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

## Repository Structure
- `models/`: Contains the LSTM model implementation.
- `utils/`: Includes utility functions for data preprocessing and sequence generation.
- `data/`: Directory to store training and testing datasets.
- `train.py`: Script for training the LSTM model.
- `generate.py`: Script for generating sequences using the trained model.
- `README.md`: Project documentation.

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/Kshitij301199/Character_Level_LM_using_LSTM.git
   cd Character_Level_LM_using_LSTM
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   ```bash
   python train.py --input_path data/training_text.txt --output_model_path models/lstm_model.pth
   ```

4. Generate sequences:
   ```bash
   python generate.py --model_path models/lstm_model.pth --prime_str "Hello, " --predict_len 200
   ```

## Training
To train the LSTM model on a custom text dataset, use the `train.py` script. Specify the path to your training text file using the `--input_path` argument.

```bash
python train.py --input_path path/to/your/training_text.txt --output_model_path path/to/save/model.pth
```

## Evaluation
Evaluation of the model can be done by generating sequences using the `generate.py` script. Specify the path to the trained model using the `--model_path` argument and provide optional parameters like `--prime_str` and `--predict_len`.

```bash
python generate.py --model_path path/to/your/model.pth --prime_str "Example starting text" --predict_len 300
```

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests for any improvements or additional features.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

This README provides an overview of the repository, information on dependencies, details about the repository structure, usage instructions, guidelines for training and evaluation, an invitation for contributions, and information about the project's license. Customize the content as needed for your specific repository.