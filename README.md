# gpt2-game-agent

Welcome to the **gpt2-game-agent** repository! This repository contains code to generate ConnectFour game samples, train a GPT-2 model from Huggingface on this data, and evaluate the trained model. It serves as an example of how to train GPT-2 on custom data.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Generate Training Samples](#generate-training-samples)
  - [Train the Model](#train-the-model)
  - [Evaluate the Model](#evaluate-the-model)
- [License](#license)

## Installation

First, clone the repository and navigate to the project directory:
```bash
git clone https://github.com/p4vv37/experiments.git
cd experiments
```

Create a virtual environment and install the required dependencies:

```
conda env create -f env.yml
conda activate gpt2-game-agent
```

## Usage
### Generate Training Samples
To generate ConnectFour game samples, run the following script:
```
python generate_training_samples.py
```
This script will generate a dataset of ConnectFour games, which will be used for training the GPT-2 model.

### Train the Model
To train the GPT-2 model on the generated data, use the `train_network.py` script:

```
python train_network.py
```

This script leverages Huggingface's GPT-2 model and fine-tunes it on the ConnectFour game data.

### Evaluate the Model
After training, evaluate the model using the `evaluate_network.py` script:

This script evaluates the performance of the trained model and provides metrics to assess its accuracy and effectiveness.
Result:
```
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
 99%|█████████▉| 99/100 [00:22<00:00,  4.43it/s]
GPT2 win-rate: 80.0%
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.