# cs6910_assignment3

## Sequence-to-Sequence Model Configuration:
This document outlines the configuration parameters for training a sequence-to-sequence model for various tasks such as machine translation, text summarization, and more.

## Hyperparameters:
### Input Embedding Size
Values: [16, 32, 64, 256]<br>
Description: Specifies the size of the input embedding vector.
### Encoder Number of Layers
Values: [1, 2, 3]
Description: Specifies the number of layers in the encoder network.
### Decoder Number of Layers
Values: [1, 2, 3]
Description: Specifies the number of layers in the decoder network.
### Hidden Size
Values: [128, 256, 512, 1024]
Description: Specifies the size of the hidden state in the RNN cells.
### Cell Type
Values: ["RNN", "GRU", "LSTM"]
Description: Specifies the type of recurrent cell to be used in the encoder and decoder.
### Bidirectional
Values: [True, False]
Description: Specifies whether the encoder is bidirectional or not.
### Batch Size
Values: [32, 64, 128]
Description: Specifies the number of training examples in each batch.
### Learning Rate
Values: [0.001, 0.0001]
Description: Specifies the learning rate for training the model.
### Number of Epochs
Values: [5, 10, 15]
Description: Specifies the number of training epochs.
### Dropout
Values: [0.2, 0.3]
Description: Specifies the dropout probability for regularization.
### Teacher Forcing Ratio
Values: [0.5]
Description: Specifies the probability of using teacher forcing during training.
### Attention
Values: [True,False]
Description: Specifies whether attention mechanism is used in the model.
## Usage:
Adjust the hyperparameters according to your specific task requirements.
Experiment with different combinations of hyperparameters to find the optimal configuration for your model.
Refer to the documentation of the sequence-to-sequence model implementation for more details on how to use these parameters.
