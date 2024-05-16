# cs6910_assignment3

## Sequence-to-Sequence Model Configuration:
This document outlines the configuration parameters for training a sequence-to-sequence model for various tasks such as machine translation, text summarization, and more.

## Hyperparameters:
### Input Embedding Size : [16, 32, 64, 256]
Description: Specifies the size of the input embedding vector.<br>

### Encoder Number of Layers : [1, 2, 3]
Description: Specifies the number of layers in the encoder network.<br>
### Decoder Number of Layers : [1, 2, 3]
Description: Specifies the number of layers in the decoder network.<br>
### Hidden Size : [128, 256, 512, 1024]
Description: Specifies the size of the hidden state in the RNN cells.<br>
### Cell Type : ["RNN", "GRU", "LSTM"]
Description: Specifies the type of recurrent cell to be used in the encoder and decoder.<br>
### Bidirectional : [True, False]
Description: Specifies whether the encoder is bidirectional or not.<br>
### Batch Size : [32, 64, 128]
Description: Specifies the number of training examples in each batch.<br>
### Learning Rate : [0.001, 0.0001]
Description: Specifies the learning rate for training the model.<br>
### Number of Epochs : [5, 10, 15]
Description: Specifies the number of training epochs.<br>
### Dropout : [0.2, 0.3]
Description: Specifies the dropout probability for regularization.<br>
### Teacher Forcing Ratio : [0.5]
Description: Specifies the probability of using teacher forcing during training.<br>
### Attention : [True,False]
Description: Specifies whether attention mechanism is used in the model.<br>
<br>

## Usage:
