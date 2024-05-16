# Sequence-to-Sequence Model Configuration:
This document outlines the configuration parameters for training a sequence-to-sequence model for various tasks such as machine translation, text summarization, and more.

## Dependencies

Make sure you have the following libraries installed:

- torch >= 1.0
- pandas
- matplotlib >= 3.0
- tqdm >= 4.0
- wandb >= 0.10
- argparse

You can install these dependencies using pip:

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









                        
# Example Usage:

python train.py --num_filters 128 --activation gelu --filter_organization double --batch_normalization Yes --dropout_value 0.3 --learning_rate 0.0001 --num_epochs 10 --dense_neurons 1024 --batch_size 32 --data_augmentation Yes --wandb_project DL_Assignment_2_CS23M046 --wandb_entity cs23m046 --train_dataset /path/to/training_dataset  --freeze_option all_except_last --freeze_index 2




## Replace /path/to/training_dataset  with the actual paths to your dataset.


