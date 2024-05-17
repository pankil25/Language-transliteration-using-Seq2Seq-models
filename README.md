# Sequence-to-Sequence Model Configuration:
This document outlines the configuration parameters for training a sequence-to-sequence model for various tasks such as machine translation, text summarization, and more.


## Dependencies

Make sure you have the following libraries installed:

- torch >= 1.0
- pandas
- scikit-learn (needed for heatmap plot using wandb)
- tqdm >= 4.0
- wandb = 0.14.0 (If you want to plot attention heatmap using wandb plots as in latest version heatmap functionality is depriciated)
- argparse

You can install these dependencies using pip:

!pip install wandb==0.14.0 scikit-learn

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
### Number of Epochs : [5, 10, 15 ,20]
Description: Specifies the number of training epochs.<br>
### Dropout : [0.2, 0.3]
Description: Specifies the dropout probability for regularization.<br>
### Teacher Forcing Ratio : [0.5]
Description: Specifies the probability of using teacher forcing during training.<br>
### Attention : [True,False]
Description: Specifies whether attention mechanism is used in the model.<br>
### Mode: ["Normal","Test"]
Description: Specifies whether You want to run Train and Validation Dataset or want to run on Train and Test Dataset.<br>
<br>
<br>

## For CS23M046_DL_assignment_3.ipynb :<br>
- After satisfying above mentioned dependencies you are now need to following steps for ipynb file to run.
  
- in main_1() function which is present in last cell of ipynb file you have to add dataset folder aksharantar_sampled path and **make sure this path is unzipped** and not the zip file path.<br>

- in main_1() function which is present in last cell of ipynb file you can choose language of your choice in Folder_name parameter by passing folder name <br>

- in main_1() function you can choose mode parameter value to be 'Normal' if you want to use Train and Validation dataset only and if you want to choose Train and Test dataset then assign 'Test' to mode parameter.
  
- After implementing above steps you can run ipynb file sequencially from top cell number 1 to last cell and you can see the results according to sweeep config.
  
- Commentes are also applied in each cell code to understand the code flow.<br>
<br>

## TFor train.py<br>

- After satisfying above dependencies you can choose values of hyper parameters and and you can run code by passing as command line .<br>

- Description of each hyper parameters also mentioned above so based on that you can modify following command to run train.py script.<br>


                        
# Example Usage:<br>

python train.py --input_embedding_size 256 --encoder_num_layers 3 --decoder_num_layers 3 --hidden_size 1024 --cell_type LSTM --bidirectional True --batch_size 128 --learning_rate 0.0001 --num_epochs 15 --dropout 0.2 --teacher_forcing_ratio 0.5 --attention True --mode Normal --wandb_project DL_Assignment_2_CS23M046 --wandb_entity cs23m046 --Folder_path '/kaggle/input/aksharantar-sampled/aksharantar_sampled' --Folder_name 'guj'





## Replace '/kaggle/input/aksharantar-sampled/aksharantar_sampled'  with the actual path to dataset in Folder_path argument.If you want to use different language then you can replace 'guj' with your language name in Folder_name argument


