# ME19B031_CS6910_ASSIGNMENT_3

This repository contains ME19B031_VANILLA_FINAL.ipynb file.
I have used Tokenize,Encoder,Decoder,Seq2Seq,Train_Model funtions to train the model.
Tokenize Function will take input language,output language as parameters while initializing and this function can be used to convert word to tensor and tensor to word and also this function can convert input,target pair together to the tensore. This can be achieved using Tokenize.tensorsFromPair(pair) and Tokenize.PairFromtensors(pair_of_Tensor).
## Q1:
### Encoder class
Encoder Function will encode the given token and returns output,hidden,cell vectors. <br>
Encoder Functions have the following hyperparameters:<br>
Encoder : input_size,embedding_size,hidden_size,num_layers, dropouts,cell_type,bidirectional<br>
This function gives us the flexibility to choose the hyperparameters such as:<br>
cell type : "RNN","GRU","LSTM"<br>
bidirectional : "True", "False"<br>
dropout : [number between 0 to 1]<br>
and remaining parameters can take any integer <br>
### Decoder class
Decoder Function will take the encoded hidden vectors, previous output tokens and returns the predicted probability distribution over vocabulary length.<br>
Decoder Functions have the following hyperparameters:<br>
Decoder : input_size,embedding_size,hidden_size,output_size,num_layers,dropouts,cell_type,bidirectional<br>
This function gives us the flexibility to choose the hyperparameters such as:<br>
cell type : "RNN","GRU","LSTM"<br>
bidirectional : "True", "False"<br>
dropout : [number between 0 to 1]<br>
and remaining parameters can take any integer <br>
### Seq2Seq Class<br>
This is our Model which will help in training/Predicting the sequence with the help of encoder and decoder functions. This Seq2Seq function gives the input sequence to the encoder, takes the hidden vector and feeds that to the decoder function and it will iterate over the sequence length to predict probability disctribution over vocabulary at each position.
This Class has both forward and predict functions.
### Train_Model Function:<br>
Train_Model will take the parameter (mentioned below are the default parameters):<br>
num_epochs = 10,<br>
learning_rate = 0.001,<br>
input_size_encoder = 28,<br>
input_size_decoder = 130,<br>
output_size = 130,<br>
encoder_embeddings_size = 256,<br>
decoder_embeddings_size = 256,<br>
hidden_size = 512,<br>
num_enc_layers = 3,<br>
num_dec_layers = 3,<br>
enc_dropout = 0.2,<br>
dec_dropout = 0.2,<br>
cell_type = "LSTM",<br>
bidirectional = True<br>
and takes train and validation data sets with batches<br>
This function will train the Seq2Seq model while printing Train_Loss,Validatioon_Loss,Validation_Accuracy for each epoch and returns the best_model obtained during the the training.
### Validation
Now we can use that returned model to predict (model.predict) the outputs of valiation data and check the accuracy.<br>
Sample code for this is shown in ME19B031_TRAIN_VALID.ipynb<br>
Similarly we do the same thing with the attention model as shown in ME19B031_ASSIGNMENT_3_ATTENTION.ipynb<br>

