# ME19B031_CS6910_ASSIGNMENT_3

This repository contains ME19B031_VANILLA_FINAL.ipynb file.
I have used Tokenize,Encoder,Decoder,Seq2Seq,Train_Model funtions to train the model.
Tokenize Function will take input language,output language as parameters while initializing and this function can be used to convert word to tensor and tensor to word and also this function can convert input,target pair together to the tensore. This can be achieved using Tokenize.tensorsFromPair(pair) and Tokenize.PairFromtensors(pair_of_Tensor).
Q1:
### Encoder and Decoder classes
Encoder Function will encode the given token and returns output,hidden,cell vectors.
Decoder Function will take the encoded hidden vectors, target tokens(while training) and returns returns the predicted token.
Encoder and Decoder Functions have the following hyperparameters:
Encoder : input_size,embedding_size,hidden_size,num_layers, dropouts,cell_type,bidirectional
Decoder : input_size,embedding_size,hidden_size,output_size,num_layers,dropouts,cell_type,bidirectional
These functions gives us the flexibility to choose the hyperparameters such as:
cell type : "RNN","GRU","LSTM"
bidirectional : "True", "False"
dropout : [number between 0 to 1]
and remaining parameters can take any integer 
### Seq2Seq Class
This is our Model which will help in training/Predicting the sequence with the help of encoder and decoder functions. This Seq2Seq function gives the input sequence to the encoder, takes the hidden vector and feeds that to the decoder function and it will iterate over the sequence length to predict probability disctribution over vocabulary at each position.
This Class has both forward and predict functions.
### Train_Model Function:
Train_Model will take the parameter (mentioned below are the default parameters):
num_epochs = 10,
learning_rate = 0.001,
input_size_encoder = 28,
input_size_decoder = 130,
output_size = 130,
encoder_embeddings_size = 256,
decoder_embeddings_size = 256,
hidden_size = 512,
num_enc_layers = 3,
num_dec_layers = 3,
enc_dropout = 0.2,
dec_dropout = 0.2,
cell_type = "LSTM",
bidirectional = True
and takes train and validation data sets with batches
This function will train the Seq2Seq model while printing Train_Loss,Validatioon_Loss,Validation_Accuracy for each epoch and returns the best_model obtained during the the training.
### Validation
Now we can use that returned model to predict (model.predict) the outputs of valiation data and check the accuracy.
Sample code for this is shown in ME19B031_TRAIN_VALID.ipynb
Similarly we do the same thing with the attention model as shown in ME19B031_ASSIGNMENT_3_ATTENTION.ipynb

