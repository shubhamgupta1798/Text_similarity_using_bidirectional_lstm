# Text_similarity_using_bidirectional_lstm
Comparing two text inputs and predicting how similar they are. Used a variety of approaches for encoding the words such as wordtovec and using pertained glove embeddings
We have used Bidrectional LSTM for training our model and have got an accuracy of 87% by fine tuning the model parameters. 
Have used a large dataset of about 10,000 sentences to train the model with training and validation set being divided randomly in ratio 1:7.
We have also preprocessed the data by removing the stop words which occur in the sentence often but don't provide crutial information about the meaning of the sentence
