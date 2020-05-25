import warnings
warnings.filterwarnings("ignore")
from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd
from test import pre
import re
from inputHandler import *
df = pd.read_csv('train_data.csv')

def fn():
	sentence1=[]
	sentence2=[]
	is_similar=[]
	f = open("train_snli.txt", "r")

	for x in f:
		y=re.split('\\t+', x)
		sentence1.append(y[0].rstrip("\n"))
		sentence2.append(y[1].rstrip("\n"))
		is_similar.append(y[2].rstrip("\n"))
	return sentence1,sentence2,is_similar
#sentences1,sentences2,is_similar=fn()
#print(sentences1)

sentences1 = list(df['sentences1'])
sentences2 = list(df['sentences2'])
is_similar = list(df['is_similar'])

del df

sentences1=pre.preprocessing(sentences1)
sentences2=pre.preprocessing(sentences2)
tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])
#embedding_matrix["hello"]
embedding_meta_data = {
	'tokenizer': tokenizer,
	'embedding_matrix': embedding_matrix
}
def train_model(sentences1, sentences2 , is_similar):

	sentences_pair = [(x1, x2) for x1, x2 in zip(sentences1, sentences2)]
	class Configuration(object):
	    """"""

	CONFIG = Configuration()

	CONFIG.embedding_dim = siamese_config['EMBEDDING_DIM']#100d
	CONFIG.max_sequence_length = siamese_config['MAX_SEQUENCE_LENGTH'] #20
	CONFIG.number_lstm_units = siamese_config['NUMBER_LSTM'] #100
	CONFIG.rate_drop_lstm = siamese_config['RATE_DROP_LSTM'] #0.2
	CONFIG.number_dense_units = siamese_config['NUMBER_DENSE_UNITS'] #100
	CONFIG.activation_function = siamese_config['ACTIVATION_FUNCTION'] #relu
	CONFIG.rate_drop_dense = siamese_config['RATE_DROP_DENSE'] #0.2
	CONFIG.validation_split_ratio = siamese_config['VALIDATION_SPLIT'] #0.15

	siamese = SiameseBiLSTM(CONFIG.embedding_dim , CONFIG.max_sequence_length, CONFIG.number_lstm_units , CONFIG.number_dense_units, CONFIG.rate_drop_lstm, CONFIG.rate_drop_dense, CONFIG.activation_function, CONFIG.validation_split_ratio)
	best_model_path = siamese.train_model(sentences_pair, is_similar, embedding_meta_data, model_save_directory='./')
	return best_model_path


from operator import itemgetter
from keras.models import load_model

#to train a model
#bst_path=train_model(sentences1,sentences2,is_similar)
#print(bst_path)
#1589893283
bst_path="./stored_model/1589893283/lstm_new.h5"

model = load_model(bst_path)
#model.summary()
del sentences1
del sentences2
import re
import csv
f = open("test.txt", "r")
a=0
lis=[]
li=[]



for x in f:
    y=re.split('\\t+', x)
    if(a==0):
        a=a+1
        continue
    lis.append((pre.preprocessing_single_sentence(y[3].rstrip("\n")), pre.preprocessing_single_sentence(y[4].rstrip("\n")),y[0]))
    a=a+1

test_sentence_pairs = [(x, y) for (x, y, w) in (lis)]

test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

preds = list(model.predict([test_data_x1, test_data_x2, leaks_test]).ravel())
y_pred =preds
z1=[]
for i in y_pred:
	if float(i) < 0.45:
		z1.append(0)
	else:
		z1.append(1)
results = [( x, y, z, w, v) for (x, y, w), z ,v in zip(lis, preds,z1)]
y_actu = [ int(w) for (x, y, w), z in zip(lis, preds)]
res=[]
for temp in results:
	if int(temp[3]) != temp[4]:
		res.append((temp))
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp =confusion_matrix(y_actu, z1).ravel()
print(tn, fp, fn, tp)
print('accuracy is =')
print((tn+tp)/(tn+tp+fn+fp))
res.sort(key = lambda x: x[2])
res.reverse()
with open("difference.txt", "w") as outfile:
   outfile.write("\n".join(str(i) for i in res))

with open("output.txt", "w") as outfile:
   outfile.write("\n".join(str(i) for i in results))

#print(results)
