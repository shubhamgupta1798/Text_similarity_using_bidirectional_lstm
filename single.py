from model import SiameseBiLSTM
from inputHandler import word_embed_meta_data, create_test_data
from config import siamese_config
import pandas as pd
from test import pre
from inputHandler import *
import warnings
import re
warnings.filterwarnings("ignore", category=DeprecationWarning)
def fn():
	df = pd.read_csv('train_data.csv')
	sentences1 = list(df['sentences1'])
	sentences2 = list(df['sentences2'])
	is_similar = list(df['is_similar'])
	sentences1=pre.preprocessing(sentences1)
	sentences2=pre.preprocessing(sentences2)
	tokenizer, embedding_matrix = word_embed_meta_data(sentences1 + sentences2,  siamese_config['EMBEDDING_DIM'])
	embedding_meta_data = {
		'tokenizer': tokenizer,
		'embedding_matrix': embedding_matrix
	}
	return tokenizer,embedding_matrix,embedding_meta_data
tokenizer,embedding_matrix,embedding_meta_data=fn()

def single_pred(x,y):
	"""

	"""
	#print(sentences1,sentences2)

	from operator import itemgetter
	from keras.models import load_model
	bst_path="./stored_model/1589893283/lstm_new.h5"
	model = load_model(bst_path)

	test_sentence_pairs=[(pre.preprocessing_single_sentence(x),pre.preprocessing_single_sentence(y))]
	test_data_x1, test_data_x2, leaks_test = create_test_data(tokenizer,test_sentence_pairs,  siamese_config['MAX_SEQUENCE_LENGTH'])

	preds = model.predict([test_data_x1, test_data_x2, leaks_test]).ravel()
	print(preds[0])
	l=str(preds[0]*100)
	return l


from tkinter import *
window = Tk()
window.title("Welcome")
window.geometry('500x500')
lbl = Label(window, text="Sentence 1")
lbl.grid(column=0, row=0)
txt = Entry(window,width=50)
txt.grid(column=1, row=0)
lbl2 = Label(window, text="Sentence 2")
lbl2.grid(column=0, row=1)
txt2 = Entry(window,width=50)
txt2.grid(column=1, row=1)
lbl3=Label(window, text="Result")
lbl3.grid(column=1, row=3)
def clicked():
    res = 'Similarity is '+single_pred(txt.get(),txt2.get())
    lbl3.configure(text= res)
btn = Button(window, text="Check Similarity", command=clicked)
btn.grid(column=1, row=2)
window.mainloop()
