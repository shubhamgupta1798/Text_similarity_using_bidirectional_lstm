import warnings
warnings.filterwarnings("ignore")
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize
import re
class pre:
    def preprocessing(sentence):
        results=[]
        for temp in sentence:
            example_sent = temp
            stop_words = set(sw.words('english'))
            word_tokens = word_tokenize(example_sent)
            punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            filtered=''
            for w in word_tokens:
                if w not in stop_words:
                    filtered+=w+' '
            rem_pun=''
            for x in filtered:
                if x not in punctuations:
                    rem_pun+=x
            results.append(rem_pun)
            #print(rem_pun)
        return results
    def preprocessing_single_sentence(sentence):
        example_sent = sentence
        stop_words = set(sw.words('english'))
        word_tokens = word_tokenize(example_sent)
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        filtered=''
        for w in word_tokens:
            if w not in stop_words:
                filtered+=w+' '
        rem_pun=''
        for x in filtered:
            if x not in punctuations:
                rem_pun+=x
        return rem_pun
