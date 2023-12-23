# importing modules
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk import pos_tag
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')

factual_tags = ["CD", "JJ", "JJR", "JJS", "NN", "NNS", "NNP", "NNPS"]

def stem(text):
    #tokenizer = word_tokenize(r'\w+')

    stemmer = SnowballStemmer("english")

    #sentence = "founders of the sss"
    words = word_tokenize(text)
    #words = tokenizer.tokenize(text)

    s = " ".join([stemmer.stem(w) for w in words if w not in stopwords.words("english")])
    return s
    
def tokenize_process(text):
    tokenizer = RegexpTokenizer(r'\w+')

    #stemmer = SnowballStemmer("english")

    #sentence = "founders of the sss"
    #words = word_tokenize(text)
    words = tokenizer.tokenize(text)

    s = [w.lower() for w in words if w not in stopwords.words("english")]
    return s

def pos_tag_method(text):
    tokenizer = RegexpTokenizer(r'\w+')

    words = tokenizer.tokenize(text.lower())    
    tokens_tag = pos_tag(words)
    ret_tags = [ele[0] for ele in tokens_tag if ele[1] in factual_tags and ele[0] not in stopwords.words("english")]

    return ret_tags



