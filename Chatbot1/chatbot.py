# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 23:37:40 2018

@author: Hafsa Asad
"""
#%%
import nltk  
#nltk.download()

# Sample code to remove noisy words from a text

noise_list = ["is", "a", "this", "..."] 
def _remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

_remove_noise("this is a sample text")


from nltk.stem.wordnet import WordNetLemmatizer 
lem = WordNetLemmatizer()

from nltk.stem.porter import PorterStemmer 
stem = PorterStemmer()

word = "multiplying" 
lem.lemmatize(word, "v")

#=====================================================================================
#%%
#  Replacing Slang Words. We can make a dictionary of slang words and repalce them with real words
lookup_dict = {'rt':'Retweet', 'dm':'direct message', "awsm" : "awesome", "luv" :"love"}
def _lookup_words(input_text):
    words = input_text.split() 
#input_text = 'Hi you are awsm'
#words=input_text.split()
    new_words = [] 
    for word in words:
        lower_case=word.lower()
        if lower_case in lookup_dict:
            word = lookup_dict[word.lower()]
                  
        new_words.append(word) 
    new_text = " ".join(new_words)

        
    return new_text

new_text= _lookup_words("You are awsm because you send me dm")
#==============================================================================
#  Parts of Speech Tagging
#==============================================================================
from nltk import word_tokenize, pos_tag
text = "I am preparing codes for machine learning workshop at Experts Vision"
tokens = word_tokenize(text)
print (pos_tag(tokens))
#%%
#==============================================================================
#  Topic Modelling
#==============================================================================

""""
Topic modeling is a process of automatically identifying the topics present in a text corpus, it derives the hidden patterns among the words in the corpus in an unsupervised manner. Topics are defined as “a repeating pattern of co-occurring terms in a corpus”. A good topic model results in – “health”, “doctor”, “patient”, “hospital” for a topic – Healthcare, and “farm”, “crops”, “wheat” for a topic – “Farming”.

Latent Dirichlet Allocation (LDA) is the most popular topic modelling technique,
"""""
doc1 = "Sugar is bad to consume. My sister likes to have sugar, but not my father." 
doc2 = "My father spends a lot of time driving my sister around to dance practice."
doc3 = "Doctors suggest that driving may cause increased stress and blood pressure."
doc_complete = [doc1, doc2, doc3]
doc_clean = [doc.split() for doc in doc_complete]

import gensim
from gensim import corpora


# Creating the term dictionary of our corpus, where every unique term is assigned an index.  
dictionary = corpora.Dictionary(doc_clean)

# Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above. 
doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

# Creating the object for LDA model using gensim library
Lda = gensim.models.ldamodel.LdaModel

# Running and Training LDA model on the document term matrix
ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

# Results 
print(ldamodel.print_topics())
#==============================================================================
#%%
"""
A combination of N words together are called N-Grams. N grams (N > 1) are generally more informative as
 compared to words (Unigrams) as features. Also, bigrams (N = 2) are considered as the most important 
 features of all the others. The following code generates bigram of a text.
"""
def generate_ngrams(text, n):
    words = text.split()
    output = []  
    for i in range(len(words)-n+1):
        output.append(words[i:i+n])
    return output

#==============================================================================
#%%
"""
Text data can also be quantified directly into numbers using several techniques described in this 
section:
A.  Term Frequency – Inverse Document Frequency (TF – IDF)
"""   

from sklearn.feature_extraction.text import TfidfVectorizer
obj = TfidfVectorizer()
corpus = ['This is sample document.', 'another random document.', 'third sample document text']
X = obj.fit_transform(corpus) 

#=============================================================================
#%%
"""
Word2Vec and GloVe are the two popular models to create word embedding of a text. These models takes a 
text corpus as input and produces the word vectors as output.

Word2Vec model is composed of preprocessing module, a shallow neural network model called Continuous 
Bag of Words and another shallow neural network model called skip-gram. These models are widely used 
for all other nlp problems. It first constructs a vocabulary from the training corpus and then learns
word embedding representations. Following code using gensim package prepares the word embedding as 
the vectors.
"""    
from gensim.models import Word2Vec
sentences = [['data', 'science'], ['sajid', 'science', 'data', 'analytics'],['machine', 'learning'], ['deep', 'learning']]

# train the model on your corpus  
model = Word2Vec(sentences, min_count = 1)

print (model.similarity('data', 'science'))
print (model['learning'])