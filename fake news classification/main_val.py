# import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer

# read data file
df_trn = pd.read_csv('leadership_train_data.csv', encoding='latin-1')
X_TRN= df_trn["combined_caption_title"].tolist()
Y_TRN= df_trn["label"].tolist()

###################################################
df_tst = pd.read_csv('val_data.csv', encoding='latin-1')
df_tst= df_tst.dropna()
X_TST= df_tst["combined_caption_title"].tolist()

#######################################################

#vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')
vectorizer = CountVectorizer()

train_corpus_tf_idf = vectorizer.fit_transform(X_TRN)
test_corpus_tf_idf = vectorizer.transform(X_TST)

# Create the SVC model object
svm_model= svm.SVC(kernel='linear') #gamma never affects linear kernel
svc = svm_model.fit(train_corpus_tf_idf, Y_TRN)
pred_targets = svc.predict(test_corpus_tf_idf)

# finding accuracy
#print("Accuracy on 70/30 division: {0:.2%}".format(accuracy_score(pred_targets, Y_TST)))
##
##
##loo = LeaveOneOut()
##scores = cross_val_score(svm_model, X_tfidf, Y, cv=loo)
##print("Accuracy with LOO: %0.2f (+/- %0.2f)" % (100*scores.mean(), scores.std() * 2))
