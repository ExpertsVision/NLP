# import relevant libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import svm
# read data file
df = pd.read_csv('leadership_train_data.csv', encoding='latin-1')

considered_params= ['combined_caption_title', 'label']
new_df = df[considered_params].copy()

dataset = new_df.values
X = dataset[:,0:1] # starts AT column 0 and ends BEFORE column 1
Y = dataset[:,1] # column 6 ONLY

X= df["combined_caption_title"].tolist()
Y= df["label"].tolist()

count_vect = CountVectorizer()
X_counts = count_vect.fit_transform(X)
X_counts.shape

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_counts)
XX = np.array(X_tfidf)

## DIVIDE TO 80/20 RANDOMLY TO DIVIDE INTO TRAINING AND TESTING DATA
X_trn, X_tst, Y_trn, Y_tst = train_test_split(X_tfidf, Y, test_size = .3)
#X_trn is training inputs

# Create the SVC model object
svm_model= svm.SVC(kernel='linear') #gamma never affects linear kernel
svc = svm_model.fit(X_trn, Y_trn)
pred_targets = svc.predict(X_tst)

# finding accuracy
print("Accuracy on 70/30 division: {0:.2%}".format(accuracy_score(pred_targets, Y_tst)))
#
#
#loo = LeaveOneOut()
#scores = cross_val_score(svm_model, X_tfidf, Y, cv=loo)
#print("Accuracy with LOO: %0.2f (+/- %0.2f)" % (100*scores.mean(), scores.std() * 2))
