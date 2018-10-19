# import relevant libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
#from sklearn.feature_extraction.text import TfidfVectorizer

df_trn = pd.read_csv('leadership_train_data.csv', encoding='latin-1')
X_TRN= df_trn["posts_"].tolist()
Y_TRN= df_trn["label"].tolist()

###################################################
df_tst = pd.read_csv('val_data.csv', encoding='latin-1')
X_TST= df_tst["title"].tolist()

#######################################################

#vectorizer = TfidfVectorizer(min_df=5, max_df = 0.8, sublinear_tf=True, use_idf =True, stop_words = 'english')
vectorizer = CountVectorizer()

train_corpus_tf_idf = vectorizer.fit_transform(X_TRN)
test_corpus_tf_idf = vectorizer.transform(X_TST)

# Create the SVC model object
svm_model= svm.SVC(kernel='linear') #gamma never affects linear kernel
svc = svm_model.fit(train_corpus_tf_idf, Y_TRN)
pred_targets = svc.predict(test_corpus_tf_idf)
df_tst['Label']= pred_targets

# DF TO CSV
df_tst.to_csv('ValidationResults.csv')