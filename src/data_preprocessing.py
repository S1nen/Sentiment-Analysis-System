import pandas as pd
import nltk
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
nltk.download("stopwords")
STOPWORDS=set(stopwords.words("english"))
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
data=pd.read_csv("data/amazon_alexa_cleaned.csv")

#removing stopwords and stemming
corpus=[]
stemmer=PorterStemmer()
for i in range(0,data.shape[0]):
    review=re.sub('[^a-zA-Z]',' ',data.iloc[i]["verified_reviews"])
    review=review.lower().split()
    review=[stemmer.stem(word) for word in review if not word in STOPWORDS]
    review=' '.join(review)
    corpus.append(review)

cv=CountVectorizer(max_features=2500)
x=cv.fit_transform(corpus).toarray()
y=data["feedback"].values
pickle.dump(cv,open('models/countVectorizer.pkl',"wb"))

print("x shape",x.shape)
print("y shape",y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=20)
print("x train shape:",x_train.shape)
print("y train shape:",y_train.shape)
print("x test shape:",x_test.shape)
print("y test shape:",y_test.shape)

scaler=MinMaxScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

pickle.dump(scaler,open("models/scaler.pkl","wb"))