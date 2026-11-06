import streamlit as st
from nltk.stem import SnowballStemmer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import dill
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
words=stopwords.words('english')
with open('tf.pkl','rb') as f:
  tf=dill.load(f)
with open('model.pkl','rb') as f:
  model=dill.load(f)
def remove_impurity(col):
  col=col.lower()
  col=re.sub('<.*?>',' ',col)
  col=re.sub('http?://\S+|www\.\S+',' ',col)
  return re.sub('[^a-zA-Z]',' ',col)
def remove_stopwords(col):
  L=[]
  tokens=word_tokenize(col)
  for i in tokens:
    if i not in words:
      L.append(i)
  return " ".join(L)
sbs=SnowballStemmer('english')
def stemmer(col):
  L=[]
  tokens=word_tokenize(col)
  for i in tokens:
    L.append(sbs.stem(i))
  return " ".join(L)
def predict(msg:str):
  msg=remove_impurity(msg)
  msg=remove_stopwords(msg)
  msg=stemmer(msg)
  msg=tf.transform([msg])
  prediction=model.predict(msg)[0]
  if prediction==0:
    return 'This is Negative Sentiment'
  else:
    return 'This is Positive Sentiment'
st.title('REVIW ANALYSER')
container=st.container(border=True)
message=container.text_input('Enter Review')
if st.button('Predict'):
  st.header(predict(message))