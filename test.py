import streamlit as st
import re
import pickle
import nltk
import tensorflow as tf
#from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.sequence import pad_sequences
#from keras.models import load_model

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

with open('tokenizer.pickle', 'rb') as f:
      tokenizer=pickle.load(f)
#with open('tokenizer.pickle', 'rb') as f1:
#      nlp=pickle.load(f1)

model = load_model("M3P_512_spam.h5")


st.header("Spam Classifier")
st.subheader("Enter the message you want to analyze")

text_input = st.text_area( "Enter sentence",height=50)
 
# model_select = st.selectbox("Model Selection", ["Naive Bayes", "SVC", "Logistic Regression"])
if st.button("Analyze"):
    print("Result")
    pattern = r'((http[s]*)?(:\/\/)?(www\.)?[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}[-a-zA-Z0-9()@:%_+.~#?&/=]*)'
    pattern2= r'[0-9]+'
    
    text_input= re.sub(pattern,"",text_input)
    text_input= re.sub(pattern2,"",text_input)
    text_input= text_input.replace("+","")
    #doc=nlp(text_input)
    #text= " ".join([token.lemma for token in doc  if not (token.is_stop or token.is_punct)])
    
    #doc=nlp(text_input)
    #s=" ".join([token.lemma_ for token in doc if not (token.is_stop or token.is_punct)])
    #st.write(s)
    encoded = tokenizer.texts_to_sequences(text_input)
    padded = pad_sequences(encoded, maxlen=76, padding='post')
    st.write(padded)
    ypred = model.predict([padded,padded,padded])>0.5
   
