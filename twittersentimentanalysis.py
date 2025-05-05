import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#

data = pd.read_csv('Twitter_Data.csv')

data.head()

data['clean text'] = data['text'].astype(str).str.replace(r'[^a-zA-Z\s]','',regex=True).str.lower()

sentiment_map ={'negative':0,'neutral':1,'positive':2}
data['sentiment'] = data['sentiment'].str.lower().map(sentiment_map)

X = data['clean text']
y = data['sentiment']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

vocab_size = 5000
max_len =100

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=max_len)
X_test_pad = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=max_len)

LabelEncoder = LabelEncoder()
y_train_encoded = LabelEncoder.fit_transform(y_train)
y_test_encoded = LabelEncoder.transform(y_test)

num_classes = len(LabelEncoder.classes_)
y_train_cat = tf.keras.utils.to_categorical(y_train_encoded, num_classes=num_classes)
y_test_cat = tf.keras.utils.to_categorical(y_test_encoded, num_classes=num_classes)

y_train_cat

model = Sequential([
    Embedding(vocab_size,100,input_length=max_len),
    LSTM(128,dropout=0.2,recurrent_dropout=0.2),


    Dropout(0.5),
    Dense(num_classes,activation='softmax')
])

model.compile(optimizer ='adam',loss ='categorical_crossentropy',metrics=['accuracy'])

model.fit(X_train_pad,y_train_cat,validation_data=(X_test_pad,y_test_cat),epochs=5,batch_size=32)

model.save('sentiment_model.keras')

dump(tokenizer,'tokenixer.joblib')
dump(LabelEncoder,'LabelEnocoder.joblib')

print('Completed')