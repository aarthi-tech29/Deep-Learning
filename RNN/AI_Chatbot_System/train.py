import json
import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,LSTM,Dense,Dropout
from sklearn.preprocessing import LabelEncoder

with open("data/intents.json") as file:
    data = json.load(file)

texts = []
labels = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

max_len = max(len(x) for x in sequences)

X = pad_sequences(sequences,maxlen=max_len)

encoder = LabelEncoder()
y = encoder.fit_transform(labels)

model = Sequential()

model.add(
    Embedding(
        input_dim=len(tokenizer.word_index)+1,
        output_dim=64,
        input_length=max_len
    )
)

model.add(LSTM(128))

model.add(Dropout(0.5))

model.add(Dense(64,activation='relu'))

model.add(Dense(len(set(labels)),activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(X,y,epochs=200,batch_size=8)

model.save("model/chatbot_model.h5")

pickle.dump(
    tokenizer,
    open("model/tokenizer.pkl","wb")
)

pickle.dump(
    encoder,
    open("model/label_encoder.pkl","wb")
)

pickle.dump(
    max_len,
    open("model/max_len.pkl","wb")
)

print("Training Completed")