import random
import json
import pickle
import numpy as np
import nltk
from nltk import WordNetLemmatizer
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import legacy as keras_legacy
from keras.optimizers import SGD

from tensorflow.python.keras.optimizers import *

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Pasa la información a unos y ceros según las palabras presentes en cada categoría para hacer el entrenamiento
training = np.zeros((len(documents), len(words)))
output = np.zeros((len(documents), len(classes)))
for i, document in enumerate(documents):
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in word_patterns:
        if word in words:
            training[i, words.index(word)] = 1
    output[i, classes.index(document[1])] = 1

# Mezcla los datos de entrenamiento
data = list(zip(training, output))
random.shuffle(data)
training, output = zip(*data)

training = np.array(training)
output = np.array(output)
print(training)
print(output)


train_x = training
train_y = output

#Creamos la red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

#Creamos el optimizador y lo compilamos
sgd = keras_legacy.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#Entrenamos el modelo y lo guardamos
train_process = model.fit(train_x, train_y, epochs=100, batch_size=5, verbose=1)
model.save("chatbot_model.h5", train_process)