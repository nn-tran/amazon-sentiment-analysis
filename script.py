import numpy as np
import pandas as pd 
import re
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.optimizers import SGD, Adam
import bz2
import csv
from sklearn.metrics import roc_auc_score
import neural_net as net

# Load the training data 
training_data = bz2.BZ2File("./train.ft.txt.bz2")
training_data = training_data.readlines()
training_data = [x.decode('utf-8') for x in training_data]
print(len(training_data))

# Load the test data 
test_data = bz2.BZ2File("./test.ft.txt.bz2")
test_data = test_data.readlines()
test_data = [x.decode('utf-8') for x in test_data]
print(len(test_data))

# Split the data into labels and texts
train_labels = [int(re.findall(r'__label__(\d)', line)[0]) for line in training_data]
train_texts = [re.sub(r'__label__\d ', '', line) for line in training_data]

test_labels = [int(re.findall(r'__label__(\d)', line)[0]) for line in test_data]
test_texts = [re.sub(r'__label__\d ', '', line) for line in test_data]

# Convert labels to binary (0 and 1)
train_labels = [0 if label == 1 else 1 for label in train_labels]
test_labels = [0 if label == 1 else 1 for label in test_labels]

def clean_text(text):
    # Remove non-alphanumeric characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert multiple whitespace characters to a single space
    text = re.sub(r'\s+', ' ', text)
    # Convert the text to lowercase
    text = text.lower()
    return text

train_texts=pd.DataFrame(train_texts)[0].apply(clean_text)
test_texts=pd.DataFrame(test_texts)[0].apply(clean_text)

pd.DataFrame(test_labels).value_counts()


# Tokenization and padding
max_words = 1000
max_sequence_length = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)

X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(X_train, maxlen=max_sequence_length)
X_test = pad_sequences(X_test, maxlen=max_sequence_length)

X_train = np.array(X_train)
print(X_train.shape)
X_test = np.array(X_test)
print(X_test.shape)
y_train = np.array(train_labels)
print(y_train.shape)
y_test = np.array(test_labels)
print(y_test.shape)

# Instantiate and compile the model
vocab_size = max_words
embedding_dim = 128
lstm_units = 128
output_dim = 1

model = net.CustomModel(vocab_size, embedding_dim, lstm_units, output_dim, max_sequence_length)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

# Train the model
batch_size = 32
epochs = 5

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          verbose=1)

# Evaluate the model
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Test score:", score)
print("Test accuracy:", acc)

