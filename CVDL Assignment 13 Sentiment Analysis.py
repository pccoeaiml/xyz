#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv(r"Sentiment analysis_Social media post\sentiment_analysis.csv")

# Tokenization and padding sequences
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text'])
padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')

# Map sentiments to numerical values
sentiment_map = {'positive': 2, 'negative': 0, 'neutral': 1}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# Prepare training and test data
X = np.array(padded_sequences)
y = np.array(df['sentiment'])
y = to_categorical(y, num_classes=3)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128),  # Embedding layer without input_length
    LSTM(256, return_sequences=False),  # LSTM layer
    Dropout(0.2),  # Dropout layer
    Dense(32, activation='relu'),  # Dense layer
    Dense(3, activation='softmax')  # Output layer for 3 classes (positive, negative, neutral)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"\nTest accuracy: {test_accuracy:.4f}")

# Predict on a sample text without saving/loading the model
text = ["That's really good!"]

# Tokenize and pad the input text
tokenized_seq = tokenizer.texts_to_sequences(text)
padded_seq = pad_sequences(tokenized_seq, padding='post', truncating="post", maxlen=100)

# Predict sentiment for the text
prediction = model.predict(padded_seq)
print(f"Predicted sentiment: {prediction}")


# In[ ]:




