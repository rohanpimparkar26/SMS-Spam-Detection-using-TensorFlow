# index.py

#  Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#  Step 2: Load and Clean the Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep only necessary columns
df = df[['v1', 'v2']]
df.columns = ['label', 'text']  # Rename for clarity

#  Step 3: Explore the Dataset (EDA)
print(" Basic Information:")
print(df.info())
print("\n Class Distribution (Ham = 0, Spam = 1):")
print(df['label'].value_counts())

# Plot class distribution in black & white
plt.figure(figsize=(6, 4))
sns.set(style="whitegrid")
sns.countplot(x='label', data=df, palette=['black', 'gray'])
plt.title("Class Distribution: Ham vs Spam", fontsize=12)
plt.xlabel("Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

#  Step 4: Encode Labels (Ham=0, Spam=1)
encoder = LabelEncoder()
df['label'] = encoder.fit_transform(df['label'])

#  Step 5: Prepare Texts and Labels
texts = df['text'].values
labels = df['label'].values

#  Step 6: Tokenize and Pad Texts
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded_sequences = pad_sequences(sequences, padding='post')

#  Step 7: Split into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, labels, test_size=0.2, random_state=42
)

#  Step 8: Build the Neural Network Model
model = keras.Sequential([
    layers.Embedding(input_dim=5000, output_dim=16, input_length=padded_sequences.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(24, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

#  Step 9: Compile the Model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#  Step 10: Train the Model
print("\n Training the model...\n")
model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test),
    verbose=2
)

# 📌 Step 11: Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n Final Test Accuracy: {accuracy:.4f}")
