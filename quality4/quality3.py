import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Load the data (skip if the notebook already created these variables)
train_df = pd.read_csv('train-data.tsv', sep='\t', header=None, names=['label', 'message'])
test_df = pd.read_csv('valid-data.tsv', sep='\t', header=None, names=['label', 'message'])

# 2. Labels: ham = 0, spam = 1
train_labels = (train_df['label'] == 'spam').astype('float32').values
test_labels = (test_df['label'] == 'spam').astype('float32').values

train_text = train_df['message'].astype(str).values
test_text = test_df['message'].astype(str).values

# 3. Text vectorization (learned from training data only)
VOCAB_SIZE = 10000
MAX_LEN = 60

vectorizer = layers.TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_LEN
)
vectorizer.adapt(train_text)

# 4. Model
model = keras.Sequential([
    keras.Input(shape=(1,), dtype=tf.string),
    vectorizer,
    layers.Embedding(VOCAB_SIZE, 32),
    layers.GlobalAveragePooling1D(),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 5. Train
history = model.fit(
    train_text, train_labels,
    epochs=15,
    batch_size=32,
    validation_data=(test_text, test_labels),
    verbose=2
)

# 6. Prediction function
def predict_message(pred_text):
    p = float(model.predict(np.array([pred_text]), verbose=0)[0][0])
    label = 'spam' if p > 0.5 else 'ham'
    return [p, label]

# Quick check before running the final cell
print(predict_message("how are you doing today?"))
print(predict_message("You have won a $1000 prize! Call now to claim your free reward."))
