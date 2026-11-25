import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Convert categorical columns to numbers
dataset = dataset.copy()
dataset['sex'] = dataset['sex'].map({'female': 0, 'male': 1})
dataset['smoker'] = dataset['smoker'].map({'no': 0, 'yes': 1})
dataset = pd.get_dummies(dataset, columns=['region'], dtype=float)  # region has 4 values, so one-hot encode it

# 2. 80/20 split
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# 3. Pop off the labels
train_labels = train_dataset.pop('expenses')
test_labels = test_dataset.pop('expenses')

# 4. Normalization layer (adapted on training data only)
normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset, dtype='float32'))

# 5. Model
model = keras.Sequential([
    normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mae',
    metrics=['mae', 'mse']
)

# 6. Train
history = model.fit(
    np.array(train_dataset, dtype='float32'),
    train_labels,
    epochs=200,
    validation_split=0.2,
    verbose=0
)

# Quick check before running the final cell
loss, mae, mse = model.evaluate(
    np.array(test_dataset, dtype='float32'), test_labels, verbose=2)
print("Mean Absolute Error: {:5.2f} expenses".format(mae))
