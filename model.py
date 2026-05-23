from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras import layers

TF_ENABLE_ONEDNN_OPTS=0
housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns = housing.feature_names)

df["HouseValue"] = housing.target

# Splitting the data frame 
X = df.drop(columns=["HouseValue"])
Y = df["HouseValue"]

# Splits the data into training and testing ,80% training, 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scaling in the data so that each column has equal influence despite difference in numbers
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

array_Y_train = Y_train.to_numpy()
array_Y_test = Y_test.to_numpy()

model = keras.Sequential(
    [   
        layers.Input(shape=(8,)),
        layers.Dense(32, activation='relu',),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(1)
    ]
)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

model.fit(
    scaled_X_train,
    array_Y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2
)

model.evaluate(scaled_X_train, array_Y_train)
print(model.summary())
model.save("house_model.keras")
