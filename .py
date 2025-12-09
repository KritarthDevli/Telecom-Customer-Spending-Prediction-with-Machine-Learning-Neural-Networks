import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("CustomerChurn.csv")
print(df.head())
print(df.info())
print(df.describe())
print(df.isna().sum())

df = df.drop_duplicates()
df = df.dropna()
categorical_columns = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop("totalCharges", axis=1)
y = df["totalCharges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_processed = scaler.fit_transform(X_train)
X_test_processed = scaler.transform(X_test)

model_sklearn = LinearRegression()
model_sklearn.fit(X_train_processed, y_train)
y_pred_sklearn = model_sklearn.predict(X_test_processed)
mse_sklearn = mean_squared_error(y_test, y_pred_sklearn)
print(mse_sklearn)

model_keras = Sequential()
model_keras.add(Dense(64, activation='relu', input_dim=X_train_processed.shape[1]))
model_keras.add(Dense(32, activation='relu'))
model_keras.add(Dense(1))
model_keras.compile(optimizer='adam', loss='mse')
model_keras.fit(X_train_processed, y_train, epochs=50, batch_size=32, verbose=0)

y_pred_keras = model_keras.predict(X_test_processed)
mse_keras = mean_squared_error(y_test, y_pred_keras)
print(mse_keras)

print(mse_sklearn, mse_keras)
df.sample(10)
