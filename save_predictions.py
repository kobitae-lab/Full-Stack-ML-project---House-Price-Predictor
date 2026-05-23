from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from pymongo import MongoClient
import tensorflow as tf

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

model = tf.keras.models.load_model("house_model.keras")

predictions = model.predict(scaled_X_test).flatten()

documents = []

for i in range(len(scaled_X_test)):
    # X_test.iloc[i] gets the original unscaled row
    # so the stored features are human readable
    original_row = X_test.iloc[i]

    doc = {
        "features": {
            "MedInc":     float(original_row["MedInc"]),
            "HouseAge":   float(original_row["HouseAge"]),
            "AveRooms":   float(original_row["AveRooms"]),
            "AveBedrms":  float(original_row["AveBedrms"]),
            "Population": float(original_row["Population"]),
            "AveOccup":   float(original_row["AveOccup"]),
            "Latitude":   float(original_row["Latitude"]),
            "Longitude":  float(original_row["Longitude"]),
        },
        # Multiply by 100000 to convert from $100k units to dollars
        "actual_price":    round(float(array_Y_test[i]) * 100000, 2),
        "predicted_price": round(float(predictions[i])  * 100000, 2),
        # Difference between actual and predicted
        "error":           round(float(abs(array_Y_test[i] - predictions[i])) * 100000, 2)
    }
    documents.append(doc)

client = MongoClient("mongodb://localhost:27017")
db = client["house_predictor"]
collection = db["predictions"]

collection.delete_many({})
result = collection.insert_many(documents)
print(f"Inserted {len(result.inserted_ids)} predictions into MongoDB")

print("\nFirst 3 predictions from MongoDB:")
for doc in collection.find().limit(3):
    print(f"  Actual: ${doc['actual_price']:,.0f} | "
          f"Predicted: ${doc['predicted_price']:,.0f} | "
          f"Error: ${doc['error']:,.0f}")
    
all_errors = [doc["error"] for doc in collection.find()]
avg_error  = np.mean(all_errors)
print(f"\nAverage prediction error across {len(all_errors)} houses: ${avg_error:,.0f}")


