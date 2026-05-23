from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Activate -> terminal -> command prompt -> venv\Scripts\activate
# py filename.py -> run code in terminal

## Pandas Methods:
# .head() -> see first 5 columns
# .info() -> see column names and data types
# .describe() -> see statistics such as mean, min, max, std, for each column
# .isnull().sum() -> count number of missing values

## Data Cleaning:
# df.drop() -> used to removed rows/columns from a dataframe, returns a new dataframe so must reasign
# df.dropna() -> remove data where there is a missing value
# df.fillna() -> fill missing values
# df.replace() -> replace values
# df.drop_duplicates() -> drops duplicate values

# Loading the dataset
housing = fetch_california_housing()

# convert the dataset to a dataframe, the top of each column becomes the descriptive name (default is indices)
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# add a new column "MedHouseValue" with data as housing.target
df["MedHouseValue"] = housing.target

plt.hist(df["MedHouseValue"])
plt.title("California House Values")
plt.ylabel("House Value (100s)")
plt.xlabel("Intervals")
plt.show()

x = df["MedInc"]
y = df["MedHouseValue"]
plt.scatter(x, y)
plt.title("Income vs House Value")
plt.ylabel("Median Income")
plt.xlabel("House Value")
plt.show()

print()