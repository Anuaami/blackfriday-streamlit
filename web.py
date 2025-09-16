import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import joblib

print("⏳ Training the model...")

# Load dataset
df = pd.read_csv("train.csv")

# Fill missing values safely
df.loc[:, 'Product_Category_2'] = df['Product_Category_2'].fillna(-1)
df.loc[:, 'Product_Category_3'] = df['Product_Category_3'].fillna(-1)

# Encode categorical variables
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Age'] = le.fit_transform(df['Age'])
df['City_Category'] = le.fit_transform(df['City_Category'])
df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].str.replace('+', '', regex=False).astype(int)
df['Marital_Status'] = df['Marital_Status'].astype(int)

# Drop IDs
df.drop(['User_ID', 'Product_ID'], axis=1, inplace=True)

# Split data
X = df.drop('Purchase', axis=1)
y = df['Purchase']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"✅ Model trained. RMSE: {rmse:.2f}")

# Save model
joblib.dump(model, 'blackfriday_modelstr.pkl')
print("✅ Model saved as blackfriday_modelstr.pkl")