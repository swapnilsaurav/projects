import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv('D:/datasets/OnlineRetail/Superstore.csv', encoding='ISO-8859-1')

# Preprocessing
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Ship Date'] = pd.to_datetime(df['Ship Date'], dayfirst=True)
df = df[df['Sales'] > 0]

label_enc = LabelEncoder()
for col in ['Category', 'Sub-Category', 'Segment', 'Region']:
    df[col] = label_enc.fit_transform(df[col])

sns.set(style="whitegrid")

# 1. Predict Profit (Regression)
features = ['Sales', 'Quantity', 'Discount', 'Category', 'Sub-Category', 'Segment']
X = df[features]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg_model = LinearRegression().fit(X_train, y_train)
y_pred = reg_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"\n1. Linear Regression MSE for Profit Prediction: {mse:.2f}")

# 2. Classify Loss-Making Transactions
df['Loss_Label'] = df['Profit'].apply(lambda x: 1 if x < 0 else 0)
X_cls = df[features]
y_cls = df['Loss_Label']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42).fit(X_train_c, y_train_c)
y_pred_c = clf.predict(X_test_c)
acc = accuracy_score(y_test_c, y_pred_c)
print(f"\n2. Random Forest Accuracy for Loss Classification: {acc:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_c)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Loss Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 3. Customer Segmentation using Clustering
cust_df = df.groupby('Customer ID').agg({
    'Sales': 'sum',
    'Discount': 'mean',
    'Quantity': 'mean',
    'Segment': 'first'
}).reset_index()
cust_df['Segment'] = label_enc.fit_transform(cust_df['Segment'])
X_cust = cust_df[['Sales', 'Discount', 'Quantity', 'Segment']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cust)

kmeans = KMeans(n_clusters=3, random_state=42).fit(X_scaled)
cust_df['Cluster'] = kmeans.labels_

# PCA for plotting clusters
pca = PCA(n_components=2)
cust_pca = pca.fit_transform(X_scaled)
cust_df['PC1'] = cust_pca[:, 0]
cust_df['PC2'] = cust_pca[:, 1]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=cust_df, x='PC1', y='PC2', hue='Cluster', palette='Set2')
plt.title("Customer Segmentation with K-Means Clustering")
plt.show()

# 4. Predict Shipping Mode (Classification)
df['Ship Mode'] = label_enc.fit_transform(df['Ship Mode'])
X_ship = df[['Sales', 'Quantity', 'Discount', 'Category', 'Sub-Category', 'Segment']]
y_ship = df['Ship Mode']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_ship, y_ship, test_size=0.2, random_state=42)
ship_model = RandomForestClassifier(random_state=42).fit(X_train_s, y_train_s)
y_pred_s = ship_model.predict(X_test_s)
ship_acc = accuracy_score(y_test_s, y_pred_s)
print(f"\n4. Random Forest Accuracy for Shipping Mode Prediction: {ship_acc:.2%}")

# 5. Sales Forecasting (Time Series)
df_sales_ts = df.groupby(df['Order Date'].dt.to_period('M')).agg({'Sales': 'sum'}).reset_index()
df_sales_ts['Order Date'] = df_sales_ts['Order Date'].dt.to_timestamp()

plt.figure(figsize=(14, 6))
sns.lineplot(data=df_sales_ts, x='Order Date', y='Sales', marker='o')
plt.title("Monthly Sales Over Time")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
