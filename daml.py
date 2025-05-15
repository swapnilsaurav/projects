'''
Example 1: Customer Demographics Summary
Goal: Understand the customer base.
•	The average age of customers
•	Gender distribution
•	Income distribution by city
🛠 Techniques Used:
•	Group-by analysis
•	Descriptive statistics
•	Histograms / bar charts

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("D:\\datasets\\Customer_Transactions_Dataset.csv")

# Set plot style
sns.set(style="whitegrid")

# Compute average age
average_age = df['Age'].mean()

# Compute gender distribution
gender_distribution = df['Gender'].value_counts()

# Compute average income by city (top 10 cities)
top_cities = df['City'].value_counts().nlargest(10).index
income_by_city = df[df['City'].isin(top_cities)].groupby('City')['AnnualIncome'].mean().sort_values()

# Create visualizations
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Plot 1: Age distribution
sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0], color='lightblue')
axes[0].set_title(f'Age Distribution (Average Age: {average_age:.1f})')
axes[0].set_xlabel('Age')
axes[0].set_ylabel('Count')

# Plot 2: Gender distribution
sns.barplot(x=gender_distribution.index, y=gender_distribution.values, ax=axes[1], palette='Set2')
axes[1].set_title('Gender Distribution')
axes[1].set_xlabel('Gender')
axes[1].set_ylabel('Count')

# Plot 3: Income distribution by city
sns.barplot(x=income_by_city.values, y=income_by_city.index, ax=axes[2], palette='coolwarm')
axes[2].set_title('Average Annual Income by Top 10 Cities')
axes[2].set_xlabel('Average Income (₹)')
axes[2].set_ylabel('City')

# Display plots
plt.tight_layout()
plt.show()

'''
Example 2: Product Performance

Goal: Identify best-selling product categories and their average revenue.
•	Total sales per product category
•	Average purchase amount per category

Techniques Used:
•	Aggregation functions (sum(), mean())
•	Pivot tables
•	Visualization (e.g., pie chart of category-wise revenue)

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("D:\\datasets\\Customer_Transactions_Dataset.csv")

# Group by ProductCategory to get total and average revenue
product_summary = df.groupby('ProductCategory').agg(
    TotalRevenue=pd.NamedAgg(column='PurchaseAmount', aggfunc='sum'),
    AverageRevenue=pd.NamedAgg(column='PurchaseAmount', aggfunc='mean'),
    PurchaseCount=pd.NamedAgg(column='PurchaseAmount', aggfunc='count')
).sort_values(by='TotalRevenue', ascending=False)

# Create visualizations
plt.figure(figsize=(14, 6))

# Plot 1: Total revenue pie chart
plt.subplot(1, 2, 1)
plt.pie(product_summary['TotalRevenue'], labels=product_summary.index, autopct='%1.1f%%', startangle=140)
plt.title('Total Sales Distribution by Product Category')

# Plot 2: Average revenue bar chart
plt.subplot(1, 2, 2)
sns.barplot(x=product_summary.index, y=product_summary['AverageRevenue'], palette='viridis')
plt.title('Average Purchase Amount by Product Category')
plt.ylabel('Average Revenue (₹)')
plt.xlabel('Product Category')

plt.tight_layout()
plt.show()

# Display summary table
print(product_summary.reset_index())

'''
Example 3: Customer Segments Based on Spending Score

Goal: Group customers by low, medium, and high spending score to design marketing campaigns.
•	Low: 1-40, Medium: 41-70, High: 71-100
•	Count of customers in each segment
•	Average income and purchase amount per segment

Techniques Used:
•	Binning / Categorization
•	Grouping and aggregation

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("D:/datasets/Customer_Transactions_Dataset.csv")

# Create Spending Segment based on Spending Score
bins = [0, 40, 70, 100]
labels = ['Low', 'Medium', 'High']
df['SpendingSegment'] = pd.cut(df['SpendingScore'], bins=bins, labels=labels, right=True)

# Group by Spending Segment
segment_summary = df.groupby('SpendingSegment').agg(
    CustomerCount=('CustomerID', 'count'),
    AverageIncome=('AnnualIncome', 'mean'),
    AveragePurchaseAmount=('PurchaseAmount', 'mean')
).reset_index()

# Plotting
plt.figure(figsize=(15, 5))

# Plot 1: Count of customers in each segment
plt.subplot(1, 3, 1)
sns.barplot(data=segment_summary, x='SpendingSegment', y='CustomerCount', palette='pastel')
plt.title('Customer Count per Spending Segment')
plt.ylabel('Number of Customers')
plt.xlabel('Spending Segment')

# Plot 2: Average Income per Segment
plt.subplot(1, 3, 2)
sns.barplot(data=segment_summary, x='SpendingSegment', y='AverageIncome', palette='cool')
plt.title('Average Income per Spending Segment')
plt.ylabel('Average Income (₹)')
plt.xlabel('Spending Segment')

# Plot 3: Average Purchase Amount per Segment
plt.subplot(1, 3, 3)
sns.barplot(data=segment_summary, x='SpendingSegment', y='AveragePurchaseAmount', palette='viridis')
plt.title('Average Purchase Amount per Segment')
plt.ylabel('Average Purchase (₹)')
plt.xlabel('Spending Segment')

plt.tight_layout()
plt.show()

# Display the segment summary
print(segment_summary)

'''
Data Science: Predictive & Prescriptive Modeling

Example 1: Predicting High-Value Customers
Goal: Build a classification model to predict whether a customer will make a high-value purchase (say > ₹1000).
•	Features: Age, Gender, Income, Spending Score, Product Category
•	Target: PurchaseAmount > 1000 (Yes/No)
🛠 Techniques Used:
•	Logistic Regression / Random Forest
•	Feature scaling & encoding
•	Model evaluation (precision, recall)

'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv("D:\\datasets\\Customer_Transactions_Dataset.csv")

# Create the target variable: HighValue (1 if PurchaseAmount > 1000, else 0)
df['HighValue'] = (df['PurchaseAmount'] > 1000).astype(int)

# Encode categorical variables
le_gender = LabelEncoder()
le_category = LabelEncoder()
df['GenderEncoded'] = le_gender.fit_transform(df['Gender'])
df['CategoryEncoded'] = le_category.fit_transform(df['ProductCategory'])

# Select features and target
features = ['Age', 'GenderEncoded', 'AnnualIncome', 'SpendingScore', 'CategoryEncoded']
X = df[features]
y = df['HighValue']

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# Train Random Forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Generate evaluation metrics
print("=== Logistic Regression Classification Report ===")
print(classification_report(y_test, y_pred_logreg))

print("=== Random Forest Classification Report ===")
print(classification_report(y_test, y_pred_rf))

print("=== Random Forest Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_rf))

'''
Example 2: Customer Lifetime Value Prediction
Goal: Estimate potential future value of customers based on past purchases.
•	Features: Time since signup, purchase frequency, average amount, recency of last purchase
Techniques Used:
•	Regression models (Linear, Ridge)
•	Time series components
•	CLV (Customer Lifetime Value) models

'''
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("D:/datasets/Customer_Transactions_Dataset.csv")

# Convert dates
df['SignupDate'] = pd.to_datetime(df['SignupDate'])
df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'])

# Reference date for recency calculation
reference_date = datetime(2025, 4, 1)

# Feature engineering
df['DaysSinceSignup'] = (reference_date - df['SignupDate']).dt.days
df['Recency'] = (reference_date - df['LastPurchaseDate']).dt.days

# Aggregate data for each customer
purchase_counts = df.groupby('CustomerID').size().rename("PurchaseFrequency")
average_amounts = df.groupby('CustomerID')['PurchaseAmount'].mean().rename("AvgPurchaseAmount")
days_since_signup = df.groupby('CustomerID')['DaysSinceSignup'].max()
recency = df.groupby('CustomerID')['Recency'].min()

# Merge features
clv_data = pd.concat([purchase_counts, average_amounts, days_since_signup, recency], axis=1).reset_index()
clv_data['CLV'] = clv_data['PurchaseFrequency'] * clv_data['AvgPurchaseAmount']

# Features and target
features = ['PurchaseFrequency', 'AvgPurchaseAmount', 'DaysSinceSignup', 'Recency']
X = clv_data[features]
y = clv_data['CLV']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)

# Output results
print("=== Linear Regression MSE ===")
print(mse_lr)
print("\n=== Ridge Regression MSE ===")
print(mse_ridge)
print("\n=== Sample CLV Predictions (Linear) ===")
print(y_pred_lr[:5])
print("\n=== Sample CLV Predictions (Ridge) ===")
print(y_pred_ridge[:5])

'''
Example 3: Customer Clustering for Targeted Marketing
Goal: Use unsupervised learning to cluster customers into segments for personalized campaigns.
•	Use features: Age, Income, Spending Score, ProductCategory
•	Identify distinct clusters using K-Means or DBSCAN
🛠 Techniques Used:
•	K-Means clustering
•	PCA for dimensionality reduction
•	Cluster visualization (2D scatter plots)

'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load dataset
df = pd.read_csv("D:/datasets/Customer_Transactions_Dataset.csv")

# Encode ProductCategory
le = LabelEncoder()
df['ProductCategoryEncoded'] = le.fit_transform(df['ProductCategory'])

# Features for clustering
features = ['Age', 'AnnualIncome', 'SpendingScore', 'ProductCategoryEncoded']
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

# Plot clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Customer Segments (K-Means Clustering with PCA)')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()

# Show sample output
print(df[['CustomerID', 'Age', 'AnnualIncome', 'SpendingScore', 'ProductCategory', 'Cluster']].head())
