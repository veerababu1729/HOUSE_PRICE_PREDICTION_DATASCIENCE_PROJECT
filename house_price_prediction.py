# House Price Prediction - Complete Data Science Project
# This project covers pandas, numpy, and scikit-learn fundamentals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set display options for better output
pd.set_option('display.max_columns', None)
np.random.seed(42)

print("🏠 HOUSE PRICE PREDICTION PROJECT")
print("=" * 50)
print("Libraries covered: pandas, numpy, scikit-learn")
print("=" * 50)

# STEP 1: CREATE SAMPLE DATASET USING NUMPY
print("\n📊 STEP 1: Creating Sample Dataset with NumPy")
print("-" * 40)

# Generate synthetic house data
n_samples = 1000

# Using numpy to generate features
np.random.seed(42)
bedrooms = np.random.randint(1, 6, n_samples)
bathrooms = np.random.randint(1, 4, n_samples)
sqft = np.random.normal(2000, 500, n_samples)
sqft = np.clip(sqft, 500, 5000)  # Keep realistic range
age = np.random.randint(0, 50, n_samples)
location = np.random.choice(['Downtown', 'Suburbs', 'Rural'], n_samples, p=[0.3, 0.5, 0.2])

# Create price using a realistic formula with some noise
location_multiplier = np.where(location == 'Downtown', 1.5,
                      np.where(location == 'Suburbs', 1.2, 1.0))

price = (sqft * 8000 + 
         bedrooms * 500000 + 
         bathrooms * 400000 - 
         age * 50000) * location_multiplier + np.random.normal(0, 800000, n_samples)

price = np.clip(price, 2500000, 50000000)  # Keep realistic price range

print(f"✅ Generated {n_samples} house records using NumPy")
print(f"Features: bedrooms, bathrooms, sqft, age, location")

# STEP 2: CREATE PANDAS DATAFRAME
print("\n📋 STEP 2: Creating DataFrame with Pandas")
print("-" * 40)

# Create DataFrame
df = pd.DataFrame({
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'sqft': sqft.round(0).astype(int),
    'age': age,
    'location': location,
    'price': price.round(0).astype(int)
})

print("✅ DataFrame created successfully!")
print(f"Shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# STEP 3: EXPLORATORY DATA ANALYSIS WITH PANDAS
print("\n🔍 STEP 3: Exploratory Data Analysis")
print("-" * 40)

# Basic info
print("Dataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nLocation Distribution:")
print(df['location'].value_counts())

# STEP 4: DATA VISUALIZATION
print("\n📈 STEP 4: Data Visualization")
print("-" * 40)

plt.figure(figsize=(15, 10))

# Price distribution
plt.subplot(2, 3, 1)
plt.hist(df['price'], bins=30, edgecolor='black', alpha=0.7)
plt.title('Price Distribution')
plt.xlabel('Price (₹)')
plt.ylabel('Frequency')

# Price vs Square Feet
plt.subplot(2, 3, 2)
plt.scatter(df['sqft'], df['price'], alpha=0.5)
plt.title('Price vs Square Feet')
plt.xlabel('Square Feet')
plt.ylabel('Price (₹)')

# Price by Location
plt.subplot(2, 3, 3)
df.boxplot(column='price', by='location', ax=plt.gca())
plt.title('Price by Location')
plt.suptitle('')  # Remove default title

# Correlation heatmap (numeric columns only)
plt.subplot(2, 3, 4)
numeric_df = df.select_dtypes(include=[np.number])
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

# Price vs Bedrooms
plt.subplot(2, 3, 5)
df.groupby('bedrooms')['price'].mean().plot(kind='bar')
plt.title('Average Price by Bedrooms')
plt.xlabel('Bedrooms')
plt.ylabel('Average Price (₹)')
plt.xticks(rotation=0)

# Age vs Price
plt.subplot(2, 3, 6)
plt.scatter(df['age'], df['price'], alpha=0.5)
plt.title('Age vs Price')
plt.xlabel('Age (years)')
plt.ylabel('Price (₹)')

plt.tight_layout()
plt.show()

print("✅ Visualizations complete!")

# STEP 5: FEATURE ENGINEERING WITH PANDAS & NUMPY
print("\n🔧 STEP 5: Feature Engineering")
print("-" * 40)

# Create new features using pandas and numpy
df['price_per_sqft'] = df['price'] / df['sqft']
df['room_total'] = df['bedrooms'] + df['bathrooms']
df['is_new'] = (df['age'] < 5).astype(int)
df['sqft_log'] = np.log(df['sqft'])

# Create age categories
df['age_category'] = pd.cut(df['age'], 
                           bins=[0, 10, 25, float('inf')], 
                           labels=['New', 'Medium', 'Old'])

print("✅ New features created:")
print("- price_per_sqft: Price per square foot")
print("- room_total: Total bedrooms + bathrooms") 
print("- is_new: Binary indicator for houses < 5 years old")
print("- sqft_log: Log transformation of square feet")
print("- age_category: Categorical age groups")

print(f"\nUpdated DataFrame shape: {df.shape}")
print("\nNew features sample:")
print(df[['price_per_sqft', 'room_total', 'is_new', 'sqft_log', 'age_category']].head())

# STEP 6: DATA PREPROCESSING FOR SCIKIT-LEARN
print("\n⚙️ STEP 6: Data Preprocessing for Machine Learning")
print("-" * 40)

# Select features for modeling
feature_columns = ['bedrooms', 'bathrooms', 'sqft', 'age', 'room_total', 'is_new', 'sqft_log']
categorical_columns = ['location', 'age_category']

# Handle categorical variables using pandas get_dummies
df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns)

# Prepare feature matrix (X) and target vector (y)
X = df_encoded[feature_columns + [col for col in df_encoded.columns if any(cat in col for cat in categorical_columns)]]
y = df_encoded['price']

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Target vector shape: {y.shape}")
print(f"\nSelected features: {list(X.columns)}")

# STEP 7: TRAIN-TEST SPLIT USING SCIKIT-LEARN
print("\n🔄 STEP 7: Train-Test Split")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Testing set: {X_test.shape[0]} samples")

# STEP 8: FEATURE SCALING
print("\n📏 STEP 8: Feature Scaling")
print("-" * 40)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled using StandardScaler")
print(f"Mean of scaled training features: {np.mean(X_train_scaled, axis=0).round(3)}")
print(f"Std of scaled training features: {np.std(X_train_scaled, axis=0).round(3)}")

# STEP 9: MODEL TRAINING AND EVALUATION
print("\n🤖 STEP 9: Model Training and Evaluation")
print("-" * 40)

# Model 1: Linear Regression
print("Training Linear Regression...")
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
lr_predictions = lr_model.predict(X_test_scaled)

# Model 2: Random Forest (works with unscaled data)
print("Training Random Forest...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)  # Using unscaled data
rf_predictions = rf_model.predict(X_test)

# STEP 10: MODEL EVALUATION USING SCIKIT-LEARN METRICS
print("\n📊 STEP 10: Model Evaluation")
print("-" * 40)

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} Results:")
    print(f"Mean Absolute Error: ₹{mae:,.2f}")
    print(f"Root Mean Square Error: ₹{rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Evaluate both models
lr_metrics = evaluate_model(y_test, lr_predictions, "Linear Regression")
rf_metrics = evaluate_model(y_test, rf_predictions, "Random Forest")

# STEP 11: FEATURE IMPORTANCE ANALYSIS
print("\n🎯 STEP 11: Feature Importance Analysis")
print("-" * 40)

# Random Forest feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 Most Important Features (Random Forest):")
print(feature_importance.head(10))

# Visualize feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance.head(10)['feature'], feature_importance.head(10)['importance'])
plt.title('Top 10 Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# STEP 12: PREDICTION EXAMPLES
print("\n🔮 STEP 12: Making Predictions")
print("-" * 40)

# Create sample houses for prediction
sample_houses = pd.DataFrame({
    'bedrooms': [3, 4, 2],
    'bathrooms': [2, 3, 1],
    'sqft': [2000, 2500, 1200],
    'age': [5, 15, 30],
    'location': ['Suburbs', 'Downtown', 'Rural']
})

print("Sample houses for prediction:")
print(sample_houses)

# Process sample houses the same way as training data
sample_houses['room_total'] = sample_houses['bedrooms'] + sample_houses['bathrooms']
sample_houses['is_new'] = (sample_houses['age'] < 5).astype(int)
sample_houses['sqft_log'] = np.log(sample_houses['sqft'])
sample_houses['age_category'] = pd.cut(sample_houses['age'], 
                                      bins=[0, 10, 25, float('inf')], 
                                      labels=['New', 'Medium', 'Old'])

# Encode categorical variables
sample_encoded = pd.get_dummies(sample_houses, columns=['location', 'age_category'], 
                               prefix=['location', 'age_category'])

# Ensure all columns match training data
for col in X.columns:
    if col not in sample_encoded.columns:
        sample_encoded[col] = 0

sample_X = sample_encoded[X.columns]

# Make predictions
lr_sample_pred = lr_model.predict(scaler.transform(sample_X))
rf_sample_pred = rf_model.predict(sample_X)

print("\nPredicted Prices:")
for i in range(len(sample_houses)):
    print(f"House {i+1}:")
    print(f"  Linear Regression: ₹{lr_sample_pred[i]:,.0f}")
    print(f"  Random Forest: ₹{rf_sample_pred[i]:,.0f}")

# STEP 13: MODEL COMPARISON AND INSIGHTS
print("\n📈 STEP 13: Final Model Comparison & Insights")
print("-" * 40)

comparison_df = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest'],
    'MAE': [lr_metrics['MAE'], rf_metrics['MAE']],
    'RMSE': [lr_metrics['RMSE'], rf_metrics['RMSE']],
    'R²': [lr_metrics['R2'], rf_metrics['R2']]
})

print("Model Comparison:")
print(comparison_df.round(2))

best_model = 'Random Forest' if rf_metrics['R2'] > lr_metrics['R2'] else 'Linear Regression'
print(f"\n🏆 Best performing model: {best_model}")

print("\n💡 Key Insights:")
print("1. Square footage is typically the most important factor in price prediction")
print("2. Location significantly impacts house prices")
print("3. Random Forest often performs better than Linear Regression for this type of data")
print("4. Feature engineering (creating new features) can improve model performance")
print("5. Always evaluate multiple models and compare their performance")

print("\n🎓 LEARNING SUMMARY:")
print("✅ NumPy: Array operations, random data generation, mathematical functions")
print("✅ Pandas: Data manipulation, cleaning, feature engineering, categorical encoding") 
print("✅ Scikit-learn: Model training, evaluation, preprocessing, metrics")
print("✅ Complete ML pipeline: Data → EDA → Preprocessing → Training → Evaluation")

print("\n🚀 Next Steps for Further Learning:")
print("- Try different algorithms (SVM, XGBoost, Neural Networks)")
print("- Implement cross-validation")
print("- Add more advanced feature engineering")
print("- Explore hyperparameter tuning")
print("- Work with real datasets from Kaggle or UCI ML Repository")

print("\n" + "="*60)
print("🎯 SCRIPT COMPLETE! All plots are available in separate windows.")
print("   You can interact with the plots while exploring the results.")
print("="*60)

# Keep plots alive (optional - remove if not needed)
input("\nPress Enter to close all plots and exit...")
plt.close('all')
