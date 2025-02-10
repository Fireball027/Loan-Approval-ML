import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load and Explore Data
loans = pd.read_csv('loan_data.csv')

print(loans.info())
print(loans.describe())
print(loans.head())


# Exploratory Data Analysis (EDA)
# Histogram of FICO distributions for credit policy
plt.figure(figsize=(10, 6))
sns.histplot(loans[loans['credit.policy'] == 1]['fico'], bins=35, color='blue', label='Credit Policy = 1', alpha=0.6)
sns.histplot(loans[loans['credit.policy'] == 0]['fico'], bins=35, color='red', label='Credit Policy = 0', alpha=0.6)
plt.legend()
plt.xlabel('FICO')
plt.title('FICO Distribution by Credit Policy')
plt.show()

# Histogram of FICO distributions for not fully paid
plt.figure(figsize=(10, 6))
sns.histplot(loans[loans['not.fully.paid'] == 1]['fico'], bins=35, color='blue', label='Not Fully Paid = 1', alpha=0.6)
sns.histplot(loans[loans['not.fully.paid'] == 0]['fico'], bins=35, color='red', label='Not Fully Paid = 0', alpha=0.6)
plt.legend()
plt.xlabel('FICO')
plt.title('FICO Distribution by Loan Repayment Status')
plt.show()

# Countplot of the counts of loans by purpose
plt.figure(figsize=(11, 7))
sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')
plt.xticks(rotation=30)
plt.title('Loan Purpose vs. Loan Repayment Status')
plt.show()

# Jointplot of the trend between FICO scores and interest rate
sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')
plt.show()

# Lmplots: Interest Rate vs. FICO Score
sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')
plt.show()


# Categorical Features
# Preparing Data for Machine Learning
# Convert categorical feature 'purpose' into dummy variables
final_data = pd.get_dummies(loans, columns=['purpose'], drop_first=True)
print(final_data.info())


# Train-Test Split
# Split data into a training set and testing set
X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# Training a Decision Tree Model
# Create and train the model
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)


# Predictions And Evaluations Of Decision Tree
# Make predictions
dtree_predictions = dtree.predict(X_test)

print("\nDecision Tree Model Performance:")
print(classification_report(y_test, dtree_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, dtree_predictions))


# Training a Random Forest Model
# Create and train the model
rfc = RandomForestClassifier(n_estimators=100, random_state=101)
rfc.fit(X_train, y_train)


# Predictions And Evaluations
# Make predictions
rfc_predictions = rfc.predict(X_test)

print("\nRandom Forest Model Performance:")
print(classification_report(y_test, rfc_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rfc_predictions))
