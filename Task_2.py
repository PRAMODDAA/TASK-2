#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Load the Dataset

# In[2]:


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
gender_submission_df = pd.read_csv('gender_submission.csv')


# # Checking the top rows of data

# ## Checking for train dataset

# In[3]:


train_df.head()


# ## Checking for test dataset

# In[8]:


test_df.head()


# ## Checking for gender_submission dataset

# In[9]:


gender_submission_df.head()


# # Checking data from the bottom

# ## Checking for train dataset

# In[5]:


train_df.tail()


# ## Checking for test dataset

# In[6]:


test_df.tail()


# ## Checking for gender_submission dataset

# In[7]:


gender_submission_df.tail()


# # Data Overview

# In[10]:


# Get a summary of the training data
train_df.info()

# Check for missing values in the training dataset
train_df.isnull().sum()

# Statistical summary of the training dataset
train_df.describe()


# # Data Cleaning

# In[11]:


# Fill missing 'Age' with the median age
train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

# Fill missing 'Embarked' with the most common value
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

# Drop the 'Cabin' column
train_df.drop(columns=['Cabin'], inplace=True)

# Verify that there are no more missing values
train_df.isnull().sum()


# # Convert Categorical Variables:

# In[12]:


# Convert 'Sex' into numerical values: 0 for female, 1 for male
train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})

# Convert 'Embarked' into numerical values: 0 for S, 1 for C, 2 for Q
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Display the first few rows to verify the changes
train_df.head()


# # Exploratory Data Analysis (EDA)

# ## 1. Survival Rate by Gender.

# In[13]:


# Plot the survival rate by gender
sns.barplot(x='Sex', y='Survived', data=train_df)
plt.title('Survival Rate by Gender')
plt.show()


# ## 2. Survival Rate by Passenger Class

# In[26]:


# Plot the survival rate by passenger class
sns.barplot(x='Pclass', y='Survived', data=train_df)
plt.title('Survival Rate by Passenger Class')
plt.show()


# ## 3. Age Distribution of Survived vs. Not Survived

# In[24]:


# Plot the age distribution of survived vs not survived
sns.histplot(train_df[train_df['Survived'] == 1]['Age'], bins=20, kde=False, label='Survived', color='green')
sns.histplot(train_df[train_df['Survived'] == 0]['Age'], bins=20, kde=False, label='Not Survived', color='red')
plt.legend()
plt.title('Age Distribution: Survived vs Not Survived')
plt.show()


# ## 4. Correlation Matrix

# In[18]:


# Identify non-numeric columns
non_numeric_cols = train_df.select_dtypes(exclude=[np.number]).columns
print(non_numeric_cols)


# In[19]:


# Drop non-numeric columns
train_df_numeric = train_df.drop(columns=non_numeric_cols)

# Calculate the correlation matrix
corr_matrix = train_df_numeric.corr()

# Plot the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# ### Handle Essential Non-Numeric Columns

# In[20]:


# Example: Convert 'Sex' and 'Embarked' if not already done
train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Drop the remaining non-numeric columns
train_df_numeric = train_df.drop(columns=['Name', 'Ticket'])  # Adjust this list based on your data

# Calculate the correlation matrix
corr_matrix = train_df_numeric.corr()

# Plot the correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # Summary and Insights

# In[22]:


# Summarize key insights from the EDA
print("Key Insights:")
print("1. Gender: Females had a higher survival rate compared to males.")
print("2. Passenger Class: Passengers in higher classes (1st class) had better survival rates.")
print("3. Age: Younger passengers, particularly children, had a higher chance of survival.")
print("4. Correlation: Passenger class and gender are strong predictors of survival.")


# In[ ]:




