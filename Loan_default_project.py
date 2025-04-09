#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv('loan_default_dataset_100k.csv')
df.head(5)


# In[6]:


df.info()


# In[3]:


# Check data types and missing values
df.info()
df.isnull().sum()

# Convert categorical columns using one-hot encoding or label encoding
categorical_cols = ['marital_status', 'education_level', 'loan_purpose']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Features and target
X = df.drop('defaulted', axis=1)
y = df['defaulted']


# In[4]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# In[7]:


import pickle

# Save the model
with open('loan_default_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save the columns used for prediction
with open('model_features.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)


# In[8]:


import pandas as pd
import numpy as np
import pickle
from IPython.display import display
import ipywidgets as widgets

# Load the model and columns
with open('loan_default_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('model_features.pkl', 'rb') as f:
    model_columns = pickle.load(f)

# Define user input widgets
age = widgets.IntSlider(value=35, min=18, max=70, description='Age:')
loan_amount = widgets.IntSlider(value=10000, min=500, max=100000, step=500, description='Loan:')
monthly_income = widgets.IntSlider(value=20000, min=0, max=200000, step=1000, description='Income:')
loan_term = widgets.IntSlider(value=6, min=1, max=36, step=1, description='Term (mo):')
repayment_score = widgets.FloatSlider(value=0.7, min=0, max=1, step=0.01, description='Repayment Score:')

# Categorical fields
marital_status = widgets.Dropdown(options=['Single', 'Married', 'Divorced', 'Widowed'], description='Marital:')
education = widgets.Dropdown(options=['Primary', 'Secondary', 'Tertiary'], description='Education:')
loan_purpose = widgets.Dropdown(options=['Emergency', 'Business', 'Medical', 'School Fees'], description='Purpose:')
past_defaults = widgets.ToggleButtons(options=[0, 1], description='Past Defaults:')

# Prediction function
def predict_default(*args):
    # Create a single-row dataframe with same structure as training data
    input_dict = {
        'loan_amount': loan_amount.value,
        'age': age.value,
        'monthly_income': monthly_income.value,
        'loan_term_months': loan_term.value,
        'repayment_history_score': repayment_score.value,
        'past_defaults': past_defaults.value,
        f'marital_status_{marital_status.value}': 1,
        f'education_level_{education.value}': 1,
        f'loan_purpose_{loan_purpose.value}': 1,
    }

    # Fill all columns, missing get 0
    input_data = pd.DataFrame([{col: input_dict.get(col, 0) for col in model_columns}])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    msg = "⚠️ Likely to Default!" if prediction == 1 else "✅ Likely to Repay"
    print(f"Prediction: {msg}")
    print(f"Default Probability: {probability:.2%}")

# Button to trigger prediction
predict_button = widgets.Button(description="Predict Default Risk", button_style='info')
predict_button.on_click(predict_default)

# Display widgets
display(age, loan_amount, monthly_income, loan_term, repayment_score,
        marital_status, education, loan_purpose, past_defaults,
        predict_button)


# In[ ]:




