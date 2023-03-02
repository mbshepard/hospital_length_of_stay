import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# Load the data into a pandas DataFrame
data = pd.read_csv('train_data.csv')

# Map the 'Stay' variable to numeric values
stay_map = {'0-10': 5, '11-20': 15, '21-30': 25, '31-40': 35, '41-50': 45, '51-60': 55, '61-70': 65, '71-80': 75, '81-90': 85, '91-100': 95, 'More than 100 Days': 110}
data['Stay'] = data['Stay'].map(stay_map)

# Preprocess the categorical variables using one-hot encoding
cat_vars = ['Severity of Illness', 'Type of Admission', 'Department', 'Age']
encoder = OneHotEncoder()
encoded = encoder.fit_transform(data[cat_vars])
encoded_df = pd.DataFrame(encoded.toarray(), columns=encoder.get_feature_names_out(cat_vars))

# Combine the encoded variables with the numerical variables
numerical_vars = []
X = pd.concat([encoded_df, data[numerical_vars]], axis=1)

# Define the target variable
y = data['Stay']

# Fit a linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on new data
new_data = pd.DataFrame({'Age': ['31-40'], 'Severity of Illness': ['Moderate'], 'Type of Admission': ['Emergency'], 'Department': ['gynecology']})
new_encoded = encoder.transform(new_data[cat_vars])
new_encoded_df = pd.DataFrame(new_encoded.toarray(), columns=encoder.get_feature_names_out(cat_vars))
new_X = pd.concat([new_encoded_df, new_data[numerical_vars]], axis=1)
predictions = model.predict(new_X)
print(predictions)

# Define the Streamlit app
st.title('Length of Stay Prediction')
st.write('Enter the following information to get a predicted length of stay:')

# Define the input fields
age = st.selectbox('Age', data['Age'].unique())
severity = st.selectbox('Severity of Illness', data['Severity of Illness'].unique())
admission = st.selectbox('Type of Admission', data['Type of Admission'].unique())
department = st.selectbox('Department', data['Department'].unique())

# Encode the input variables using the same encoder as before
input_data = pd.DataFrame({'Age': [age], 'Severity of Illness': [severity], 'Type of Admission': [admission], 'Department': [department]})
input_encoded = encoder.transform(input_data[cat_vars])
input_encoded_df = pd.DataFrame(input_encoded.toarray(), columns=encoder.get_feature_names_out(cat_vars))
input_X = pd.concat([input_encoded_df, input_data[numerical_vars]], axis=1)

# Generate predictions using the linear regression model
prediction = model.predict(input_X)

# Display the predicted length of stay
st.write('Predicted length of stay:', prediction[0])