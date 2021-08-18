import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


st.write("""
# Salary Prediction App

This app predicts the **Salary** of employee!

""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/shiv-2025/Salary_pred/blob/main/adult.csv?raw=true)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        workclass = st.sidebar.selectbox('workclass',('State-gov',
                  'Self-emp-not-inc','Private','Federal-gov','Local-gov',
                  'Self-emp-inc','Without-pay','Never-worked'))
        sex = st.sidebar.selectbox('Sex',('Male','Female'))
        occupation = st.sidebar.selectbox('occupation',('Adm-clerical', 'Exec-managerial',
                    'Handlers-cleaners', 'Prof-specialty', 'Other-service', 'Sales', 
                    'Craft-repair', 'Transport-moving', 'Farming-fishing', 'Machine-op-inspct',
                    'Tech-support', '?', 'Protective-serv', 'Armed-Forces', 'Priv-house-serv'))
        relationship = st.sidebar.selectbox('relationship', ('Not-in-family', 'Husband', 'Wife',
                                             'Own-child', 'Unmarried', 'Other-relative'))
        marital_status = st.sidebar.selectbox('marital_status',('Never-married',
            'Married-civ-spouse','Divorced','Married-spouse-absent','Separated',
            'Married-AF-spouse', 'Widowed'))
        race = st.sidebar.selectbox('race', ('White', 'Black', 'Asian-Pac-Islander', 
                                    'Amer-Indian-Eskimo', 'Other'))
        country = st.sidebar.selectbox('country', ('United-States', 'Cuba', 'Jamaica', 'India',
                   'Mexico', 'South', 'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany',
                   'Iran', 'Philippines', 'Italy', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
                   'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal', 'Dominican-Republic',
                   'El-Salvador', 'France', 'Guatemala', 'China', 'Japan', 'Yugoslavia',
                   'Peru', 'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
                   'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary', 
                   'Holand-Netherlands'))
        education_num = st.sidebar.slider('education_num (1 to 16)', 1,16,13)
        age = st.sidebar.slider('age', 0,100,24)
        fnlwgt = st.sidebar.slider('fnlwgt', 0,1000000,200)
        capital_gain = st.sidebar.slider('capital_gain', 0,100000,400)
        capital_loss = st.sidebar.slider('capital_loss', 0,100000,400)
        hours_per_week = st.sidebar.slider('hours_per_week', 0,100,40)
        
        data = {'age':age,
                'workclass': workclass,
                'fnlwgt': fnlwgt, 
                'education_num' : education_num,
                'marital_status' : marital_status,
                'occupation' : occupation,
                'relationship': relationship,
                'race': race,
                'sex': sex, 
                'capital_gain': capital_gain,
                'capital_loss': capital_loss,
                'hours_per_week': hours_per_week,
                'country': country}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
salary_raw = pd.read_csv('https://github.com/shiv-2025/Salary_pred/blob/main/adult.csv?raw=true')
salary = salary_raw.drop(columns=['salary', 'education'], axis=1)
salary.columns = ['age', 'workclass', 'fnlwgt', 'education_num', 'marital_status', 'occupation',
       'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
       'hours_per_week', 'country']

#extract categorical and numerical columns
categorical_feature = [feature for feature in salary.columns if salary[feature].dtypes == 'O']
numerical_feature = [feature for feature in salary.columns if salary[feature].dtypes != 'O']

#trim whitespaces from this df
salary[categorical_feature] = salary[categorical_feature].apply(lambda salary: salary.str.strip())

#replace '?' with None values to find out null values
salary = salary.replace({'?': None})
salary.fillna(salary.mode().iloc[0], inplace=True)

df = pd.concat([input_df,salary],axis=0)

#One hot encoding of categorical variables using get_dummies function
df = pd.concat([pd.get_dummies(df[categorical_feature], drop_first=True), df[numerical_feature]], axis=1)
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('salary_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

