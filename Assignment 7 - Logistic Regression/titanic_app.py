# Save the trained model to a file
with open('logistic_regression_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Create a Streamlit app
st.title('Titanic Survival Prediction App')

def user_input_features():
    pclass = st.sidebar.selectbox('Passenger Class', [1, 2, 3])
    sex = st.sidebar.selectbox('Sex', ['male', 'female'])
    age = st.number_input('Age', min_value=0, max_value=80)
    fare = st.number_input('Fare', min_value=0, max_value=500)
    sibsp = st.number_input('Siblings/Spouses Aboard', min_value=0, max_value=8)
    parch = st.number_input('Parents/Children Aboard', min_value=0, max_value=8)
    data = {'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'SibSp': [sibsp],
            'Parch': [parch]}
    features = pd.DataFrame(data, index=[0])
    return features

# Encode categorical variables
def encode_features(df):
    df_encoded = df.copy()
    df_encoded['Sex'] = df_encoded['Sex'].map({'male': 0, 'female': 1})
    return df_encoded

df = user_input_features()
encoded_df = encode_features(df)

# Ensure columns match the training data
expected_columns = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
for col in expected_columns:
    if col not in encoded_df.columns:
        encoded_df[col] = 0

# Reorder columns to match the training set
encoded_df = encoded_df[expected_columns]

# Predict
if st.button('Predict'):
    prediction = model.predict(encoded_df)
    if prediction[0] == 1:
        st.write('The passenger is predicted to survive.')
    else:
        st.write('The passenger is predicted not to survive.')
