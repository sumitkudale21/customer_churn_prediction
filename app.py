import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Load your trained model
model = joblib.load('churn_predict_joblib')

# Load your dataset for EDA and drop the first three columns
data = pd.read_csv('dataset/Churn_Modelling.csv').iloc[:, 3:]  # Adjust the path to your dataset
d_data = pd.get_dummies(data, drop_first=True, dtype=int)  # Convert categorical variables to dummy variables

# Dummy metrics for example purposes
accuracy = 0.85
precision = 0.75
recall = 0.65
f1 = 0.70

# Dummy predictions for different models
y_test = np.random.randint(2, size=100)
y_pred = np.random.randint(2, size=100)
y_pred2 = np.random.randint(2, size=100)
y_pred3 = np.random.randint(2, size=100)
y_pred4 = np.random.randint(2, size=100)
y_pred5 = np.random.randint(2, size=100)
y_pred6 = np.random.randint(2, size=100)

# Metrics for multiple models
metrics = {
    'Model': ['LR', 'SVC', 'KNN', 'DT', 'RF', 'GBC'],
    'ACC': [
        accuracy_score(y_test, y_pred),
        accuracy_score(y_test, y_pred2),
        accuracy_score(y_test, y_pred3),
        accuracy_score(y_test, y_pred4),
        accuracy_score(y_test, y_pred5),
        accuracy_score(y_test, y_pred6)
    ],
    'Precision': [
        precision_score(y_test, y_pred),
        precision_score(y_test, y_pred2),
        precision_score(y_test, y_pred3),
        precision_score(y_test, y_pred4),
        precision_score(y_test, y_pred5),
        precision_score(y_test, y_pred6)
    ],
    'Recall': [
        recall_score(y_test, y_pred),
        recall_score(y_test, y_pred2),
        recall_score(y_test, y_pred3),
        recall_score(y_test, y_pred4),
        recall_score(y_test, y_pred5),
        recall_score(y_test, y_pred6)
    ],
    'F1-Score': [
        f1_score(y_test, y_pred),
        f1_score(y_test, y_pred2),
        f1_score(y_test, y_pred3),
        f1_score(y_test, y_pred4),
        f1_score(y_test, y_pred5),
        f1_score(y_test, y_pred6)
    ]
}

# Create the DataFrame
final_data = pd.DataFrame(metrics)

# Reshape the DataFrame to long format
final_data_long = pd.melt(final_data, id_vars=['Model'], var_name='Metric', value_name='Value')

def main_page():
    st.title("Customer Churn Prediction")

    html_temp = '''
    <div style="background-color:gray;padding:15px;border-radius:10px;">
    <h2 style="color:white;text-align:center">Customer Churn Prediction</h2>
    </div>
    '''
    
    st.markdown(html_temp, unsafe_allow_html=True)

    st.sidebar.header('Input Parameters')

    p1 = st.sidebar.number_input('Credit Score', min_value=300, max_value=850, value=600, step=1)
    p2 = st.sidebar.slider('Age', min_value=18, max_value=100, value=30, step=1)
    p3 = st.sidebar.slider('Tenure (years)', min_value=0, max_value=10, value=3, step=1)
    p4 = st.sidebar.number_input('Balance', min_value=0.0, max_value=250000.0, value=10000.0, step=1000.0)
    p5 = st.sidebar.slider('Number of Products', min_value=1, max_value=4, value=1, step=1)
    
    s1 = st.sidebar.selectbox('Do you have a Credit Card?', ('Yes', 'No'))
    p6 = 1 if s1 == 'Yes' else 0

    s2 = st.sidebar.selectbox('Are you an Active Member?', ('Yes', 'No'))
    p7 = 1 if s2 == 'Yes' else 0

    p8 = st.sidebar.number_input('Estimated Salary', min_value=0.0, max_value=200000.0, value=50000.0, step=5000.0)
    
    p9 = st.sidebar.selectbox('Geography', ['Germany', 'Spain', 'France'])
    p10 = st.sidebar.selectbox('Gender', ['Male', 'Female'])

    # Initialize dummy variables
    input_data = {
        'Geography_Germany': 0,
        'Geography_Spain': 0,
        'Gender_Male': 0
    }

    # Set the corresponding dummy variables
    if p9 == 'Germany':
        input_data['Geography_Germany'] = 1
    elif p9 == 'Spain':
        input_data['Geography_Spain'] = 1
    # If p9 is 'France', both Geography_Germany and Geography_Spain remain 0

    if p10 == 'Male':
        input_data['Gender_Male'] = 1
    # If p10 is 'Female', Gender_Male remains 0

    # Collect all input features
    feature_list = [
        p1,  # Credit Score
        p2,  # Age
        p3,  # Tenure
        p4,  # Balance
        p5,  # Number of Products
        p6,  # Has Credit Card
        p7,  # Is Active Member
        p8,  # Estimated Salary
        input_data['Geography_Germany'],  # Geography_Germany
        input_data['Geography_Spain'],    # Geography_Spain
        input_data['Gender_Male']         # Gender_Male
    ]

    # Convert the feature list to a numpy array
    input_vector = np.array(feature_list).reshape(1, -1)

    if st.button('Predict'):
        prediction = model.predict(input_vector)
        if prediction == 1:
            st.success('The prediction is: No Churn')
        else:
            st.error('The prediction is: Churn')

    st.write('---')
    st.write('This application predicts customer churn based on several input parameters. Adjust the inputs in the sidebar to see the prediction change.')

def metrics_page():
    st.title("Model Metrics")

    # Show metrics for multiple models
    st.write(f"**Accuracy**: {accuracy:.2f}")
    st.write(f"**Precision**: {precision:.2f}")
    st.write(f"**Recall**: {recall:.2f}")
    st.write(f"**F1 Score**: {f1:.2f}")

    # Create bar plots for model performance
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Model', y='Value', hue='Metric', data=final_data_long, palette='viridis', ax=ax)
    ax.set_title('Model Performance Metrics')
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    plt.legend(title='Metric')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)


def eda_page():
    st.title("Data Information and EDA")
    st.write("### Data Overview")
    st.write(data.head())

    st.write("### Data Description")
    st.write(data.describe())

    st.write("### Univariate Analysis")
    selected_column = st.selectbox('Select column for univariate analysis', data.columns)
    if data[selected_column].nunique() <= 10:
        # Use countplot for categorical variables with less than 10 unique values
        fig, ax = plt.subplots()
        sns.countplot(x=selected_column, data=data, ax=ax)
        ax.set_title(f'Countplot of {selected_column}')
        ax.set_xlabel(selected_column)
        ax.set_ylabel('Count')
        st.pyplot(fig)
    else:
        # Use histplot for numerical variables or those with more than 10 unique values
        fig, ax = plt.subplots()
        sns.histplot(data[selected_column], kde=True, ax=ax)
        ax.set_title(f'Histogram of {selected_column}')
        ax.set_xlabel(selected_column)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    st.write("### Bivariate Analysis")
    st.write("Select two columns for bivariate analysis")
    selected_columns = st.multiselect('Select two columns for bivariate analysis', data.columns, default=[data.columns[0], data.columns[1]])

    if len(selected_columns) == 2:
        # Check unique values to determine plot type
        if data[selected_columns[0]].nunique() > 10 and data[selected_columns[1]].nunique() > 10:
            # Scatter plot for features with more than 10 unique values
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[selected_columns[0]], y=data[selected_columns[1]], ax=ax)
            ax.set_title(f'{selected_columns[0]} vs {selected_columns[1]}')
            st.pyplot(fig)
        else:
            # Bar plot for features with less than or equal to 10 unique values
            fig, ax = plt.subplots()
            sns.barplot(x=selected_columns[0], y=selected_columns[1], data=data, ax=ax)
            ax.set_title(f'{selected_columns[0]} vs {selected_columns[1]}')
            st.pyplot(fig)
            
    st.title("Multivariate Analysis")

    st.write("### Multivariate Analysis")
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(d_data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Create a sidebar menu for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "Metrics", "EDA"])

# Display the selected page
if page == "Prediction":
    main_page()
elif page == "Metrics":
    metrics_page()
elif page == "EDA":
    eda_page()

