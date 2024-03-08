import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import toml

#from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
#from sklearn.linear_model import Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor 




config = toml.load("config.toml")
theme_config = config.get("theme", {})

# Set page config using theme configurations
st.set_page_config(
    page_title=theme_config.get("page_title", "Calories Burnt Prediction"),
    page_icon=theme_config.get("page_icon", ":chart_with_upwards_trend:"),
    initial_sidebar_state=theme_config.get("initial_sidebar_state", "expanded"),
)

# Apply theme configurations
st.markdown(
    f"""
    <style>
        body {{
            background-color: {theme_config.get("backgroundColor")};
        }}
        
        
        div.stApp {{
            background-color: {theme_config.get("backgroundColor")};
        }}
        
        .css-17eq0hr {{
            font-family: {theme_config.get("font", "sans-serif")};
        }}
        
        .stButton {{
            background-color: {theme_config.get("buttonColor")};
        }}
        
        .stSelectbox > div > div > div, .stTextInput > div > div > input {{
            background-color: {theme_config.get("buttonColor")};
            color: white !important;
        }}

        .stMultiSelect > div > div > div > div {{
            background-color: {theme_config.get("buttonColor")} !important;
        }}

        .stHighlighted {{
            background-color: {theme_config.get("highlightBackgroundColor", "#fdae61")} !important;
        }}

        /* Set sidebar background color */    
        [data-testid=stSidebar] {{
            background-color: #D7C0A2 !important;
            color: {theme_config.get("textColor", "#6e7074")};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)




st.sidebar.write("Use the widgets to alter the graphs:")


sns.set()

# Function for interactive gender distribution plot
def interactive_gender_distribution_plot(df, num_rows):
    categorical = df[['Gender']].head(num_rows)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.set(style="whitegrid")
    sns.countplot(x='Gender', data=categorical, palette="Set2")
    plt.title(f'Gender Distribution (First {num_rows} People)')
    plt.xlabel('Gender')
    plt.ylabel('Count')

    # Display the plot in Streamlit
    st.pyplot(fig)

# Set page title
st.title('Calories Burnt After Exercise Prediction')

# Introduction Section
st.header('Introduction')

st.subheader('Dataset Information')
st.text('The dataset used for this project is from Kaggle: https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos.')
st.text('The project aims to predict the calories burnt during workout sessions based on various features. The dataset used for this analysis comprises workout information for 15,000 individuals, including details such as age, gender, weight, height, and calories burnt.')

st.text('For this project, we are using data from the first 15 individuals in the dataset.')

st.subheader('Dataset Overview')
st.text('Below is detailed information on the first 15 individuals in the dataset who were part of this study. Names were omitted for anonymization, but details on age, gender, weight, height, calories burnt, and average heart rate and body temperature are provided.')

# Display tables
df1 = pd.read_csv("cedata/calories.csv")
df2 = pd.read_csv("cedata/exercise.csv")
df = pd.concat([df2, df1["Calories"]], axis=1)
st.table(df.head(15))

st.subheader('Overall distribution and statistical properties of the numerical features in the dataset')

st.table(df.describe())

# Gender Distribution Plot
st.subheader('Gender Distribution')
st.text('The gender distribution among the first 15 individuals is visualized interactively. A countplot is utilized to showcase the distribution, allowing us to explore the proportion of male and female participants.')

cat_col = [col for col in df.columns if df[col].dtype == 'O']  # Object type
categorical = df[cat_col]
categorical = pd.get_dummies(categorical["Gender"], drop_first=True)

st.text('Gender distribution among the first 15 individuals:')
interactive_gender_distribution_plot(df, 15)

# Distribution Plot for Numerical Columns
st.subheader('Distribution Plot for Numerical Columns')

st.text('This segment enables us to explore the distribution of numerical columns. A sidebar selector allows us to choose a specific numerical feature, and a distribution plot is generated to illustrate the spread of data.')

st.text('Use the sidebar to choose a numerical column.')

Num_col = df.select_dtypes(include=['float64', 'int64']).columns
Numerical = df[Num_col]

selected_column = st.sidebar.selectbox('Select Numerical Column', Numerical.columns)

fig, ax = plt.subplots(figsize=(20, 15))
sns.distplot(Numerical[selected_column], ax=ax)
plt.xlabel(selected_column, fontsize=15)
plt.title(f'Distribution of {selected_column}', fontsize=18)
st.pyplot(fig)

# Correlation Heatmap
st.subheader('Correlation Heatmap for Numerical Columns')
st.text('We can investigate the correlation between selected numerical columns using a heatmap. The interactive sidebar facilitates the selection of columns for correlation analysis, offering insights into potential relationships within the data.')
st.text('Please select from the sidebar if you see an error.')
selected_columns = st.sidebar.multiselect('Select One or More to see Correlation', Numerical.columns)
correlation_df = Numerical[selected_columns]

fig, ax = plt.subplots(figsize=(10, 10))
heatmap = sns.heatmap(correlation_df.corr(), cmap='viridis', annot=True)
plt.title('Correlation Heatmap')
st.pyplot(fig)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Combined Categorical and Numerical Distribution Plot
st.subheader('Combined Categorical and Numerical Distribution Plot')
st.text("This section combines categorical and numerical variables for a holistic view of data distribution. Subplots display distributions for each variable, providing a comprehensive understanding of the dataset's characteristics.")

data = pd.concat([categorical, Numerical], axis=1)

plt.figure(figsize=(20, 15))
plotnumber = 1

for column in data:
    if plotnumber <= 8:
        ax = plt.subplot(3, 3, plotnumber)
        sns.distplot(data[column])
        plt.xlabel(column, fontsize=15)
    plotnumber += 1
st.pyplot(plt.show())

st.set_option('deprecation.showPyplotGlobalUse', False)

# Data Splitting Information
st.subheader('Data Splitting Information')
st.text('The dataset is split into training and testing sets to evaluate machine learning models effectively. Information about the shapes of the training and testing sets is presented, offering transparency into the model evaluation process.')


X = data.drop(columns=["Calories"], axis=1)
y = data["Calories"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=1)

st.text(f"Shape of X Train: {X_train.shape}")
st.text(f"Shape of X Test: {X_test.shape}")
st.text(f"Shape of y Train: {y_train.shape}")
st.text(f"Shape of y Test: {y_test.shape}")

def predict(ml_model, X_train, y_train, X_test, y_test):
    model = ml_model.fit(X_train, y_train)
    st.write('Score : {}'.format(model.score(X_train, y_train)))
    y_prediction = model.predict(X_test)
    st.write('Predictions are:\n {}'.format(y_prediction))
    st.write('\n')

    r2_score = metrics.r2_score(y_test, y_prediction)
    st.write('R2 Score: {}'.format(r2_score))

    st.write('MAE:', metrics.mean_absolute_error(y_test, y_prediction))
    st.write('MSE:', metrics.mean_squared_error(y_test, y_prediction))
    st.write('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_prediction)))

    # Plotting the distribution of the residuals
    st.write('Distribution of Residuals:')
    sns.distplot(y_test - y_prediction)
    st.pyplot()
    
    st.set_option('deprecation.showPyplotGlobalUse', False)

# Model Predictions and Evaluation
st.header('Model Predictions and Evaluation')

st.subheader('XGBoost Regression: The XGBoost regression model is applied to predict calories burnt. The model\'s score, predictions, and key evaluation metrics (R2 score, MAE, MSE, RMSE) are displayed. Additionally, a distribution plot of residuals provides insights into model performance.')


predict(XGBRegressor(), X_train, y_train, X_test, y_test)
st.set_option('deprecation.showPyplotGlobalUse', False)


st.subheader('Linear Regression: Similar to XGBoost, linear regression is employed, and its predictions and evaluation metrics are showcased. This allows us to compare the performance of different regression models.')

predict(LinearRegression(), X_train, y_train, X_test, y_test)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Decision Tree Regression:The decision tree regression model is utilized, and its predictions and evaluation metrics are presented. This provides a diverse set of regression algorithms for us to explore.')

predict(DecisionTreeRegressor(), X_train, y_train, X_test, y_test)
st.set_option('deprecation.showPyplotGlobalUse', False)

st.subheader('Random Forest Regression: The project concludes with the application of a random forest regression model. Predictions and evaluation metrics offer a comprehensive understanding of how different models perform in predicting calories burnt.')

predict(RandomForestRegressor(), X_train, y_train, X_test, y_test)
st.set_option('deprecation.showPyplotGlobalUse', False)


