import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import warnings
warnings.filterwarnings('ignore')
import pickle
from sklearn.linear_model import LinearRegression
import streamlit as st

data = pd.read_csv('startUpData.csv')

#load the model
model = loaded_model = pickle.load(open('linearRegModel.sav', 'rb'))


# ---------------------- StreamLit Development Starts -----------------
st.markdown("<h1 style = 'color: #64CCC5; font-family: Arial, sans-serif; text-align: center;'>START UP BUSINESS PREDICTOR</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border-width: 2px; border-color: #FF0000;'>", unsafe_allow_html=True)

st.markdown("<h6 style = 'top-margin: 0rem; color: #EEEEEE'>BUILT BY Gomycode Yellow Orange Beast</h1>", unsafe_allow_html = True)

# st.title('START UP BUSINESS PREDICTOR')
# st.write('Built by Gomycode Yellow Orange Beast')
st.markdown("<br>", unsafe_allow_html= True)
username = st.text_input('Please Enter Username :')
if st.button('Submit Name'):
    if username:
        st.success(f'Welcome {username}, Please enjoy your usage')
    else:
        st.error("Please input your name")

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h2 style = 'top-margin: 0rem;text-align: center; color: #4F709C; text-decoration:underline'>Project Introduction</h1>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<p style = 'text-align: justify; color : #F0F0F0'>This project utilizes linear regression analysis to predict the potential profit of startup companies. By analyzing various factors such as investment, marketing expenditure, and industry trends, this model aims to provide valuable insights for entrepreneurs and investors, aiding in informed decision-making for new business ventures.</p>", unsafe_allow_html=True)

# heat_map = plt.figure(figsize = (14, 7))
# corr_data = data[['R&D Spend', 'Administration', 'Marketing Spend', 'Profit']]
# sns.heatmap(corr_data.corr(), annot = True, cmap='BuPu')
# st.write(heat_map)

st.write(data.sample(10).drop('Unnamed: 0', axis = 1).reset_index(drop = True))

# pic = st.camera_input("Take a picture")
with st.sidebar:
    st.image('pngwing.com (2).png', width = 200, caption=f"Welcome {username}", use_column_width=True)
    
    # if pic:
    #     st.image(pic, use_column_width=True, caption= f"Welcome {username}")
    
    st.write("Please decide your variable input")
    input_style = st.selectbox('Pick Your Preferred input',['','Slider Input','Number Input'],placeholder='choose one')
    
    if input_style == 'Slider Input':
        research = st.slider('R&D Spend', data['R&D Spend'].min(), data['R&D Spend'].max())
        admin = st.slider('Administration',data['Administration'].min(), data['Administration'].max())
        market = st.slider('Marketing Spend',data['Marketing Spend'].min(), data['Marketing Spend'].max())
        # profit = st.slider('Profit',data['Profit'].min(), data['Profit'].max())
        state = st.selectbox('choose state', [''] + list(data['State'].unique()))
    else:
        research = st.number_input('R&D Spend', data['R&D Spend'].min(), data['R&D Spend'].max())
        admin = st.number_input('Administration',data['Administration'].min(), data['Administration'].max())
        market = st.number_input('Marketing Spend',data['Marketing Spend'].min(), data['Marketing Spend'].max())
        # profit = st.number_input('Profit',data['Profit'].min(), data['Profit'].max())
        state = st.selectbox('choose state', [''] + list(data['State'].unique())) 
        

st.subheader("Your Inputted Data")
input_var = pd.DataFrame([{'R&D Spend': research, 'Administration' : admin, 'Marketing Spend' : market}])
st.write(input_var)

st.markdown("<br>", unsafe_allow_html= True)
tab1, tab2 = st.tabs(["Prediction Pane", "Intepretation Pane"])

with tab1:
    if st.button('PREDICT'):

        st.markdown("<br>", unsafe_allow_html= True)
        prediction = model.predict(input_var)
        st.write("Predicted Profit is :", prediction)
    else:
        st.write('Pls press the predict button for prediction')

with tab2:
    st.subheader('Model Interpretation')
    st.write(f"Profit = {model.intercept_.round(2)} + {model.coef_[0].round(2)} R&D Spend + {model.coef_[1].round(2)} Administration + {model.coef_[2].round(2)} Marketing Spend")

    st.markdown("<br>", unsafe_allow_html= True)

    st.markdown(f"- The expected Profit for a startup is {model.intercept_}")

    st.markdown(f"- For every additional 1 dollar spent on R&D Spend, the expected profit is expected to increase by ${model.coef_[0].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Administration Expense, the expected profit is expected to decrease by ${model.coef_[1].round(2)}  ")

    st.markdown(f"- For every additional 1 dollar spent on Marketting Expense, the expected profit is expected to increase by ${model.coef_[2].round(2)}  ")