# Core Packages
import streamlit as st
from PIL import Image

# Data Viz Packages
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

# EDA Packages
import pandas as pd
import numpy as np

st.set_page_config(page_title='GAD Analysis',page_icon = 'logo.png', layout = 'wide', initial_sidebar_state = 'auto')
sns.set(rc={'figure.figsize':(20,15)})

DATA_URL = ('gad.csv')

st.markdown('# Graduate Admission Dataset')
st.markdown('### **Analysis of Graduate Admission Dataset**')

img = Image.open('gad.png')
st.image(img, width = 720, caption = 'Graduate Admission Dataset')

st.markdown('### **About the Dataset:**')
st.info('This dataset was built \
    with the purpose of helping students in \
        shortlisting universities with their profiles. \
            The predicted output gives them a fair \
                idea about their chances for a particular university. \
                    This dataset is inspired by the UCLA Graduate Dataset from Kaggle. \
                        The graduate studies dataset is a dataset which describes the probability of \
                            selections for Indian students dependent on the following parameters below.')

img = Image.open('univ.png')
st.image(img, width = 720, caption = "Top 5 Universities in the US")

st.markdown('### **Dataset Info:**')
st.markdown('##### **Attributes of the Dataset:**')
st.info('\t 1. GRE Score (out of 340), \
        \n\t 2. TOEFL Score (out of 120), \
        \n\t 3. University Rating (out of 5), \
        \n\t 4. Statement of Purpose/ SOP (out of 5), \
        \n\t 5. Letter of Recommendation/ LOR (out of 5), \
        \n\t 6. Research Experience (either 0 or 1), \
        \n\t 7. CGPA (out of 10), \
        \n\t 8. Chance of Admittance (ranging from 0 to 1)') 

img = Image.open('par.png')
st.image(img, width = 720, caption = "Influence of the Attributes on the Dataset")

def load_data(nrows):
    df = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    df.set_index('Serial No.', inplace=True)
    df.rename(lowercase, axis='columns', inplace=True)
    return df

st.title('Lets explore the Graduate Admission Dataset')
# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading graduate admissions dataset...')
# Load 500 rows of data into the dataframe.
df = load_data(500)
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading graduate admissions dataset...Completed!')

# Explore Dataset
st.header('Quick  Explore')
st.sidebar.subheader('Quick  Explore')
st.markdown("Tick the box on the side panel to explore the dataset.")

if st.sidebar.checkbox("Show Raw Data"):
    st.subheader('Raw data')
    st.write(df)
if st.sidebar.checkbox('Dataset Quick Look'):
    st.subheader('Dataset Quick Look:')
    st.write(df.head())
if st.sidebar.checkbox("Show Columns"):
    st.subheader('Show Columns List')
    all_columns = df.columns.to_list()
    st.write(all_columns)
if st.sidebar.checkbox('Statistical Description'):
    st.subheader('Statistical Data Descripition')
    st.write(df.describe())
if st.sidebar.checkbox('Missing Values?'):
    st.subheader('Missing values')
    st.write(df.isnull().sum())

st.header('Create Own Visualization')
st.markdown("Tick the box on the side panel to create your own Visualization.")
st.sidebar.subheader('Create Own Visualization')
if st.sidebar.checkbox('Count Plot'):
    st.subheader('Count Plot')
    st.info("If error, please adjust column name on side panel.")
    column_count_plot = st.sidebar.selectbox("Choose a column to plot count.", df.columns[:5])
    fig = sns.countplot(x=column_count_plot,data=df)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
if st.sidebar.checkbox('Distribution Plot'):
    st.subheader('Distribution Plot')
    st.info("If error, please adjust column name on side panel.")
    column_dist_plot = st.sidebar.selectbox('Choose a column to plot density.', df.columns[:5])
    fig = sns.distplot(df[column_dist_plot])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

# Showing the Prediction Model
st.header('Building Prediction Model')
st.sidebar.subheader('Prediction Model')
st.markdown("Tick the box on the side panel to run Prediction Model.")

import pickle

if st.sidebar.checkbox('View Prediction Model'):
    st.subheader('Prediction Model')
    pickle_in = open('model.pkl', 'rb')
    model = pickle.load(pickle_in)

    @st.cache()

    # defining the function to predict the output
    def convert_toefl_to_ielts(val):
        if val > 69 and val < 94:
            score = 6.5
        if val > 93 and val < 102:
            score = 7.0
        if val > 101 and val < 110:
            score = 7.5
        if val > 109 and val < 115:
            score = 8.0
        if val > 114 and val < 118:
            score = 8.5
        if val > 117 and val < 121:
            score = 9.0
        return score

    def pred(gre, toefl, sop, lor, cgpa, resc):
        
        # Preprocessing user input
        ielts = convert_toefl_to_ielts(toefl)

        if resc == 'Yes':
            resc = 1
        else:
            resc = 0
        
        st.success("GRE Score = {} TOEFL Score = {} IELTS Score = {} CGPA = {} ".format(gre, toefl, ielts, cgpa))
        for univ in range(1, 6):
        # Predicting the output
            prediction = model.predict([[gre, toefl, ielts, univ, sop, lor, cgpa, resc]])
            
            st.info("Chance of Admittance for University Rank " + str(univ) + " = {0:.2f} %".format(prediction[0]*100))
            if prediction[0] >= 0.6667:
                st.success('Congratulations! You are eligible to apply for this university!')
            else:
                st.caption('Better Luck Next Time :)')

    # this is the main function in which we define our webpage  
    def main():       
        
        # following lines create boxes in which user can enter data required to make prediction 
        gre = st.slider("GRE Score (out of 340):", 0, 340, 0, step = 1)
        toefl = st.slider("TOEFL Score (out of 120):", 0, 120, 0, step = 1)
        sop = st.slider("SOP Score (out of 5):", value = 0.0, min_value = 0.0, max_value = 5.0, step = 0.5)
        lor = st.slider("LOR Score (out to 5):", value = 0.0, min_value = 0.0, max_value = 5.0, step = 0.5)
        resc = st.selectbox('Research Experience:', ("Yes", "No"))
        cgpa = st.number_input('Enter CGPA (out of 10):')
        
        # when 'Predict' is clicked, make the prediction and store it 
        if st.button("Predict"): 
            result = pred(gre, toefl, sop, lor, cgpa, resc)
        
    if __name__=='__main__': 
        main()

st.sidebar.subheader('Data Source')
st.sidebar.info("https://www.kaggle.com/graduate-admissions")
st.sidebar.subheader('Source Article')
st.sidebar.info("https://medium.com/analytics-vidhya/a-fresh-look-at-graduate-admissions-dataset-d39e4d20803e")
st.sidebar.subheader('Author Credits')
st.sidebar.info("[Sai Mihir Jakkaraju](https://github.com/saimihirj)\
    \n [Rushab Prakash Kulkarni](https://github.com/rushab14)")
st.sidebar.subheader('Built with Streamlit')
st.sidebar.info("https://www.streamlit.io/")