# Add all of your comments here after the magic command

import pickle
import streamlit as st

pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

st.set_page_config(page_title='Graduate Admission Prediction', page_icon = 'logo.png', layout = 'wide', initial_sidebar_state = 'auto')

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
        if prediction[0] >= 0.70:
            st.success('Congratulations! You are eligible to apply for this university!')
        else:
            st.caption('Better Luck Next Time :)')

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:orange;padding:10px"> 
    <h1 style ="color:white;text-align:center;">Graduate Admissions Prediction Using Machine Learning</h1> 
    </div> 
    """
    # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 
      
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