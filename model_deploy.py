import streamlit as st
import pickle

# loading the trained model
pickle_in = open('classifier.pkl', 'rb') 
classifier = pickle.load(pickle_in)

# defining the function which will make the prediction using the data which the user inputs 
def prediction(self_employed,income_annum,loan_amount,loan_term,cibil_score):     
    if self_employed == "no":
        self_employed = 0
    else:
        self_employed = 1
        
    loan_amount = loan_amount / 1000

    # Making predictions 
    prediction = classifier.predict([[self_employed,income_annum,loan_amount,loan_term,cibil_score]])

    if prediction == 0:
        prediction = 'Rejected'
    else:
        prediction = 'Approved'
        
    print(prediction)
    return prediction


#this is the main function in which we define our webpage  
def main():       
    #front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Loan Approval Prediction ML App </h1> 
    </div> 
    """

    #display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

    #following lines create boxes in which user can enter data required to make prediction 
    self_employed = st.selectbox('self_employed',("yes","no"))
    income_annum = st.number_input("Applicants monthly income") 
    loan_amount = st.number_input("Total loan amount")
    loan_term = st.number_input("Total loan_term")
    cibil_score = st.number_input("Enter cibil_score")
    result =""
    #when 'Predict' is clicked, make the prediction and store it 
    if st.button("Predict"): 
        result = prediction(self_employed,income_annum,loan_amount,loan_term,cibil_score) 
    st.success('Your loan is {}'.format(result))
    print(loan_amount)

if __name__=='__main__': 
    main()