import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('G:/ckdprediction/CKD_AdaBoost_web/trained_model.sav', 'rb'))

# Creating function for prediction
def ckd_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data, dtype=np.float64)

    # Impute missing values with the mean of the non-missing values
    input_data_as_numpy_array = np.nan_to_num(input_data_as_numpy_array, nan=np.nanmean(input_data_as_numpy_array))

    # Reshape the numpy array to match the expected shape for the model prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Assuming you have already defined and trained your model
    prediction = loaded_model.predict(input_data_reshaped)

    if prediction[0] == 0:
        return "The person does not have CKD disease."
    else:
        return "The person has CKD disease."

def main():
    st.title("CKD prediction web app")

    age = st.text_input("Age")
    bp = st.text_input("BP")
    al = st.text_input("al")
    su = st.text_input("su")
    rbc = st.text_input("rbc")
    pc = st.text_input("pc")
    pcc = st.text_input("pcc")
    ba = st.text_input("ba")
    bgr = st.text_input("bgr")
    bu = st.text_input("bu")
    sc = st.text_input("sc")
    pot = st.text_input("pot")
    wc = st.text_input("wc")
    htn = st.text_input("htn")
    dm = st.text_input("dm")
    cad = st.text_input("cad")
    pe = st.text_input("pe")
    ane = st.text_input("ane")

    # Code for prediction
    diagnosis = ''

    # Creating button for prediction
    if st.button('CKD Test Results'):
        diagnosis = ckd_prediction([age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc, pot, wc, htn, dm, cad, pe, ane])

    st.success(diagnosis)

if __name__ == '__main__':
    main()
