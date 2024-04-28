import numpy as np
import pickle

loaded_model=pickle.load(open('G:/ckdprediction/CKD_AdaBoost_web/trained_model.sav','rb'))

# Assuming your original input_data with missing values
input_data = (7.0,50.0,1.02,4.0,0.0,0,0,0,18.0,0.8,11.3,38,6000,0,0,0,1,0)

# Change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Impute missing values with the mean of the non-missing values
input_data_as_numpy_array = np.nan_to_num(input_data_as_numpy_array, nan=np.nanmean(input_data_as_numpy_array))

input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person does not have CKD disease.")
else:
    print("The person has CKD disease.")
