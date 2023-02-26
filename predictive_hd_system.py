# -*- coding: utf-8 -*-
"""
Created on Sun Feb 26 10:54:56 2023

@author: HP
"""

import numpy as np
import pickle
import warnings

warnings.filterwarnings("ignore")
loaded_model = pickle.load(open('C:/Users/HP/Notebook files/trained_hd_model.sav','rb'))

input_data = (63,0,0,124,197,0,1,136,1,0.0,1,0,2)

input_data_as_numpy_Array = np.array(input_data)

input_reshaped = input_data_as_numpy_Array.reshape(1,-1)

prediction = loaded_model.predict(input_reshaped)
print(prediction)
