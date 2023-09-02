# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 12:00:26 2023

@author: Hp
"""

import numpy as np
import pickle as pk
import streamlit as st

loaded_model=pk.load(open('C:\weight fish predict/trained_model.sav','rb'))

def fish_pred(input_data):
    input_data_array=np.asarray(input_data)
    input_data_array_new=input_data_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_array_new)
    return 'Expected Weight Of Fish Is {0}'.format(prediction)

def main():
    st.title('Fish Weight Predition Using Machine Learning')
    height=st.number_input('Enter Height Of Fish')
    width=st.number_input('Enter Width Of Fish')
    length1=st.number_input('Enter length1 Of Fish')
    length2=st.number_input('Enter length2 Of Fish')
    length3=st.number_input('Enter length3 Of Fish')
    st.write('Fish Species Encode Values As Species_Encode In DataFrame')
    from PIL import Image
    img=Image.open('C:\weight fish predict/species encode.png')
    st.image(img)
    species_encode=st.radio('Choose Species_Encode Value Of Fish Species From Above DataFrame',[0,1,2,3,4,5,6])
    fish_predict=' '
    if st.button('Click Here To Get Expected Fish Weight'):
        fish_predict=fish_pred([height,width,length1,length2,length3,species_encode])
    st.success(fish_predict)
    st.markdown('##### Exploratory Data Analysis Done And Machine Learning Model Deployed By "Anubhav Kumar Gupta"')
    
if __name__=='__main__':
    main()