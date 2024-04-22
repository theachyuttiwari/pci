import streamlit as st
import pandas as pd
import xgboost as xgb

model_filename = './model/best_xgb_model.json'

# Load the XGBoost model
model = xgb.XGBClassifier()
model.load_model(model_filename)

def main():
    st.title('Soil Liquifaction Prediction')
    mag = st.slider('Moment Magnitude', 4.5, 8.5, 6.5)
    #s0 = st.slider('S0', 67.4, 990.4, 262.9)
    #sp0 = st.slider('sp0',216.7, 3816.1, 1020.0)
    spt = st.slider('Standard Penetration Test', 0.0, 65.5, 11.3)
    ag = st.slider('peak horizontal acceleration at ground surface', 0.1, 0.8, 0.2)
    #tau_dinamic = st.slider('equivalent dynamic shear stress', 0.0, 0.5, 0.1)
    pfine = st.slider('Fines content', 51, 99, 26)
    d50 = st.slider('Median grain size', 0.0, 1.6, 0.1)
    #uwgwt = st.slider('unit weight above ground water table', 60, 125, 100)
    uwbgwt = st.slider('unit weight below ground water table', 90, 135, 120)
    wt = st.slider('Depth of ground water table', 0.1, 3.0, 0.3)
    wd = st.slider('Depth from ground surface', 0.0, 25.0, 3.0)
    vts = st.slider('Vertical Total Stress', 370, 5708, 2006)
    #ccb = st.slider('Correlation coefficient between σv′ and σv', 0.2, 1.0, 0.9)
    #smmc = st.slider('Shear mass modal participation factor', 0.4, 1.0, 0.8)
    #msf = st.slider('magnitude scaling factor', 0.1, 0.6, 0.3)
    #nonvalues = st.slider('Number of N values', 1, 19, 7)
    #ocf = st.slider('overburden correction factor', 0.7, 2.0, 1.4)
    #swvu = st.slider('swvu', 0, 950, 450)
    #cos = st.slider('correction for overburden stress', 0.8, 2.1, 1.2)
    #cmde = st.slider('correction for magnitude (duration) effects', 0.7, 1.7, 1.3)
    #csrmd = st.slider('CSR normalized to σ′v= 100 kPa, Mw=7.5', 0.0, 0.5, 0.2)
    #n1cs = st.slider('fines-corrected N1,60 value.', 5.0, 66.5, 14.1)

    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'mag': [mag],
            #'s0': [s0],  
            #'sp0': [sp0],
            'spt': [spt],
            'ag': [ag],
            #'tau_dinamic': [tau_dinamic],
            'pfine': [pfine],
            'd50': [d50],
            'uwgwt': [uwgwt],
            #'uwbgwt': [uwbgwt],
            'wt': [wt],
            'wd': [wd],
            #'msf': [msf],
            #'nonvalues': [nonvalues],
            #'ocf': [ocf],
            #'swvu': [swvu],
            #'cos': [cos],
            #'cmde': [cmde],
            #'csrmd': [csrmd],
            'vts': [vts]
        })
        prediction = model.predict(user_input)
        prediction_proba = model.predict_proba(user_input)

        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
        
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
