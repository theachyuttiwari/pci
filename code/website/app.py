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
    spt = st.slider('Standard Penetration Test', 0.0, 65.5, 11.3)
    ag = st.slider('peak horizontal acceleration at ground surface', 0.1, 0.8, 0.2)
    pfine = st.slider('Fines content', 51, 99, 26)
    d50 = st.slider('Median grain size', 0.0, 1.6, 0.1)
    uwgwt = st.slider('unit weight below ground water table', 90, 135, 120)
    wt = st.slider('Depth of ground water table', 0.1, 3.0, 0.3)
    wd = st.slider('Depth from ground surface', 0.0, 25.0, 3.0)
    vts = st.slider('Vertical Total Stress', 370, 5708, 2006)

    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'mag': [mag],
            'spt': [spt],
            'ag': [ag],
            'pfine': [pfine],
            'd50': [d50],
            'uwgwt': [uwgwt],
            'wt': [wt],
            'wd': [wd],
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
