import streamlit as st
import pandas as pd
import xgboost as xgb

model_filename = './model/best_xgb_model.json'

# Load the XGBoost model
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.7, min_child_weight=1)
model.load_model(model_filename)

def main():
    st.title('Soil Liquifaction Prediction')
    mag = st.slider('Moment Magnitude', 4.5, 8.5, 6.5)
    spt = st.slider('Standard Penetration Test', 0.0, 65.5, 11.3)
    ag = st.slider('Peak Horizontal Acceleration at Ground Surface', 0.1, 0.8, 0.2)
    pfine = st.slider('Fines Content', 51, 99, 26)
    uwgwt = st.slider('Unit Weight Below Ground Water Table', 90, 135, 120)
    wt = st.slider('Depth of Ground Water Table', 0.1, 3.0, 0.3)
    wd = st.slider('Depth from Ground Surface', 0.0, 25.0, 3.0)
    vts = st.slider('Vertical Total Stress', 370, 5708, 2006)

    if st.button('Predict'):
        user_input = pd.DataFrame(data={
            'mag': [mag],
            'spt': [spt],
            'ag': [ag],
            'pfine': [pfine],
            'uwgwt': [uwgwt],
            'wt': [wt],
            'wd': [wd],
            'vts': [vts]
        })

        # Display the shape and feature names of user_input
        st.write("Shape of user_input:", user_input.shape)
        st.write("Feature names:", user_input.columns)
        st.write("Data types:", user_input.dtypes)

        # Convert user_input to float if necessary
        user_input = user_input.astype(float)

        # Convert DataFrame to NumPy array if needed
        prediction = model.predict(user_input.values)
        prediction_proba = model.predict_proba(user_input.values)

        # Handling prediction results
        if prediction[0] == 1:
            bg_color = 'red'
            prediction_result = 'Positive'
        else:
            bg_color = 'green'
            prediction_result = 'Negative'
        
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]

        # Display the result with background color
        st.markdown(f"<p style='background-color:{bg_color}; color:white; padding:10px;'>Prediction: {prediction_result}<br>Confidence: {((confidence*10000)//1)/100}%</p>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
