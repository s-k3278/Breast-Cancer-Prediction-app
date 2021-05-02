import streamlit as st
import pickle
import numpy as np

model=pickle.load(open('classifier.pkl','rb'))
encoder=pickle.load(open('label_encoder.pkl','rb'))
scaler = pickle.load(open('StandardScaler.pkl','rb'))
pca = pickle.load(open('pca.pkl','rb'))

def predict_price(radius_mean, texture_mean, perimeter_mean,
       area_mean, smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, radius_se, perimeter_se,
       area_se, concave_points_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst):
    input=np.array([[radius_mean, texture_mean, perimeter_mean,
       area_mean, smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, radius_se, perimeter_se,
       area_se, concave_points_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst]]).astype(np.float64)

    input = scaler.transform(input)
    input = pca.transform(input)
    prediction=model.predict(input)
    prediction = encoder.inverse_transform(prediction)
    return prediction

def main():
    st.title("Breast Cancer Prediction")
    html_temp = """
    <div style="background-color:#000000 ;padding:5px">
    <h2 style="color:white;text-align:center;">Enter the Details Below </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    radius_mean = st.text_input("radius_mean")
    texture_mean = st.text_input("texture_mean")
    perimeter_mean = st.text_input("perimeter_mean")
    area_mean = st.text_input("area_mean")
    smoothness_mean = st.text_input("smoothness_mean")
    compactness_mean = st.text_input("compactness_mean")
    concavity_mean = st.text_input("concavity_mean")
    concave_points_mean = st.text_input("concave_points_mean")
    symmetry_mean = st.text_input("symmetry_mean")
    radius_se = st.text_input("radius_se")
    perimeter_se = st.text_input("perimeter_se")
    area_se = st.text_input("area_se")
    concave_points_se = st.text_input("concave_points_se")
    radius_worst = st.text_input("radius_worst")
    texture_worst = st.text_input("texture_worst")
    perimeter_worst = st.text_input("perimeter_worst")
    area_worst = st.text_input("area_worst")
    smoothness_worst = st.text_input("smoothness_worst")
    compactness_worst = st.text_input("compactness_worst")
    concavity_worst = st.text_input("concavity_worst")
    concave_points_worst = st.text_input("concave_points_worst")
    symmetry_worst = st.text_input("symmetry_worst")
    fractal_dimension_worst = st.text_input("fractal_dimension_worst")

    if st.button("Predict"):
        output=predict_price(radius_mean, texture_mean, perimeter_mean,
       area_mean, smoothness_mean, compactness_mean, concavity_mean,
       concave_points_mean, symmetry_mean, radius_se, perimeter_se,
       area_se, concave_points_se, radius_worst, texture_worst,
       perimeter_worst, area_worst, smoothness_worst,
       compactness_worst, concavity_worst, concave_points_worst,
       symmetry_worst, fractal_dimension_worst)
        st.success('Result is {} Diagnosis (M = malignant, B = benign)'.format(output))


if __name__=='__main__':
    main()