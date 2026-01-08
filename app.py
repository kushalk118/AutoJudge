import streamlit as st
import pickle
import gzip

# Load saved models
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("classifier.pkl", "rb") as f:
    classifier = pickle.load(f)

with gzip.open("regressor.pkl.gz", "rb") as f:
    regressor = pickle.load(f)

# App title
st.title("AutoJudge: Programming Problem Difficulty Predictor")

st.write("Paste the programming problem details below:")

# Input text boxes
title = st.text_area("Problem Title")
description = st.text_area("Problem Description")
input_desc = st.text_area("Input Description")
output_desc = st.text_area("Output Description")
sample_io = st.text_area("Sample Input / Output")

# Combine text
combined_text = (
    title + " " +
    description + " " +
    input_desc + " " +
    output_desc + " " +
    sample_io
)

# Predict button
if st.button("Predict Difficulty"):
    if combined_text.strip() == "":
        st.warning("Please enter problem details!")
    else:
        X = vectorizer.transform([combined_text])

        predicted_class = classifier.predict(X)[0]
        predicted_score = regressor.predict(X)[0]

        st.success(f"Predicted Difficulty Class: {predicted_class}")
        st.success(f"Predicted Difficulty Score: {predicted_score:.2f}")

