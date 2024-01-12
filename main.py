import streamlit as st

title = st.title("Streamlit Tutorial")

st.write("""
         
         # Explore different classifiers
         Which one is the best ?
         """)

st.selectbox("Select Dataset", ("Iris", "Breast Cancer", "Wine Dataset", "MNIST"))

