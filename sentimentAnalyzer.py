import streamlit as st


from combinedPipeline import returnActionStatus
from NaiveBayes import NaiveBayes
from RandomForest import RandomForestOutput
from LogisticRegressionFunc import logisticRegressionValue
from SVM import SVMoutput


st.title("Simple Sentiment Analysis Web App")


user_input = st.text_input("Enter a sentence:")


if st.button("Analyze Sentiment"):
    outputString =  returnActionStatus(user_input)
    NaiveBayesString = NaiveBayes(user_input)
    RandomForestOutputString = RandomForestOutput(user_input)
    logisticRegressionValueString = logisticRegressionValue(user_input)
    SVMString = SVMoutput(user_input)

    st.write("pipeline predicts: " , outputString)
    st.write("Naive Bayes predicts: ", NaiveBayesString)
    st.write("Random Forest predicts: ", RandomForestOutputString)
    st.write("Logistic Regression predicts :", logisticRegressionValueString)
    st.write("SVM predicts: ", SVMString)

