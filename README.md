# Tweet Sentiment Analyser Pipeline
 This pipeline will input a tweet addressed to the local police department and will output if action is required or not required. The pipeline has a pre-processing function – stemming, lemmatization, removing stop words – a vectorizer function (used both Count Vectorizer and TF-IDF; Count Vectorizer yielded better accuracy) and four machine learning functions: Naïve Bayes, Random Forest, Logistic Regression, and SVM. The pipeline individually passes each tweet into the four algorithms and uses of wisdom of crowd logic to derive a final answer. 

 If you want to use this application, follow the steps outlined below:
 1. Download prerequisites by running pip install -r requirements3.txt
 2. run sentimentAnalyzer.py file
 3. Once you have the streamlit application open, input tweet in text box.
 4. 1 - action required; 0 - action not required

The pipeline was tested against a dataset of labelled tweets as seen in concatenated_file3.xlsx and had the following results:

Accuracy of Pipeline: 0.8095238095238095
F1  of pipeline: 0.8888888888888888
