import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.svm import SVC

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
df = pd.read_excel('concatenated_file3.xlsx')
df = df.dropna(subset=['Action'])
df['tweets'] = df['tweets'].astype(str)

def preprocess_text(text, use_stemming=False, use_lemmatization=False, remove_stop_words = False):
    # Tokenization
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    if not isinstance(text, str):
        return ''

    # Stemming or Lemmatization
    if remove_stop_words:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

    if use_stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(token) for token in tokens]
    if use_lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    

    # Join tokens back into a string
    return ' '.join(tokens)


df['tweets'] = df['tweets'].apply(lambda x: preprocess_text(x,remove_stop_words=True))
#print(df.head)

vectorizer = CountVectorizer() #Changed from CountVectorizer
x = vectorizer.fit_transform(df['tweets'])
y = df['Action']

X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)



# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
#print(accuracy)

def SVMoutput(string):
    if pd.isna(string):
        string = 'NaN'
    string = preprocess_text(string,remove_stop_words=True)
    transformed_string = vectorizer.transform([string])

    return model.predict(transformed_string)[0]