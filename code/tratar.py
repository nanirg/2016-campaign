# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:16:31 2018

@author: david
"""
 
import re   
from sklearn.datasets import load_files 
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#nltk.download('stopwords')    
from nltk.corpus import stopwords

movie_data = load_files(r"./../texts")  
X, y = movie_data.data, movie_data.target  

documents = []

from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
stemmer = WordNetLemmatizer()

for sen in range(0, len(X)):  
    
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(X[sen]))

    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 

    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)

    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
      # Removing unwanted characters
    document = re.sub(r'x9[0-9]', '', document)

    # Converting to Lowercase
    document = document.lower()

    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)

    documents.append(document)

tfidfconverter = TfidfVectorizer(max_features=100, min_df=1, max_df=0.9, stop_words=stopwords.words('english'))  
tfidfconverter.fit(documents)
X = tfidfconverter.transform(documents).toarray()

#X = tfidfconverter.fit_transform(documents).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)  
classifier.fit(X_train, y_train) 
y_pred = classifier.predict(X_test) 



print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  
print(accuracy_score(y_test, y_pred)) 


 
 

