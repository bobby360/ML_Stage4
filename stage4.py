"""
Stage4.py
"""

# import all libraries
import sqlite3
from html.parser import HTMLParser
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
import pandas as pd
#import numpy
#from pandasgui import show
# connect to database
CONN = sqlite3.connect("database.sqlite")



#"""### **TASK 1 and 2: filtering reviews; Produce a schema where count is greater than one.**"""
DF = pd.read_sql_query("""select UserId, ProductId,
                          ProfileName, Time,
                          Score, Text, COUNT(*)
                          FROM reviews
                          WHERE Score != 3
                          GROUP BY UserId
                          HAVING COUNT(*) > 1""", CONN)

DF.head()


#"""### **TASK 3: Sort product id in ascending order and deduplicate data.**"""
DF.sort_values('ProductId', inplace=True)
DF.drop_duplicates(subset="ProductId",
                   keep=False, inplace=True)

DF.head()

#"""### **TASK 4: How many positive and negative reviews are present in our dataset**"""

# Positive review scores: 4,5; Negative review scores 1,2
POS_REV = DF['Score'].value_counts()[4] + DF['Score'].value_counts()[5]
NEG_REV = DF['Score'].value_counts()[1] + DF['Score'].value_counts()[2]

print("There are " + str(POS_REV), 'positive reviews in the dataset')
print("There are " + str(NEG_REV), 'negative reviews in the dataset')

#"""### **TASK 5: Remove the html tags**"""


class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.fed = []
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s_strip = MLStripper()
    s_strip.feed(html)
    return s_strip.get_data()

DF['Text'] = DF['Text'].apply(strip_tags)

DF.head()

#"""### **TASK 6: Remove any punctuations or limited set of special characters**"""

def remove_symbols(text):
    """substitutes symbols"""
    return re.sub("[^A-z0-9\s]+", "", text)

DF['Text'] = DF['Text'].apply(remove_symbols)

DF.head()


#"""### **TASK 7: Check if the word is made up of English letters and is not alpha-numeric**"""

def letters_only(text):
    """substitutes alphanumeric"""
    return re.sub("[^a-zA-Z\s]", "", text)

DF['Text'] = DF['Text'].apply(letters_only)

DF.head()

#"""### **TASK 8: Check to see if the length of the word is greater than 2**"""

def check_length(text):
    text = text.split()
    contain = []

    for words in text:
        if len(words) > 2:
            contain.append(words)

    final = ' '.join(contain)
    return final

DF['Text'] = DF['Text'].apply(check_length)

DF.head()

#"""### **TASK 9: Convert the word to lowercase**"""

def lower_case(text):
    """forces lowercase"""
    return text.lower()

DF['Text'] = DF['Text'].apply(lower_case)

DF.head()


#"""### **TASK 10: Remove Stopwords**"""

def remove_stopwords(text):
    """to remove stop words"""
    try:
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]

    except:
        print("Downloading NLTK resources")
        nltk.download("stopwords")
        nltk.download("punkt")
        return remove_stopwords(text)

    return " ".join(filtered_sentence)

DF['Text'] = DF['Text'].apply(remove_stopwords)

DF.head()


#"""### **TASK 11: Finally Snowball Stemming the word**"""
def snow_stemmer(text):

    snowball_stemmer = SnowballStemmer('english')
    word_tokens = nltk.word_tokenize(text)
    stemmed_word = [snowball_stemmer.stem(word) for word in word_tokens]
    return " ".join(stemmed_word)

DF['Text'] = DF['Text'].apply(snow_stemmer)

DF.head()
