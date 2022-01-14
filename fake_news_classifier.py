import pandas as pd
import nltk
import csv, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn import svm
import time
# This function encodes the given string into ascii characters(and ignores errors) 
# and returns the ascii decoded string

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('words')


def encode_Ascii(text):
    encoded_string = text.encode("ascii", "ignore")
    return encoded_string.decode()

# This function lemmatizes the given text()
def lem_dataFrame(text):
    tkn = nltk.tokenize.WhitespaceTokenizer()
    lmt = nltk.stem.WordNetLemmatizer()
    return [lmt.lemmatize(w) for w in tkn.tokenize(text)]

#This function finds all of the hashtags from the data
def get_hashtags(df):
    list_hashtags = []
    for tweet in df.values:
        for tag in re.findall(r"\B#\w*[a-zA-Z]+\w*", tweet):
            if tag.lower() not in list_hashtags:
                list_hashtags.append(tag.lower())
    return list_hashtags

words = set(nltk.corpus.words.words()) # a list of all english words

# This function removes all non-english text from a String list
def keep_english(word_list):
    english_words = []
    for w in word_list:
        if w.lower() in words and len(w) > 2: english_words.append(w)
    return english_words

def avg_word(sentence):
    words = sentence.split()
    if(len(words) == 0):
        return 0
    return (sum(len(word) for word in words)/len(words))


def data_clean(filename):
    tweet_dataset = pd.read_table(filename, sep='\t', quoting=csv.QUOTE_NONE) # read the csv file into a DataFrame object

    tweet_dataset.drop(['tweetId', 'userId', 'imageId(s)', 'timestamp', 'username'], axis = 1, inplace = True) # delete the useless columns
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].apply(str.lower)

    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].str.replace(r'\\n|\\r|\\', '', regex = True) # Remove end of line characters
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].str.replace(r'(https|http)?:(\s)*\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', regex = True) # Remove URLs (including edge cases)

    # print(tweet_dataset)

    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].apply(encode_Ascii) # Remove non-ascii characters
    
    
    hashtags = get_hashtags(tweet_dataset['tweetText']) # Stores all hashtags before the '#' is deleted
    words.union(hashtags) # add all hashtags as keywords(so they don't get deleted)
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].str.replace(r'(@[A-Za-z0–9_]+)|[^\w\s]', '', regex = True) # Remove all punctuation and twitter tags 
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].apply(lem_dataFrame) # lemmatize the words
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].apply(lambda tweet: [word for word in tweet if word not in nltk.corpus.stopwords.words('english')])
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].apply(keep_english)
    
    
    tweet_dataset['tweetText'] = tweet_dataset['tweetText'].apply(lambda tweet_words: ' '.join([word for word in tweet_words]))

    tweet_dataset['label'] = tweet_dataset['label'].apply(lambda label: label if label != 'humor' else 'fake')
    
    return tweet_dataset

starttrain = time.time()

training_set = data_clean("mediaeval-2015-trainingset.txt")
# training_set.info();

indexNames = training_set[ training_set['tweetText'] == "" ].index
training_set.drop(indexNames, inplace=True)

# print(training_set)  

# Good setting - min_df=0.0001, max_df=0.1


tf_idf = TfidfVectorizer(min_df=0.0001, max_df=0.1, ngram_range=(1,2),max_features=2000)
count_vector = tf_idf.fit_transform(training_set.tweetText)
# print(count_vector.shape) 

# MultinomialNB()
# svm.SVC()
# RandomForestClassifier(n_estimators =10,criterion="entropy",random_state =0)
# KNeighborsClassifier(neighbors=10)
# The above algorithms need to be imported from the respective sklearn libraries
clf = svm.SVC()

clf.fit(count_vector.toarray(), list(training_set['label']))

endtrain = time.time()

starttest = time.time()

test_dataset = data_clean("mediaeval-2015-testset.txt")

test_vector = tf_idf.transform(test_dataset.tweetText)
print(test_vector.shape)

prediction = clf.predict(test_vector.toarray())

endtest = time.time()

cm = confusion_matrix(list(test_dataset.label), prediction)
   
print("F1 accuracy:")
print(f1_score(list(test_dataset['label']),prediction,average = 'micro'))

print("Confusion Matrix:")
print(cm)

print("Training: ",endtrain - starttrain) # prints total calculation time of the whole algorithm
print("Testing: ",endtest - starttest) # prints total calculation time of the whole algorithm