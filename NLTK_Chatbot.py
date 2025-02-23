
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages
nltk.download('punkt') # first-time use only
nltk.download('wordnet') # first-time use only

nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

f=open('chatbot.txt','r',errors = 'ignore') #For our example,we will be using the Wikipedia page for chatbots as our corpus. Copy the contents from the page and place it in a text file named ‘chatbot.txt’. However, you can use any corpus of your choice.
raw=f.read()
raw = raw.lower()# converts to lowercase




"""
The main issue with text data is that it is all in text format (strings). However, the Machine learning algorithms need some sort of numerical feature vector in order to perform the task. So before we start with any NLP project we need to pre-process it to make it ideal for working. Basic text pre-processing includes:

    Converting the entire text into uppercase or lowercase, so that the algorithm does not treat the same words in different cases as different

    Tokenization: Tokenization is just the term used to describe the process of converting the normal text strings into a list of tokens i.e words that we actually want. Sentence tokenizer can be used to find the list of sentences and Word tokenizer can be used to find the list of words in strings
"""




sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words




#We shall now define a function called LemTokens which will take as input the tokens and return normalized tokens.
lemmer = nltk.stem.WordNetLemmatizer()
#WordNet is a semantically-oriented dictionary of English included in NLTK.
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))





#Next, we shall define a function for a greeting by the bot i.e if a user’s input is a greeting, the bot shall return a greeting response.ELIZA uses a simple keyword matching for greetings. We will utilize the same concept here.
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey", "whats up", "what's up Joeyy", "what's up joeyy", "hey")
GREETING_RESPONSES = ["movie", "hey baby", "*nods*, movie", "hey haha", "hello(;", "its joeyy, movie"]
def greeting(sentence):
 
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)




#To generate a response from our bot for input questions, the concept of document similarity will be used. We define a function response which searches the user’s utterance for one or 
#more known keywords and returns one of several possible responses. If it doesn’t find the input matching any of the keywords, it returns a response:” I am sorry! I don’t understand you”
def response(user_response):
    joeyy_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        joeyy_response=joeyy_response+"I am sorry! I am retarded still"
        return joeyy_response
    else:
        joeyy_response = joeyy_response+sent_tokens[idx]
        return joeyy_response


#we will feed the lines that we want our bot to say while starting and ending a conversation depending upon user’s input
flag=True
print("Joeyy: Hey haha my name is Joeyy, whats up, movie. If you dont want to talk anymore, type Bye!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Joeyy: You are welcome..")
        else:
            if(greeting(user_response)!=None):
                print("Joeyy: "+greeting(user_response))
            else:
                print("Joeyy: ",end="")
                print(response(user_response))
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Joeyy: Bye! take care..")