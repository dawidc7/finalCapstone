import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import pandas as pd
from nltk.corpus import stopwords

# Loads language model
nlp = spacy.load("en_core_web_md")
# Adds spacytextblob extension to nlp pipeline to aid in sentiment analysis
nlp.add_pipe('spacytextblob')
# Loads stop words to prepare the text to be analyzed
cached_stop_words = stopwords.words("english")
df = pd.read_csv('amazon_product_reviews.csv') # Extracts reviews.text column and cleans the data by removing any missing values
review_data = df['reviews.text']
clean_data = review_data.dropna() # Data to use for multiple sentiment analysis

# Data to use for single sentiment analysis
sample_reviews = ["This device performs great! For the price it's really a steal. Affordable and easy to use.",
"This is your basic Amazon tablet. Nothing too special about it.",
"It is difficult to understand the instructions. I am still working on it."]

# Function for analyzing multiple reviews and counting up how many positive, neutral and negative reviews there are.
def multiple_sentiment_check(reviews):
    # Initializing the variables to count reviews
    positive_reviews = 0
    neutral_reviews = 0
    negative_reviews = 0

    for i in reviews:
        # Iterates over each word in the sentence and filters out any words that are present in the cached_stop_words list. 
        # This effectively removes stop words from the text.
        review = nlp(' '.join(word for word in i.split() if word not in cached_stop_words))
        # Checks the polarity of each review and determines whether it is positive, neutral or negative
        # After that, it increments the corresponding variable.
        polarity = review._.blob.polarity
        if polarity > 0.5:
            positive_reviews += 1
        elif polarity > -0.5:
            neutral_reviews += 1
        else:
            negative_reviews += 1

    print("Reviews count:")
    print(f"Positive Reviews - {positive_reviews}")
    print(f"Negative Reviews - {negative_reviews}")
    print(f"Neutral Reviews - {neutral_reviews}")

# Function for analyzing individual reviews, displaying whether it is positive, neutral or negative as well as the
# polarity score
def single_sentiment_check(reviews):
    for i in reviews:
        # Iterates over each word in the sentence and filters out any words that are present in the cached_stop_words list. 
        # This effectively removes stop words from the text.
        review = nlp(' '.join(word for word in i.split() if word not in cached_stop_words))
        # Checks the polarity of the review and determines whether it is positive, neutral or negative
        polarity = review._.blob.polarity
        if polarity > 0.5:
            print(f"This is a positive review. The polarity score is {polarity}")
        elif polarity > -0.5:
            print(f"This is a neutral review. The polarity score is {polarity}")
        else:
            print(f"This is a negative review. The polarity score is {polarity}")

def get_user_choice():
    user_choice = int(input("Please select from the below: \n 1. Analyse and summarise all reviews \n 2. Analyse the sample reviews for single sentiment analysis. \n Type your choice here: "))

    if user_choice == 1:
        multiple_sentiment_check(clean_data)
    elif user_choice == 2:
        single_sentiment_check(sample_reviews)
    else:
        print("Invalid choice, please try again.")
        get_user_choice()

get_user_choice()

