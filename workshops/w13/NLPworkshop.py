import spacy
#import en_core_web_sm
#nlp = en_core_web_sm.load()
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def analyze_sentiment(sentence):
    # Tokenize
    doc = nlp(sentence)
    # Extract nouns
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    # Join the noun
    text = " ".join(noun_phrases)
    # Perform sentiment analysis
    analysis = TextBlob(text)
    # Determine polarity and subjectivity
    #polarity: goes from [-1,1] -1 is a strong negative sentiment, 0 neutral, 1 positive sentiment
    #subjectivity: from 0-1, 0 is an objective factual statement, 1 is subjective/opinionated
    sentiment_polarity = analysis.sentiment.polarity
    sentiment_subjectivity = analysis.sentiment.subjectivity

    return sentiment_polarity, sentiment_subjectivity

#test sentene
sentence = "Red is the worst color ever!"
#sentiment analysis
sentiment_polarity, sentiment_subjectivity = analyze_sentiment(sentence)

# Print the results
print("Sentence:", sentence)
print("Sentiment Polarity:", sentiment_polarity)
print("Sentiment Subjectivity:", sentiment_subjectivity)
