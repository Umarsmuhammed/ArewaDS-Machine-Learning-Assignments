# Make a Bot talk back
## Instructions
In the past few lessons, you programmed a basic bot with whom to chat. This bot gives random answers until you say 'bye'. Can you make the answers a little less random, and trigger answers if you say specific things, like 'why' or 'how'? Think a bit how machine learning might make this type of work less manual as you extend your bot. You can use NLTK or TextBlob libraries to make your tasks easier.

Here is a possible text that follows your request:

To create a simple chatbot using Python and NLTK that can answer some basic questions based on predefined intents and responses. The chatbot uses a bag-of-words model to extract features from the user input and a naive Bayes classifier to predict the most likely intent. The chatbot then selects a random response from the list of responses associated with that intent. The chatbot can handle some common questions like 'what is your name?', 'how are you?', 'what can you do?', etc. The chatbot also has a default response for unknown intents. The chatbot stops when the user says 'bye'.

To make the chatbot more intelligent and less random, I could use a self-learning approach that uses machine learning and natural language processing techniques to learn from the user feedback and improve its responses over time. For example, I could use TextBlob to perform sentiment analysis on the user input and adjust the tone of the response accordingly. I could also use TextBlob to perform spelling correction, lemmatization, and part-of-speech tagging on the user input to improve the feature extraction process. I could also use a more advanced model like a neural network or a transformer to capture the semantic and contextual information from the user input and generate more natural and relevant responses. I could also use a database or an API to fetch dynamic information from external sources and provide more informative and useful answers to the user.
''' {python}
import random
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor
import nltk
nltk.download('punkt')
extractor = ConllExtractor()

def main():   
    print("Hello, I am Marvin, the friendly robot.")
    print("You can end this conversation at any time by typing 'bye'")    
    print("After typing each answer, press 'enter'")
    print("How are you today?")
    while True:
        # wait for the user to enter some text
        user_input = input("> ")

        if user_input.lower() == "bye":            
            # if they typed in 'bye' (or even BYE, ByE, byE etc.), break out of the loop
            break
        else:
            # Create a TextBlob based on the user input. Then extract the noun phrases
            user_input_blob = TextBlob(user_input, np_extractor=extractor)                        
            np = user_input_blob.noun_phrases                                    
            response = ""
            if user_input_blob.polarity <= -0.5:
                response = "Oh dear, that sounds bad. "
            elif user_input_blob.polarity <= 0:
                response = "Hmm, that's not great. "
            elif user_input_blob.polarity <= 0.5:
                response = "Well, that sounds positive. "
            elif user_input_blob.polarity <= 1:
                response = "Wow, that sounds great. "

            if len(np) != 0:
                # There was at least one noun phrase detected, so ask about that and pluralise it
                # e.g. cat -> cats or mouse -> mice
                response = response + "Can you tell me more about " + np[0].pluralize() + "?"
            else:
                response = response + "Can you tell me more?"
            print(response)
    
    print("It was nice talking to you, goodbye!")
main()
