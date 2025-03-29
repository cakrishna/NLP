import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# load model
model_name = 'bert-classifier_tweets'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# # Load the tokenizer and the model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# @st.cache(allow_output_mutation=True)
# def load_model():
#     model = torch.load('fine_tuned_bert_model.pkl')
#     return model

# # Load the model
# model = load_model()

# Define the Streamlit app
map_dict = {0: 'The tweet has a negative sentiment.',
            1: 'Sentiment Is Neutral',
            2: 'The tweet has a positive sentiment.'
            }

st.title('Tweet Sentiment Analyzer')

tweet = st.text_area('Enter a tweet:', '')

if st.button('Predict'):
    # Preprocess the tweet
    inputs = tokenizer(tweet, return_tensors='pt')

    # Get model's prediction
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs)
    
    # Display the prediction
    if prediction == 0:
        st.write('The tweet has a negative sentiment.')
    elif prediction == 1:
        st.write('The tweet has a Netral sentiment.')
    else:
        st.write('The tweet has a positive sentiment.')

# Test text
# negative: hate the airline
# neutral: virginamerica dhepburn say
# positive: virginamerica plus add commercial experience tacky
    