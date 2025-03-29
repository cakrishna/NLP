import nltk
def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_text(text):
    import re
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.strip()

def tokenize_text(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)