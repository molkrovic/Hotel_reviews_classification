import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df_raw = pd.read_csv('../data/raw/Big_AHR.csv')

df_1_star = df_raw[df_raw['rating']==1]
df_2_stars = df_raw[df_raw['rating']==2]
df_3_stars = df_raw[df_raw['rating']==3]
df_4_stars = df_raw[df_raw['rating']==4]
df_5_stars = df_raw[df_raw['rating']==5]

# Remove columns that do not correspond to review text
def delete_columns(df):
    df = df.drop(['label', 'hotel', 'location', 'rating', 'Unnamed: 0'], axis=1)
    return df

# Create a column that combines title and review
def combine_texts(df):
    df['text'] = df['title'].astype(str)+'. '+df['review_text'].astype(str)
    return df

# Create list with each sentence (separated by .) separately
def create_sentences(df):
    lista_text = df['text'].tolist()
    sentences = []
    review_n = []
    for i in range(len(lista_text)):
        line = lista_text[i]
        line = line.rstrip()
        oraciones = line.split('.')
        for oracion in oraciones:
            oracion = oracion.lstrip(' ')
            sentences.append(oracion)
            if len(oracion)>0:
                review_n.append(i)
    sentences = list(filter(None, sentences))
    columns = [sentences, review_n]
    return columns

# Text preprocessing
def preprocess_column(df):
    df['text_processed'] = df['text'].str.strip().str.lower()
    caracteres = ['¡', '!', ',', '&', ':', ';', '(', ')', '.', '¿' ,'?', '"', '$', '€']
    for car in caracteres:
        df['text_processed'] = df['text_processed'].str.replace(car,'', regex=False)
    df['text_processed'] = df['text_processed'].str.normalize('NFKC')
    df['text_processed'] = df['text_processed'].str.replace(r'([a-zA-Z])\1{2,}', r'\1', regex=True) 
    return df

# Normalize the text (remove accent marks, umlauts, etc.)
def normalize_str(text_string):
    if text_string is not None:
        result = unicodedata.normalize('NFD', text_string).encode('ascii', 'ignore').decode()
    else:
        result = None
    return result

# Replace non-alphanumeric characters
def non_alphanumeric(texto):
    return re.sub("(\\W)+"," ", texto)

# Replace double spaces between words
def multiple_esp(texto):
    return re.sub(' +', ' ',texto)

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('spanish')

# Words that change the meaning of the expression (for example "no") will not be removed from the text.
for word in ['no', 'sin', 'nada']:
    stop_words.remove(word)

# Remove "hotel", which appears in most reviews and does not contribute significantly to the analysis
stop_words.append('hotel')

# Normalize stopwords
for i in range(len(stop_words)):
    stop_words[i] = normalize_str(stop_words[i])

# Function to remove the stopwords included in the created list
def remove_stopwords(text_string):
    word_tokens = word_tokenize(text_string)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    sentence = ' '.join(str(x) for x in filtered_sentence)
    return sentence

def create_df_sentences(df):
    df = delete_columns(df)
    df = combine_texts(df)
    cols = create_sentences(df)
    sentences = cols[0]
    review_index = cols[1]
    df_sentences = pd.DataFrame({'text':sentences, 'review_index':review_index})
    df_sentences = preprocess_column(df_sentences)
    df_sentences['text_processed'] = df_sentences['text_processed'].apply(normalize_str).apply(non_alphanumeric).apply(multiple_esp).apply(remove_stopwords)
    df_sentences = df_sentences.dropna(subset=['text_processed'])
    return df_sentences

df_sentences_1_star = create_df_sentences(df_1_star)
df_sentences_2_stars = create_df_sentences(df_2_stars)
df_sentences_3_stars = create_df_sentences(df_3_stars)
df_sentences_4_stars = create_df_sentences(df_4_stars)
df_sentences_5_stars = create_df_sentences(df_5_stars)

# Save dataframes as csv
df_sentences_1_star.to_csv('../data/interim/sentences_1_star.csv', index=False)
print('sentences_1_star was saved')
df_sentences_2_stars.to_csv('../data/interim/sentences_2_stars.csv', index=False)
print('sentences_2_stars was saved')
df_sentences_3_stars.to_csv('../data/interim/sentences_3_stars.csv', index=False)
print('sentences_3_stars was saved')
df_sentences_4_stars.to_csv('../data/interim/sentences_4_stars.csv', index=False)
print('sentences_4_stars was saved')
df_sentences_5_stars.to_csv('../data/interim/sentences_5_stars.csv', index=False)
print('sentences_5_stars was saved')