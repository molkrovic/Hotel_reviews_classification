import pandas as pd
import numpy as np
import unicodedata
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

df_raw = pd.read_csv('data/raw/Big_AHR.csv')

df_1_star = df_raw[df_raw['rating']==1]
df_2_stars = df_raw[df_raw['rating']==2]
df_4_stars = df_raw[df_raw['rating']==4]
df_5_stars = df_raw[df_raw['rating']==5]

# Eliminar columnas que no corresponden a texto de la reseña
def delete_columns(df):
    df = df.drop(['label', 'hotel', 'location', 'rating', 'Unnamed: 0'], axis=1)
    return df

# Crear una columna que combine title y review
def combine_texts(df):
    df['text'] = df['title'].astype(str)+'. '+df['review_text'].astype(str)
    return df

# Crear lista con cada oración (separada por .) por separado
def create_sentences(df):
    lista_text = df['text'].tolist()
    sentences = []
    for line in lista_text:
        line = line.rstrip()
        oraciones = line.split('.')
        for oracion in oraciones:
            oracion = oracion.lstrip(' ')
            sentences.append(oracion)
    sentences = list(filter(None, sentences))
    return sentences

# Convertir la lista anterior a dataframe

# Limpieza del texto
def preprocess_column(df):
    df['text_processed'] = df['text'].str.strip().str.lower()
    caracteres = ['!', ',', '&', ':', ';', '(', ')', '.', '?', '"']
    for car in caracteres:
        df['text_processed'] = df['text_processed'].str.replace(car,'', regex=False)
    df['text_processed'] = df['text_processed'].str.normalize('NFKC')
    df['text_processed'] = df['text_processed'].str.replace(r'([a-zA-Z])\1{2,}', r'\1', regex=True) 
    return df

# Normalizar el texto (eliminar tildes, diéresis, etc)
def normalize_str(text_string):
    if text_string is not None:
        result = unicodedata.normalize('NFD', text_string).encode('ascii', 'ignore').decode()
    else:
        result = None
    return result

# Sustituye caracteres no alfanuméricos
def non_alphanumeric(texto):
    return re.sub("(\\W)+"," ", texto)

#Sustituye los espacios dobles entre palabras
def multiple_esp(texto):
    return re.sub(' +', ' ',texto)

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('spanish')

# No se quiere remover del texto las palabras que cambian el sentido de la expresión (por ejemplo "no")
for word in ['no', 'sin', 'nada']:
    stop_words.remove(word)

# Remover "hotel", que aparece en la mayoría de las reseñas y no aporta significativamente al análisis
stop_words.append('hotel')

# Normalizar stopwords
for i in range(len(stop_words)):
    stop_words[i] = normalize_str(stop_words[i])

# Función para eliminar las stopwords incluidas en la lista creada
def remove_stopwords(text_string):
    word_tokens = word_tokenize(text_string)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    sentence = ' '.join(str(x) for x in filtered_sentence)
    return sentence

def create_df_sentences(df):
    df = delete_columns(df)
    df = combine_texts(df)
    sentences = create_sentences(df)
    df_sentences = pd.DataFrame (sentences, columns = ['text'])
    df_sentences = preprocess_column(df_sentences)
    df_sentences['text_processed'] = df_sentences['text_processed'].apply(normalize_str)
    df_sentences['text_processed'] = df_sentences['text_processed'].apply(non_alphanumeric)
    df_sentences['text_processed'] = df_sentences['text_processed'].apply(multiple_esp)
    df_sentences['text_processed'] = df_sentences['text_processed'].apply(remove_stopwords)
    return df_sentences

df_sentences_1_star = create_df_sentences(df_1_star)
df_sentences_2_stars = create_df_sentences(df_2_stars)
df_sentences_4_stars = create_df_sentences(df_4_stars)
df_sentences_5_stars = create_df_sentences(df_5_stars)

# Guardar dataframe como csv
df_sentences_1_star.to_csv('data/interim/sentences_1_star.csv', index=False)
print('Se guardó sentences_1_star')
df_sentences_2_stars.to_csv('data/interim/sentences_2_stars.csv', index=False)
print('Se guardó sentences_2_stars')
df_sentences_4_stars.to_csv('data/interim/sentences_4_stars.csv', index=False)
print('Se guardó sentences_4_stars')
df_sentences_5_stars.to_csv('data/interim/sentences_5_stars.csv', index=False)
print('Se guardó sentences_5_stars')