### **Hotel reviews classification**

_The objective is to analyze which aspects are mentioned as positive and which as negative in the reviews of Andalusian hotels._


**Content in src:**

explore.ipynb --> Initial exploration of the data.

create_df_sentences.py --> Preprocessing and cleaning the text. Division of the dataset according to the rating of the review. Divide each comment into sentences.

sentences_classification.ipynb --> Model to classify each sentence as positive or negative. Creation of datasets of positive and negative sentences.

negative_sentences.ipynb --> Clustering and analysis of sentences classified as negative.

positive_sentences.ipynb --> Clustering and analysis of sentences classified as positive.

*Data obtained from:*
https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced?select=Big_AHR.csv
