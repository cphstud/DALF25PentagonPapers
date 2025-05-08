#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:48:51 2025

@author: kasperdahl1
"""

import os
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from sentida import Sentida
import spacy
from collections import Counter
import nltk
import seaborn as sns
import glob
import numpy as np
import re
import statsmodels.api as sm
import matplotlib.pyplot as plt

nlp = spacy.load("en_core_web_sm")

folder = "/Users/kasperdahl 1/Documents/CPH_Business/DataAnalyse_2024-2026/Valgfag_NLP/projekt/DALF25PentagonPapers/tmp"

data = []
filnavne = []

for fil in os.listdir(folder):
    if fil.endswith(".txt"):
        with open(os.path.join(folder, fil), 'r', encoding='utf-8', errors='ignore') as f:
            data.append(f.read())
            filnavne.append(fil)

df = pd.DataFrame({'filename': filnavne, 'text': data})

# funktion
def giveNoun(text):
    doc=nlp(text)
    tempvar = None
    mylist = [token for token in doc if token.tag_ == "NN"]
    mylistCl=[str(item).strip() for item in mylist] 
    if len(mylist) > 0: 
        uk=Counter(mylistCl).most_common(1)
        tmpvar=uk[0][0]
    else:
        tmpvar = tempvar 
    return tmpvar



# make text size
df['size'] = df['text'].apply(lambda x: len(x))

# make sentiment score on text
sa = SentimentIntensityAnalyzer()

sscore = df['text'].apply(lambda x: sa.polarity_scores(x))
df['Sentiment'] = sscore.apply(lambda x: x['compound'])

# extract noun from text
# et eksempel
testtext = df.at[5,'text']
mydoc = nlp(testtext)
mylist = [token for token in mydoc if token.tag_ == "NN"]
mylistCl=[str(item).strip() for item in mylist]
uk=Counter(mylistCl).most_common(1)
uk[0][0]

df['noun'] = None
df['noun']=df['text'].apply(lambda x: giveNoun(x))

giveNoun('Here in the house lives an')

# extract pdf name from filename
df['ppname'] = df['filename'].str.extract(r"^(.*?)\.", flags=re.IGNORECASE)
df['ppname'] = 'Pentagon-Papers-Part-V-B-3d'

# extract pagenumber from filename
df["Page"] = df["filename"].str.extract(r"(\d{1,3})", flags=re.IGNORECASE)

plt.figure(figsize=(10,5))
top_nouns = df['noun'].value_counts().head(10)
sns.barplot(x=top_nouns.index, y=top_nouns.values)
plt.xticks(rotation=45)
plt.title("Top 10 hyppigste navneord")
plt.ylabel("Antal forekomster")
plt.xlabel("Navneord")
plt.show()

