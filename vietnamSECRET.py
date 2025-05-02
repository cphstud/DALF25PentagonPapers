#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  2 09:02:36 2025

@author: victoriasanderramsing
"""
# Importer nødvendige pakker
import pandas as pd                 # Til datahåndtering i dataframes
import numpy as np                  # Til numeriske operationer (bruges ikke direkte her)
import statsmodels.api as sm        # Til statistisk analyse (bruges ikke direkte her)
import re

# Indlæs hele .txt-filen som dataframe 
dfText = pd.read_table("/Users/victoriasanderramsing/Documents/tmp/out.txt")

# Åbn filen manuelt i læsetilstand ("r" = read)
fh = open("/Users/victoriasanderramsing/Documents/tmp/out.txt", mode="r")

# Læs hele filens indhold som én lang tekststreng
content = fh.read()

# Split teksten i bidder hver gang ordet "SECRET" forekommer
contentlist = content.split("SECRET")

# Vis tekststykke nummer 2 (index 1) – altså teksten efter første "SECRET"
contentlist[1]

# Omdan listen af tekststykker til en pandas DataFrame med kolonnenavn "text"
contentDF = pd.DataFrame(data=contentlist, columns=["text"])

# FE = Feature Engineering: Tilføj kolonnen 'word_count' med antal ord i hver tekst
contentDF['word_count'] = contentDF['text'].apply(lambda x: len(x.split()))

# Funktion til at finde datoer i formatet:
# fx: 29 Feb 1968, 30 June 1968, 4 Mar 1968
def find_dates(text):
    pattern = r'\b\d{1,2} (?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|' \
              r'May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|' \
              r'Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{4}\b'
    return re.findall(pattern, text)

# ---------------------------------------------------
# 1. Læs hele teksten ind fra out.txt
with open("out.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 2. Split teksten linjevis (én linje = én potentiel begivenhed)
lines = content.splitlines()

# 4. Gem alle linjer som en dataframe
df = pd.DataFrame(lines, columns=["event"])

# 5. Funktion: del op i dato + beskrivelse, hvis muligt
pattern = r'\b([0-9]{1,2} [A-Za-z]+ 19[0-9]{2})\b'
contentDF['dato'] = contentDF['text'].str.extract(pattern, expand=False)
# Word_count viser fra første gang SECRET bliver vist til næste gang - derfor er nogle chunks større end andre 


# Vis de første fund
print(contentDF)
