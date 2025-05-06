#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 13:02:15 2025

@author: victoriasanderramsing
"""

import os
import re
import requests
import fitz  # PyMuPDF
import pandas as pd
import spacy
from textblob import TextBlob
import glob

# -------------------------------------------------------
# TRIN 1: Find og download PDF-filer fra content.txt
# -------------------------------------------------------

# Læs indholdet af content.txt
with open("content.txt", "r", encoding="utf-8") as file:
    content_text = file.read()

# Find alle PDF-filer i content
expected_pdfs = set(re.findall(r"Pentagon-Papers-[\w\-]+\.pdf", content_text))

# Find alle allerede eksisterende .txt-filer i 'text/' mappen
txt_files = os.listdir("text")
txt_files = [f for f in txt_files if f.endswith(".txt")]

# Find PDF-filer der mangler .txt-udtræk
missing = []
for pdf in expected_pdfs:
    matching = [f for f in txt_files if f.startswith(pdf)]
    if not matching:
        missing.append(pdf)

print(f"Der mangler {len(missing)} .txt-filer:\n")
for m in sorted(missing):
    print(m)

# -------------------------------------------------------
# TRIN 2: Hent de manglende PDF-filer
# -------------------------------------------------------

# Lav mapping fra PDF-navn til URL
pdf_urls = re.findall(r"(https://[^\s]+\.pdf)", content_text)
url_map = {url.split("/")[-1]: url for url in pdf_urls}

# Download-mappe
os.makedirs("pdf_downloads", exist_ok=True)

for pdf in missing:
    if pdf in url_map:
        url = url_map[pdf]
        print(f"Henter {pdf} fra {url}")
        r = requests.get(url)
        with open(f"pdf_downloads/{pdf}", "wb") as f:
            f.write(r.content)
    else:
        print(f"Kunne ikke finde URL til {pdf}")

# -------------------------------------------------------
# TRIN 3: Konverter PDF'er til tekst (.txt)
# -------------------------------------------------------

pdf_folder = "pdf_downloads"
output_folder = "text"
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        txt_filename = filename + ".txt"
        txt_path = os.path.join(output_folder, txt_filename)

        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"Gemte: {txt_filename}")

# -------------------------------------------------------
# TRIN 4: Tjek om alle .txt-filer nu findes i mappe text
# -------------------------------------------------------

print("\nTjekker om .txt-filer er oprettet...")

for pdf in missing:
    expected_txt = pdf + ".txt"
    if expected_txt in os.listdir("text"):
        print(f"{expected_txt} er oprettet")
    else:
        print(f"{expected_txt} mangler stadig")

# -------------------------------------------------------------------------------
# TRIN 5: Analyse med spaCy og TextBlob på Part-IV-A-4 (15 filer - ikke nok ram)
# --------------------------------------------------------------------------------

# Indlæs spaCy engelsk model
nlp = spacy.load("en_core_web_sm")

# 1. Find alle .txt-filer i "text/" der matcher IV-A-4
folder = "text"
pattern = re.compile(r"Pentagon-Papers-Part-IV-A-4.*\.txt")
all_txt_files = [f for f in os.listdir(folder) if pattern.match(f)]

# 2. Begræns analysen til maks 15 filer (for at undgå MemoryError)
selected_txt_files = all_txt_files[:15]

# 3. Tom liste til at gemme resultater
data = []

# 4. Går gennem hver fil og udfører NLP-analyse
for file in selected_txt_files:
    filepath = os.path.join(folder, file)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Kører spaCy analyse
    doc = nlp(text)

    # Finder de mest hyppige substantiv (noun)
    nouns = [token.text.lower() for token in doc if token.pos_ == "NOUN" and token.is_alpha and len(token.text) > 2]
    noun_freq = pd.Series(nouns).value_counts()
    top_noun = noun_freq.index[0] if not noun_freq.empty else ""

    # Finder de mest hyppige adjektiv
    adjs = [token.text.lower() for token in doc if token.pos_ == "ADJ" and token.is_alpha and len(token.text) > 2]
    adj_freq = pd.Series(adjs).value_counts()
    top_adj = adj_freq.index[0] if not adj_freq.empty else ""

    # Sentiment analyse via TextBlob
    sentiment = TextBlob(text).sentiment.polarity

    # Mest almindelige entitetstype (f.eks. PERSON, ORG, GPE)
    entity_labels = [ent.label_ for ent in doc.ents]
    ent_freq = pd.Series(entity_labels).value_counts()
    top_entity = ent_freq.index[0] if not ent_freq.empty else ""

    # Prøv at udtrække sidetal fra filnavn (f.eks. -270.txt → 270)
    try:
        page = int(re.findall(r"-(\d+)\.txt", file)[0])
    except (IndexError, ValueError):
        page = -1  # fallback hvis sidetal ikke findes

    # Gemmer resultatet i en række
    data.append({
        "filename": file,
        "text": text.strip()[:1500],  # viser de første 1500 tegn
        "size": len(text),
        "sentiment": sentiment,
        "entity": top_entity,
        "noun": top_noun,
        "adj": top_adj,
        "page": page
    })

# -------------------------------------------------------
# TRIN 6: Byg og vis dataframe
# -------------------------------------------------------

# 5. Opretter DataFrame og vis
df = pd.DataFrame(data)
df = df.sort_values(by="page").reset_index(drop=True)

# 6. Gemmer som .pkl og print preview
df.to_pickle("IV_A4_analysis.pkl")

print(df.head(10))



