BERT Model Sentiment Analysis

Project Overview

This repository contains scripts and resources for performing sentiment analysis on news articles referring to Russinan-Ukrainian 2022 War using a pre-trained BERT model. The goal is to classify the sentiment of each article as either Pro-Russian or Pro-Ukrainian and calculate a sentiment score.

The scripts were used for an MA thesis

Steps to use the model: 
1. Scraping Journal ariticles from site archives and create a large dataset (Python_Scraper.py script will do the job)
2. Create a training dataset (for my research I created the narratives.csv)
3. Create and train the model (BERT_Trainner.py is based no the narratives.csv as a training dataset)
4. Evavaluate that the model is perfomrming effectively (BERT_Model_Evaluator.py)
5. Assess the sentiment of the articles dataset (BERT_Article_Evaluator.py)

1. Step 1  Scraping Journal Ariticles
  Open Python_Scraper.py
  Change the URL address to the URL you want
  Change the name of the CSV to save the download  
  Running Python_Scraper.py will result downloading all the articles from the URL that contain the keywords "Russia", "Ukraine", "Zelensky", "Putin" in the Title. The articles will be stored to a CSV file that contains 3 Columns: Title,URL,Content

2. Step 2 Creation of Training Dataset
  Create a CSV that contain pro-Ukrainian and pro-Russian sentences.
  The CSV I used is named narratives.csv and contains 4 columns: text,label,narrative_category,message_type.

3. Step 3 Train the Model
  BERT_Trainner.py file is set to train a BERT model with the narratives.csv
  Running BERT_Trainner.py will result the creation of an AI model in a folder.
  The model is now ready to use

4. Step 4 Evaluate Efficiency
  Create a CSV that contain messages/articles that are pro-Ukrain or pro-Russian.
  The CSV I used is named Model_evaluator.csv and contains 2 simple sentences pro-Ukrainian and 2 pro-Russian
  Running the BERT_Model_Evaluator.py will output the sentiment assessment of the model of the sentences of the CSV. This will ensure that is working properly.
  The Model will give a score to every sentence. In a scale 0 - 10 score 0 means extremely pro-Russian score 5 means impartial and score 10 means extremely pro-Ukrainian.

 5. BERT_Article_Evaluator.py file is set to assess the sentiment of articles.
  Make sure the articles are structured with 3 Columns: Title,URL,Content
  Running the fill will result the creation of a CSV file that contains 3 Columns: Sentiment,Score,Month
  You can adjust the structure of the file accordingly.
