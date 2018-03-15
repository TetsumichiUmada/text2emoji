#!/usr/bin/python

import pickle

"""
Converting data from pickle file to text file

usage: python script.py 

To save it to a txt file, run with 
usage: python script.py new_file.txt

"""

# Set the path to the pickle file 
file_name = "PsychExp/raw.pickle" 

# Open and read the pickle file 
data = pickle.load(open(file_name, "rb"))

### Read 
try:
	texts = [unicode(x).encode('utf-8') for x in data['texts']]
	labels = [x['label'] for x in data['info']]

except UnicodeDecodeError as e:
	texts = [x for x in data['texts']]
	labels = [x['label'] for x in data['info']]


for i in range(len(texts)):
	print labels[i], texts[i]

