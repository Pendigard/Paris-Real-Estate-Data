import json
import pandas as pd
import csv

# Load the csv data as a datafrme
data = pd.read_csv('../data/fr-esr-principaux-etablissements-enseignement-superieur.csv',delimiter=';')
data