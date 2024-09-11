#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup 
import requests as req 
import os
import csv


#Path to elog
base_path = "https://elog.hephy.at/testbeam-MA2024/"
output_file = "elog.csv"

#keys to extract from the elog
keys_to_extract = ["Run Number", "DUT", "Beam Energy", "Duration_min"]   

#ids to extract from elog
ids = np.arange(2,139)


#function to read in a single id and returns the values for the keys
def extract_run_parameters(id_number):    
    #modifying path for specific id an read in raw data
    output = []
    path = f'{base_path}'+f'/{id_number}' 
    entry = req.get(path)
    data = BeautifulSoup(entry.text, 'html.parser') 
    table_rows = data.find_all('input')

    #loop through raw data and extract "value=" for given keys
    for row in table_rows:
        row = str(row)
        for key in keys_to_extract:
            if re.search(key, row):
                print(key, row)
                value = re.findall(r'value=\"(.+)\"', row)
                print(value)
                output.append(value)           
    return output


#check for existing output file and remove it
if os.path.exists(output_file):
    os.remove(output_file)

#write csv file 
with open(output_file, 'a') as table: 
    writer = csv.writer(table, delimiter=',')
    writer.writerow(keys_to_extract)
    for i in ids:
        writer.writerow(extract_run_parameters(i))
    table.close()
    




