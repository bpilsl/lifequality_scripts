#!/usr/bin/env python
# coding: utf-8

import numpy as np
import re
from bs4 import BeautifulSoup 
import requests as req 
import csv
import argparse

#Parsing Path, Start and end ID for extraction
parser = argparse.ArgumentParser()
parser.add_argument('-p','--base_path', type=str, help="Path to elog")
parser.add_argument('-o','--output_file', type=str, help="Path to output file (csv)")
parser.add_argument('-s','--border_start', type=int, help="Start id" )
parser.add_argument('-e','--border_end', type=int, help="End id" )
args = parser.parse_args()

#Path to elog for DESY TB 10.2024
#https://elog.hephy.at/testbeam-rd50-desy-Oct2024

#when opening the csv file: enter "text" as colum type for RunNumber for 3-digits

base_path = args.base_path
output_file = args.output_file

#keys to extract from the elog
# keys_to_extract = ["Run Number", "DUT", "thr", "HV", "scan-val", "Subject", "Duration_min", "Biasing"]   
keys_to_write = []
#ids to extract from elog
ids = np.arange(args.border_start,args.border_end)


#function to read in a single id and returns the values for the keys
def extract_run_parameters(id_number, first_call):    
    #modifying path for specific id and read in raw data
    output = []
    path = f'{base_path}'+f'/{id_number}' 
    entry = req.get(path)
    data = BeautifulSoup(entry.text, 'html.parser') 
    table_rows = data.find_all('input')

    #loop through raw data and extract "value=" for given keys
    # for key in keys_to_extract:
    try:
        for row in table_rows:
            row = str(row)
            match = re.search('input name="(.+?)"', row)
            if match:
                input_name = match.group(1)
                if first_call:                
                    keys_to_write.append(input_name)
                else:
                    value = re.findall(r'value=\"(.+)\"', row)      
                    if input_name == 'Run Number':
                        # print('before', value[0])
                        value[0] = value[0].zfill(3)
                        # print('after', value[0])

                        if (not value or len(value) == 0):
                            raise Exception(f'entry {id_number} without Run Number')

                    if not value:
                        output.append("nan")
                    else:
                        output.append(value[0])            
        
        output[0] =  str(output[0].zfill(3))
    except Exception as ex:
        print(ex)
        output = None
    return output

extract_run_parameters(1, True) #do this to populate available keys, really gross but whatever
keys_to_write.extend(["geometry", "outputDir"])

#write csv file 
with open(output_file, 'w') as table: 
    writer = csv.writer(table, delimiter=',')
    writer.writerow(keys_to_write)
    for i in ids:
        #ensure that only entries with run number are inserted
        try:
            run_params = extract_run_parameters(i, False)
            if run_params is not None and run_params[0]!="nan":
                writer.writerow(run_params)
            print('done with run', i)
        except Exception as ex:
            print(ex)
    table.close()
    




