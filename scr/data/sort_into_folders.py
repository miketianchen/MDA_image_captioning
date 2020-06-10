#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import json


# In[2]:


json_data = {}
sizes = {}
set_names = ['test', 'train', 'valid']

# read in json files from all three datasets
for name in set_names:
    with open('data/processed/json/' + name + '.json', 'r') as data:
        json_data[name] = json.load(data)


# In[3]:


for set_name in set_names:
    # Make new directory to house the preprocessed images 
    os.mkdir('data/processed/' + set_name)
    
    for filename in json_data[set_name].keys():
        destination_path = 'data/processed/' + set_name + '/' + filename
        #origin_path = '../data/processed/' + 
        
        
        # Split the filename with `_` to find the dataset it originates from 
        dataset = filename.split('_')[0]
        
        origin_path = 'data/processed/' + 'preprocessed_'+ dataset  + '/' + filename        
        
        os.rename(origin_path, destination_path)

        #print(dataset)
        #if dataset == 'ucm':
        #    origin_path = '../data/processed/'
        #elif dataset == 'rsicd':
        #    origin_path = '../data/processed/'
        #else: # add try-catch for the script version
        #    print("uh-oh")


# In[ ]:





# In[ ]:





# In[ ]:




