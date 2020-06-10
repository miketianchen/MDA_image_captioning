#!/usr/bin/env python
# coding: utf-8

# In[63]:


import json
import numpy as np
import pandas as pd
from collections import defaultdict
from functools import reduce
from sklearn.model_selection import train_test_split


# In[64]:


json_data = {}
sizes = {}
#set_name = ['rsicd', 'ucm', 'sydney']
set_name = ['rsicd', 'ucm']

# read in json files from all three datasets
for name in set_name:
    with open('data/raw/dataset_' + name + '_modified.json', 'r') as data:
        json_data[name] = json.load(data)
        sizes[name] = len(json_data[name]['images'])
        print(f'There are {sizes[name]} images in the {name} dataset.')


# In[65]:


ucm_data = {}
sydney_data = {}
rsicd_data = {}

for name in set_name:
    for single_image in json_data[name]['images']:
        new_filename = name + '_' + single_image['filename'][:-3] + 'jpg'
        if (name == 'rsicd'):
            rsicd_data[new_filename] = single_image
            rsicd_data[new_filename]['old_dataset_name'] = 'dataset_rsicd_modified'
        elif (name == 'ucm'):
            ucm_data[new_filename] = single_image
            ucm_data[new_filename]['old_dataset_name'] = 'dataset_ucm_modified'
        elif (name == 'sydney'):
            sydney_data[new_filename] = single_image
            sydney_data[new_filename]['old_dataset_name'] = 'dataset_sydney_modified'
        else: 
            print("uh-oh") #should add try catch later
            
print(f'There are {len(rsicd_data)} images in the rsicd dataset.')
print(f'There are {len(ucm_data)} images in the ucm dataset.')
print(f'There are {len(sydney_data)} images in the sydney dataset.')


# In[66]:


# Python code to merge dict 
def merge_2(dict1, dict2, dict3):
    dict2.update(dict1)
    return(dict3.update(dict2)) 

def merge(dict1, dict2, dict3): 
    res = {**dict1, **dict2, **dict3} 
    return res


# In[67]:


combined_data = merge(rsicd_data, ucm_data, sydney_data)


# In[68]:


len(combined_data)


# In[69]:


combined_df = pd.DataFrame(combined_data).T


# In[70]:


combined_df.head(-5)


# In[71]:


train_valid, test = train_test_split(combined_df, test_size=0.2, random_state=123)
train, valid = train_test_split(train_valid, test_size=0.2, random_state=123)



train_dict = train.to_dict(orient='index')
test_dict = test.to_dict(orient='index')
valid_dict = valid.to_dict(orient='index')


# In[79]:


# In each of the images add the current set name key-value pair
new_set_name = ['train', 'test', 'valid']
for name in new_set_name:
    if (name == 'train'):
        for key, value in train_dict.items():
            value['split'] = name
    elif (name == 'test'):
        for key, value in test_dict.items():
            value['split'] = name
    elif (name == 'valid'):
        for key, value in valid_dict.items():
            value['split'] = name
    else:
        print("uh-oh")


# In[80]:


imgs_names = [(train_dict, 'train', train), 
              (valid_dict, 'valid', valid), 
              (test_dict, 'test', test)]


# In[81]:


for imgs, name, _ in imgs_names:
    with open('data/processed/json/' + name + '.json', 'w') as file:
        json.dump(imgs, file)


# In[ ]:





# In[ ]:




