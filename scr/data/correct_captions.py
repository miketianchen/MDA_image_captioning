#!/usr/bin/env python
# coding: utf-8

# In[3]:


import json
import random


# In[4]:


# load train json data
name = 'train'
with open('data/processed/json/' + name + '.json', 'r') as jsonFile:
    data = json.load(jsonFile)


# In[5]:


# create problematic caption dictionary
count_err = 0
err_cap_dict = {}
for file in data.keys():
    for caption in data[file]['sentences']:
        if len(caption['tokens']) < 2:
            if file in err_cap_dict.keys():
                err_cap_dict[file].append(caption['sentid'])
            else:
                err_cap_dict[file] = [caption['sentid']]
            


# In[6]:


# replace the problematic caption with random selected caption from the same image
for key, val in err_cap_dict.items():
    caption_list = [sentence['sentid'] for sentence in data[key]['sentences'] if sentence['sentid'] not in val]
    for sentence in data[key]['sentences']:
        if sentence['sentid'] in val:

            sel_caption = random.choice(caption_list)
            sel_tokens = [sentence['tokens'] for sentence in data[key]['sentences'] if sentence['sentid'] == sel_caption][0]
            sel_raw = [sentence['raw'] for sentence in data[key]['sentences'] if sentence['sentid'] == sel_caption][0]
            
            sentence['tokens'] = sel_tokens
            sentence['raw'] = sel_raw


# In[7]:


# save the processed data
with open('data/processed/json/' + name + '.json', 'w') as jsonFile:
    json.dump(data, jsonFile)


# In[ ]:





# In[ ]:




