import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.metrics import f1_score
import os
from pathlib import Path

'''
data_path = Path(os.path.dirname(os.path.abspath(os.getcwd()))).parent
data_root = os.path.join(data_path, "Output4.csv")
dataset = pd.read_csv(data_root)
'''

def create_labels_year(data):
    
    # create labels for each datapoint
    data = np.array(data)
    remove_index = []
    # remove invalid/missing years
    for i in range(0,len(data)):
        if data[i,10]==0:
            remove_index.append(i)
    remove_index = set(remove_index)
    remove_index = list(remove_index)
    remove_index.sort(reverse=True)
    data = list(data)
    for i in remove_index:
        del data[i]
    
    data = np.array(data)
    # LABELS
    labels = []
    '''
    for i in range(0, len(data)):
        if (data[i,10]>2015):
            labels.append([1,0])
        else:
            labels.append([0,1])
    '''
    for i in range(0, len(data)):
      if (data[i,10]==2022):
        labels.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2021):
        labels.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2020):
        labels.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2019):
        labels.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2018):
        labels.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2017):
        labels.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2016):
        labels.append([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2015):
        labels.append([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2014):
        labels.append([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2013):
        labels.append([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2012):
        labels.append([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2011):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2010):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2009):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2008):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2007):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
      elif (data[i,10]==2006):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
      elif (data[i,10]==2005):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
      elif (data[i,10]==2004):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
      elif (data[i,10]>=2000):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
      elif (data[i,10]>=1995):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
      elif (data[i,10]>=1990):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
      else:
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    
    return labels, data

def create_labels_reg(data):
    
    data = np.array(data)
    
    # labels for cites/year
    labels = []
    #labels = data[:,12]
    for i in range(0, len(data)):
        if data[i,12]<2:
            labels.append([1,0])
        else:
            labels.append([0,1])
    '''
    for i in range(0, len(data)):
        if data[i,12]<2:
            labels.append([1,0,0,0,0])
        elif data[i,12]<8:
            labels.append([0,1,0,0,0])
        elif data[i,12]<30:
            labels.append([0,0,1,0,0])
        elif data[i,12]<60:
            labels.append([0,0,0,1,0])
        else:
            labels.append([0,0,0,0,1])
    '''
    
    return labels

def create_labels_key(data):
    
    data = np.array(data)
    
    # create labels for keywords/areas
    remove_indexes = []
    for i in range(0, len(data)):
        if type(data[i,4])==float:
            remove_indexes.append(i)
        elif not "(" in data[i,4]:
            remove_indexes.append(i)
    remove_indexes.sort(reverse=True)
    data = list(data)
    for i in remove_indexes:
        del data[i]
    data = np.array(data)
    subjects = []
    labels = []
    for i in range(0,len(data)):
      sub = data[i,4].split("; ")
      for j in range(0,len(sub)):
        subjects.append(sub[j])
    subjects = list(set(subjects))
    num_labels = len(subjects)
    for i in range(0, len(data)):
        label = np.zeros(num_labels)
        for j in range(0, len(subjects)):
          if subjects[j] in data[i,4]:
            label[j] = 1
        labels.append(list(label))
        
    return labels, data

'''
data = dataset.to_numpy()
data = sep_comp(data)
data = data[:, [0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]
data = np.asarray(data).astype('float32')
#data = torch.FloatTensor(data)

# handle unknown datapoints (not ideal solution)
# -> try to average data or take previous values for same entry instead
data = data[~np.isnan(data).any(axis=1)]
#data = data[~np.isinf(data).any(axis=1)]
'''