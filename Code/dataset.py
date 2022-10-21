import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.metrics import f1_score
import os
from pathlib import Path
import copy
import nlpaug.augmenter.word as naw # for data augmentation on word level
from PIL import Image
from torchvision import transforms

'''
data_path = Path(os.path.dirname(os.path.abspath(os.getcwd()))).parent
data_root = os.path.join(data_path, "Output4.csv")
dataset = pd.read_csv(data_root)
'''

def create_labels_year(data):
    
    data = remove_data_w_nc_labels(data)
    # LABELS
    labels = []
    '''
    for i in range(0, len(data)):
        if (data[i,10]>2015):
            labels.append([1,0])
        else:
            labels.append([0,1])
    '''
    tag2id = {
            "2022":0,
            "2021":1,
            "2020":2,
            "2019":3,
            "2018":4,
            "2017":5,
            "2016":6,
            "2015":7,
            "2014":8,
            "2013":9,
            "2012":10,
            "2011":11,
            "2010":12,
            "2009":13,
            "2008":14,
            "2007":15,
            "2006":16,
            "2005":17,
            "2004":18,
            "2003":19,
            "2002":20,
            "2001":21,
            "2000":22,
            "1999":23,
            "1998":24,
            "1997":25,
            "1996":26,
            "1995":27,
            "1994":28,
            "1993":29,
            "below 1993":30}
    id2tag = {id: tag for tag, id in tag2id.items()}
    for i in range(0, len(data)):
      if (data[i,10]==2022):
        labels.append([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2021):
        labels.append([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2020):
        labels.append([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2019):
        labels.append([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2018):
        labels.append([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2017):
        labels.append([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2016):
        labels.append([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2015):
        labels.append([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2014):
        labels.append([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2013):
        labels.append([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2012):
        labels.append([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2011):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2010):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2009):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2008):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2007):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2006):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2005):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2004):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2003):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2002):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2001):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
      elif (data[i,10]==2000):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])
      elif (data[i,10]==1999):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])
      elif (data[i,10]==1998):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])
      elif (data[i,10]==1997):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])
      elif (data[i,10]==1996):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])
      elif (data[i,10]==1995):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])
      elif (data[i,10]==1994):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])
      elif (data[i,10]==1993):
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])
      else:
        labels.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    
    return labels, data, id2tag

def create_labels_reg(data):
    
    data = np.array(data)
    data = remove_data_w_nc_labels(data)
    
    # labels for cites/year
    labels = []
    #labels = data[:,12]
    '''
    for i in range(0, len(data)):
        if data[i,12]<2:
            labels.append([1,0])
        else:
            labels.append([0,1])
    '''
    tag2id = {
            "<2": 0,
            "<8": 1,
            "<30": 2,
            "<60": 3,
            ">=60": 4,
            }
    id2tag = {id: tag for tag, id in tag2id.items()}
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
    
    
    return labels, data, id2tag

def remove_empty_fig(data):
    
    np.array(data)
    remove_indexes = []
    for i in range(len(data)):
        if type(data[i,17])==float or data[i,17] == '[]':
            remove_indexes.append(i)
    
    remove_indexes = list(set(remove_indexes))
    remove_indexes.sort(reverse=True)
    data = list(data)
    for i in remove_indexes:
        del data[i]
    data = np.array(data)
    '''
    # remove figures that are too big memory wise
    images_path_files = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Figures/")
    for i in range(len(data)):
        if i%100==0:
            print("iterated through "+str(i))
        names = []
        ll = data[i,17].strip('][').split(', ')
        
        for j in range(len(ll)):
            names.append(str(images_path_files)+str(ll[j][1:][:-1]))
        names2 = []
        for name in names:
            if len(names2)>4:
                break
            file_stats = os.stat(name)
            if file_stats.st_size<200000:
                names2.append(name)
                
        data[i,17]=names2
    '''
    return data

def remove_data_w_nc_labels(data):
    
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
    
    remove_indexes = []
    for i in range(len(data)):
        if (type(data[i,4])!=float and "(cs." in data[i,4]):
            l = data[i,4].split("; ")
            flag = True        
            for j in l:
                if not "(cs." in j:
                    flag = False
            if not flag:
                remove_indexes.append(i)
    # semi harsh filtering (only containing papers that include at least one cs subject)
    for i in range(len(data)):
        if (type(data[i,4])!=float and not "(cs." in data[i,4]):
            remove_indexes.append(i)
    
    
    remove_indexes = list(set(remove_indexes))
    remove_indexes.sort(reverse=True)
    data = list(data)
    for i in remove_indexes:
        del data[i]
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
    
    return data

def create_labels_key(data):
    
    data = np.array(data)
    
    data = remove_data_w_nc_labels(data)
    
    subjects = []
    labels = []
    for i in range(0,len(data)):
      sub = data[i,4].split("; ")
      for j in range(0,len(sub)):
        subjects.append(sub[j])
    subjects = list(set(subjects))
    num_labels = len(subjects)
    tag2id = {}
    for i in range(len(subjects)):
        tag2id[str(subjects[i])] = i
    id2tag = {id: tag for tag, id in tag2id.items()}
    for i in range(0, len(data)):
        label = np.zeros(num_labels)
        for j in range(0, len(subjects)):
          if subjects[j] in data[i,4]:
            label[j] = 1
        labels.append(list(label))
    '''
    # counter data class imbalance
    # remove all papers assigned to classes with less than 1% occurrence
    labels_proportion = np.sum(labels,axis=0)/len(labels)
    label_selection = (labels_proportion>0.05)
    indexes = []
    for i in range(0,len(label_selection)):
        if label_selection[i]:
            indexes.append(i)
            
    remove_indexes = []
    for i in range(len(data)):
        for j in range(len(labels[i])):
            if labels[i][j] > 0 and not label_selection[j]:
                remove_indexes.append(i)
    remove_indexes = list(set(remove_indexes))
    remove_indexes.sort(reverse=True)
    data = list(data)
    #labels = list(labels)
    for i in remove_indexes:
        del data[i]
        del labels[i]
    data = np.array(data) 
    print(len(labels))
    print(len(labels[0]))
    print(len(indexes))
    labels = np.array(labels)[:,[indexes]]
    labels = labels.reshape(labels.shape[0],labels.shape[2]) 
    '''
    #data, labels = under_sampling(data, labels)
    
    return labels, data, id2tag

def create_labels_mtl_year_key(data):
    data = np.array(data)
    
    data = remove_data_w_nc_labels(data)
    
    labels_y, data, id2tag_year = create_labels_year(data)
    labels_k, data, id2tag_key = create_labels_key(data)
    labels = []
    for i in range(len(labels_k)):
        labels.append((labels_y[i], labels_k[i]))
        
    return labels, data, id2tag_year, id2tag_key

def under_sampling(data, labels):
    
    # under-sampling to counter class-imbalance
    # method: take amount of most underrepresented label and reduce every other class to at most 10 times that amount
    # ensures 1:10 as biggest class imbalance
    num_labels = len(labels[0])
    ratio = 1
    labels_amounts = np.sum(labels,axis=0)
    min_label_amount = min(labels_amounts)
    labels_amount_sorted = labels_amounts.copy()
    labels_amount_sorted.sort()
    min_label_amount = min(np.sum(labels,axis=0))
    max_label_amount = max(np.sum(labels,axis=0))
    print("Initial Imbalance Ratio: 1:"+str(max_label_amount/min_label_amount))
    print(min_label_amount)
    
    label_selection = (labels_amounts>(min_label_amount*ratio))
    label_count = np.zeros(num_labels)
    labels_new = []
    data_new = []
    for j in range(len(labels_amount_sorted)):
        remove_indexes = []
        for l in range(len(labels_amounts)):
            if labels_amount_sorted[j] == labels_amounts[l]:
                index = l
                break
        for i in range(len(data)):
            if labels[i][index] > 0 and label_count[index]<=(min_label_amount*ratio):
                flag = True
                for k in range(num_labels):
                    if labels[i][k] > 0 and label_count[k]>(min_label_amount*ratio):
                        flag = False
                if flag:
                    labels_new.append(labels[i])
                    data_new.append(data[i])
                    remove_indexes.append(i)
                    for k in range(num_labels):
                        if labels[i][k] > 0:
                            label_count[k] += 1
        remove_indexes = list(set(remove_indexes))
        remove_indexes.sort(reverse=True)  
        data = list(data)
        for i in remove_indexes:
            del data[i]
            del labels[i]
        data = np.array(data)
    data_new = np.array(data_new)
    labels_new = np.array(labels_new)
    min_label_amount = min(np.sum(labels_new,axis=0))
    max_label_amount = max(np.sum(labels_new,axis=0))
    print("New Imbalance Ratio: 1:"+str(max_label_amount/min_label_amount))
    
    return data_new, labels_new


def over_sampling(data, labels):
    
    # over-sampling to counter class-imbalance
    # method: take amount of most represented label and increase every other class
    # ensures 5:6 as biggest class imbalance
    num_labels = len(labels[0])
    #ratio = 1.15
    ratio = 50
    labels_amounts = np.sum(labels,axis=0)
    max_label_amount = max(labels_amounts)
    labels_amount_sorted = labels_amounts.copy()
    labels_amount_sorted.sort()
    min_label_amount = min(np.sum(labels,axis=0))
    max_label_amount = max(np.sum(labels,axis=0))
    print("Initial Imbalance Ratio: 1:"+str(max_label_amount/min_label_amount))
    print(max_label_amount)
    freq = int((1/ratio)/(1/(max_label_amount/min_label_amount)))
    print(freq)
    
    label_count = np.zeros(num_labels)
    labels_new = []
    data_new = []
    for d in range(len(data)):
        data_new.append(data[d])
        labels_new.append(labels[d])
        for lab in range(num_labels):
            if labels[d][lab] > 0:
                label_count[lab] += 1
    data_n = len(data_new)
    for j in range(len(labels_amount_sorted)):
        for l in range(len(labels_amounts)):
            if labels_amount_sorted[j] == labels_amounts[l]:
                index = l
                break
        for i in range(len(data)):
            if labels[i][index] > 0 and label_count[index]<=(max_label_amount/ratio):
                flag = True
                for rl in range(num_labels):
                    if labels[i][rl] > 0 and label_count[rl]>max_label_amount:
                        flag = False
                # augment
                if flag:
                    aug_titles = augment(data[i,0], freq)
                    for am in range(len(aug_titles)):
                        aug_d = copy.deepcopy(data[i])
                        aug_d[0] = aug_titles[am]
                        labels_new.append(labels[i])
                        data_new.append(aug_d)
                    for lc in range(num_labels):
                        if labels[i][lc] > 0:
                            label_count[lc] += len(aug_titles)
    
    min_label_amount = min(np.sum(labels_new,axis=0))
    max_label_amount = max(np.sum(labels_new,axis=0))
    print("New Imbalance Ratio: 1:"+str(max_label_amount/min_label_amount))
    print("Augmented Data: "+ str(len(data_new)-data_n))
    data_new = np.array(data_new)
    labels_new = np.array(labels_new)
    
    return data_new, labels_new


def augment(title, freq):
    # experiment with augmentation parameters
    # Augmentation techniques initialization
    # Random Deletion
    aug_del = naw.RandomWordAug(action="delete", aug_min=1, aug_max=1, aug_p=0.4)
    # Random Swap
    aug_swap = naw.RandomWordAug(action="swap", aug_min=1, aug_max=1, aug_p=0.4)
    # Back Translation from en->de->en (could try different translations)
    # too expensive to execute atm
    # aug_btrans = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')
    # Synonym Replacement (using WordNet)
    aug_syn = naw.SynonymAug(aug_src='wordnet', aug_min=2, aug_max=2, aug_p=0.7)
    
    # actual augmentation:
    # split augmentation into three parts:
    aug_titles = aug_del.augment(title, int(freq/3))
    aug_titles = aug_titles+aug_swap.augment(title, int(freq/3))
    aug_titles = aug_titles+aug_syn.augment(title, int(freq/3))
    
    return aug_titles

def match_data_to_images(data):
    
    data_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
    data = np.array(data)
    
    if len(data[0])<17:
        # add one column for images
        data = np.append(data, np.full((len(data),1),None), axis=1)
    
    for i in range(len(data)):
        path_to_file = os.path.join(data_path, "Data/Images/"+str(i)+"_1.jpg")
        if os.path.exists(path_to_file):
            data[i,16] = str(i)+"_1.jpg"
    '''
    remove_indexes = []
    for i in range(len(data)):
        if type(data[i,16])==float:
            remove_indexes.append(i)
    
    remove_indexes = list(set(remove_indexes))
    remove_indexes.sort(reverse=True)
    data = list(data)
    for i in remove_indexes:
        del data[i]
    data = np.array(data)
    print(len(data))
    '''
    return data

def save_file(data, data_path):
    
    # save as DataFrame and make descriptive header
    if (type(data)==pd.core.frame.DataFrame):
        data = data[["Title", "Article_Link", "Publication_Info", "PDF", "Keyword", "Main_Author", "Source", "Abstract", "Cites", "Versions", "Publication_Year", "Venue", "Cites/Year", "Comments", "Journal", "Venues2", "Image_Name", "Figue_Names", "Google_Article_Link"]]
    df = pd.DataFrame(data)
    df.columns = ["Title", "Article_Link", "Publication_Info", "PDF", "Keyword", "Main_Author", "Source", "Abstract", "Cites", "Versions", "Publication_Year", "Venue", "Cites/Year", "Comments", "Journal", "Venues2", "Image_Name", "Figure_Names", "Google_Article_Link"]
    
    df.to_csv(str(data_path) + '/Data/Full_Data_filtered_Test.csv', index=False) 

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