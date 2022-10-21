"""
Definition of PaperDataset dataset class
"""

# pylint: disable=too-few-public-methods

import os
import torch
from transformers import AutoTokenizer, BertTokenizer
import numpy as np
import random
import time
import copy
from nltk.corpus import wordnet # for synonyms
import nlpaug.augmenter.word as naw # for data augmentation on word level
import nlpaug.augmenter.sentence as nas # for data augmentation on sentence level only relevant for abstracts
from Data_cleaning import pre_tokenize_title, pre_tokenize_abstract
from PIL import Image
from torchvision import transforms
from pathlib import Path
import torch as nn
import math
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

import _pickle as pickle

class PaperDataset():

    def __init__(self, data, labels, model, hparams, data_type, input_type):
        self.hparams = hparams

        # Augment Training Data
        if data_type=="train" and hparams['aug_amount']>0:
            start = time.time()
            #data, labels = self.augmentation(data, labels) # own augmentation
            data, labels = self.augmentation_help(data, labels) # augmentation using nlpaug library
            print(time.time()-start)
            print("Length of Training Data after Augmentation: "+str(len(data)))
        if "title" in input_type:
            data = pre_tokenize_title(data)
        if "abstract" in input_type:
            data = pre_tokenize_abstract(data)
        tokenizer = AutoTokenizer.from_pretrained(model, add_prefix_space=True)
        # two sequences 0: title, 1: abstract
        if input_type=="title":
            encodings_t = tokenizer(list(data[:,0]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_t.pop("offset_mapping")
            self.encodings_t = encodings_t            
        elif input_type=="abstract":
            encodings_a = tokenizer(list(data[:,1]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_a.pop("offset_mapping")
            self.encodings_a = encodings_a            
        elif input_type=="title+abstract":
            encodings_t = tokenizer(list(data[:,0]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_a = tokenizer(list(data[:,1]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_t.pop("offset_mapping")
            encodings_a.pop("offset_mapping")
            self.encodings_t = encodings_t
            self.encodings_a = encodings_a
        elif input_type=="image":
            image_paths_file = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Images/")
            self.root_dir_name = os.path.dirname(image_paths_file) # need to define image_paths
            self.image_names = []
            for i in range(len(data)):
                self.image_names.append(str(image_paths_file)+str(data[i,2])) 
        elif input_type=="title+abstract+image":
            encodings_t = tokenizer(list(data[:,0]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_a = tokenizer(list(data[:,1]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_t.pop("offset_mapping")
            encodings_a.pop("offset_mapping")
            self.encodings_t = encodings_t
            self.encodings_a = encodings_a
            image_paths_file = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Images/")
            self.root_dir_name = os.path.dirname(image_paths_file) # need to define image_paths
            self.image_names = []
            for i in range(len(data)):
                self.image_names.append(str(image_paths_file)+str(data[i,2]))
        elif input_type=="figures":
            images_path_files = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Figures/")
            self.root_dir_name = os.path.dirname(images_path_files)
            self.figure_names = []
            for i in range(len(data)):
                names = []
                ll = data[i,3].strip('][').split(', ')
                for j in range(len(ll)):
                    names.append(str(images_path_files)+str(ll[j][1:][:-1]))
                self.figure_names.append(names)
        elif input_type=="title+abstract+figures":
            encodings_t = tokenizer(list(data[:,0]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_a = tokenizer(list(data[:,1]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_t.pop("offset_mapping")
            encodings_a.pop("offset_mapping")
            self.encodings_t = encodings_t
            self.encodings_a = encodings_a
            images_path_files = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Figures/")
            self.root_dir_name = os.path.dirname(images_path_files)
            self.figure_names = []
            for i in range(len(data)):
                names = []
                ll = data[i,3].strip('][').split(', ')
                for j in range(len(ll)):
                    names.append(str(images_path_files)+str(ll[j][1:][:-1]))
                self.figure_names.append(names)
        elif input_type=="title+abstract+image+figures":
            encodings_t = tokenizer(list(data[:,0]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_a = tokenizer(list(data[:,1]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
            encodings_t.pop("offset_mapping")
            encodings_a.pop("offset_mapping")
            self.encodings_t = encodings_t
            self.encodings_a = encodings_a
            image_paths_file = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Images/")
            self.root_dir_name = os.path.dirname(image_paths_file) # need to define image_paths
            self.image_names = []
            for i in range(len(data)):
                self.image_names.append(str(image_paths_file)+str(data[i,2]))
            images_path_files = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Figures/")
            self.figure_names = []
            for i in range(len(data)):
                names = []
                ll = data[i,3].strip('][').split(', ')
                for j in range(len(ll)):
                    names.append(str(images_path_files)+str(ll[j][1:][:-1]))
                self.figure_names.append(names)
        elif input_type=="image+figures":
            image_paths_file = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Images/")
            self.root_dir_name = os.path.dirname(image_paths_file) # need to define image_paths
            self.image_names = []
            for i in range(len(data)):
                self.image_names.append(str(image_paths_file)+str(data[i,2]))
            images_path_files = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Figures/")
            self.figure_names = []
            for i in range(len(data)):
                names = []
                ll = data[i,3].strip('][').split(', ')
                for j in range(len(ll)):
                    names.append(str(images_path_files)+str(ll[j][1:][:-1]))
                self.figure_names.append(names)
        else:
            AssertionError

        self.labels = labels
        self.input_type = input_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if self.input_type=="title":
            item = item = {key+'_t': torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
        elif self.input_type=="abstract":
            item = {key+'_a': torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            #item['labels'] = torch.tensor(self.labels[idx])
        elif self.input_type=="title+abstract":
            item = {key+'_t': torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
            item2 = {key+'_a': torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            item['input_ids_a'] = item2['input_ids_a']
            item['attention_mask_a'] = item2['attention_mask_a']
            #item['labels'] = torch.tensor(self.labels[idx])
        elif self.input_type=="image":
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                img, target_labels = self.get_item_from_index(idx)
                item = {"img": img}
                #item['labels'] = target_labels
            else:
                raise TypeError("Invalid argument type.")
        elif self.input_type=="title+abstract+image":
            item = {key+'_t': torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
            item2 = {key+'_a': torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            item['input_ids_a'] = item2['input_ids_a']
            item['attention_mask_a'] = item2['attention_mask_a']
            #item['labels'] = torch.tensor(self.labels[idx])
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                img, target_labels = self.get_item_from_index(idx)
                item['img'] = img
            else:
                raise TypeError("Invalid argument type.")
                
        elif self.input_type=="figures":
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                fig, target_labels, seq_len = self.get_fig_from_index(idx)
                
                item = {"fig": fig}
                item['labels'] = target_labels
                item['seq_len'] = seq_len
            else:
                raise TypeError("Invalid argument type.")
        elif self.input_type=="title+abstract+figures":
            item = {key+'_t': torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
            item2 = {key+'_a': torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            item['input_ids_a'] = item2['input_ids_a']
            item['attention_mask_a'] = item2['attention_mask_a']
            #item['labels'] = torch.tensor(self.labels[idx])
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                fig, target_labels, seq_len = self.get_fig_from_index(idx)
                
                item["fig"] = fig
                item['labels'] = target_labels
                item['seq_len'] = seq_len
            else:
                raise TypeError("Invalid argument type.")
        elif self.input_type=="title+abstract+image+figures":
            item = {key+'_t': torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
            item2 = {key+'_a': torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            item['input_ids_a'] = item2['input_ids_a']
            item['attention_mask_a'] = item2['attention_mask_a']
            #item['labels'] = torch.tensor(self.labels[idx])
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                fig, target_labels, seq_len = self.get_fig_from_index(idx)
                
                item["fig"] = fig
                item['labels'] = target_labels
                item['seq_len'] = seq_len
            else:
                raise TypeError("Invalid argument type.")
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                img, target_labels = self.get_item_from_index(idx)
                item['img'] = img
            else:
                raise TypeError("Invalid argument type.")
        elif self.input_type=="image+figures":
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                fig, target_labels, seq_len = self.get_fig_from_index(idx)
                
                item = {"fig": fig}
                item['labels'] = target_labels
                item['seq_len'] = seq_len
            else:
                raise TypeError("Invalid argument type.")
            if isinstance(idx, slice):
                # get the start, stop, and step from the slice
                #return [self[ii] for ii in range(*idx.indices(len(self)))]
                print("failed data loading")
            elif isinstance(idx, int):
                # handle negative indices
                if idx < 0:
                    idx += len(self)
                if idx < 0 or idx >= len(self):
                    raise IndexError("The index (%d) is out of range." % idx)
                    # get the data from direct index
                img, target_labels = self.get_item_from_index(idx)
                item['img'] = img
            else:
                raise TypeError("Invalid argument type.")
        else:
            AssertionError
        if self.input_type!="figures":
            if self.hparams["mtl"]:
                item['labels_year'] = torch.tensor(self.labels[idx][0])
                item['labels_key'] = torch.tensor(self.labels[idx][1])
            else:
                item['labels'] = torch.tensor(self.labels[idx])
        
        return item
    
    def get_fig_from_index(self, index):
        to_tensor = transforms.ToTensor()
        crop_size = self.hparams["fig_crop_size"]
        center_crop = transforms.CenterCrop(crop_size) # Center crop for now as img sizes are very different
        fig = []
        for name in self.figure_names[index]:
            fig_sub = Image.open(name).convert('RGB')
            new_size_ratio = self.hparams["img_sz_ratio"] # reduce size
            if (fig_sub.size[0]>400 and fig_sub.size[1]>10) or (fig_sub.size[1]>400 and fig_sub.size[0]>10):
                fig_sub = fig_sub.resize((int(fig_sub.size[0] * new_size_ratio), int(fig_sub.size[1] * new_size_ratio)), Image.ANTIALIAS)
            fig_sub = to_tensor(fig_sub)
            fig_sub = center_crop(fig_sub)
            fig.append(fig_sub)
            
            if len(fig)==self.hparams["max_seq"]:
                break
        seq_len = len(fig)
        # 0-pad sequence with 0-images
        while len(fig)<self.hparams["max_seq"]:
            fig.append(torch.zeros([3,crop_size,crop_size]))
        
        fig = pad_sequence(fig)
        fig = fig.reshape(fig.shape[1], fig.shape[0], fig.shape[2], fig.shape[3])#[0]# seq_length X rgb X length X width
        target_labels = torch.tensor(self.labels[index])
        
        return fig, target_labels, seq_len
        
    
    def get_item_from_index(self, index): # adjust this function
        to_tensor = transforms.ToTensor()
        #img_id = self.image_names[index].replace('.jpg', '')
        img = Image.open(self.image_names[index]).convert('RGB')
        
        img = self.resize_img(img, downscale=True)

        img = to_tensor(img)
        # remove arXiv description to avoid passing the labels via the input:
        img = self.remove_arXiv_text(img)

        target_labels = torch.tensor(self.labels[index])

        return img, target_labels
    
    def remove_arXiv_text(self, img):
        #trans = transforms.ToPILImage() # for converting back to Image from Tensor
        new_size_ratio = self.hparams["img_sz_ratio"]
        if new_size_ratio==1.0:
            img[:,120:700,13:37] = nn.ones([3,580,24])
        else:
            img_h_s = int(new_size_ratio*120)
            img_h_e = int(new_size_ratio*700)
            img_w_s = int(new_size_ratio*12)
            img_w_e = int(new_size_ratio*38)
            cover_h = int(new_size_ratio*580)
            cover_w = int(new_size_ratio*26)
            img[:,img_h_s:img_h_e,img_w_s:img_w_e] = nn.ones([3,cover_h,cover_w])
        
        return img
    
    def resize_img(self, img, downscale=False):
        
        if downscale:
            new_size_ratio = self.hparams["img_sz_ratio"] # reduce size
            img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.ANTIALIAS)
            
            #center_crop = transforms.CenterCrop(842)
            width, height = img.size   # Get current dimensions
            new_width, new_height = int(612*new_size_ratio), int(842*new_size_ratio) # ensures that no information is left out
            
            left = math.ceil((width - new_width)/2)
            top = math.ceil((height - new_height)/2)
            right = math.ceil((width + new_width)/2)
            bottom = math.ceil((height + new_height)/2)
            '''
            if bottom%1!=0:
                bottom = math.ceil(bottom)
            if right%1!=0:
                right = math.ceil(right)
            '''
        else:
            #center_crop = transforms.CenterCrop(842)
            width, height = img.size   # Get current dimensions
            new_width, new_height = 612, 842 # ensures that no information is left out
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

        # Crop the center of the image
        img = img.crop((left, top, right, bottom))
        #img = center_crop(img)

        return img
        
    def augmentation_help(self, data, labels):
        # More on: https://nlpaug.readthedocs.io/en/latest/augmenter/word/synonym.html
        # Random Deletion
        aug_del = naw.RandomWordAug(action="delete", aug_min=1, aug_p=0.2)
        # Random Swap
        aug_swap = naw.RandomWordAug(action="swap", aug_min=1, aug_p=0.2)
        # Back Translation from en->de->en (could try different translations)
        aug_btrans = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de', to_model_name='facebook/wmt19-de-en')
        # Synonym Replacement (using WordNet)
        aug_syn = naw.SynonymAug(aug_src='wordnet', aug_min=1, aug_max=3, aug_p=0.2)
        # Antonym Augmentation
        aug_an = naw.AntonymAug(aug_min=1, aug_max=3, aug_p=0.2)
        # Contextual Word Embedding Augmentation Substitution
        aug_con_sub = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='substitute', aug_min=1, aug_max=3, aug_p=0.2)
        # Contextual Word Embedding Augmentation Insertion
        aug_con_in = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert', aug_min=1, aug_max=3, aug_p=0.2)
        augmented_data = []
        augmented_labels = []
        for i in range(self.hparams['aug_amount']):
            aug_d = copy.deepcopy(data)
            
            # Random Deletion
            if self.hparams['rd']:
                aug_d[:,0] = aug_del.augment(list(data[:,0])) # only for title for now!
                if augmented_data == []:
                    augmented_data = aug_d
                    augmented_labels = labels
                else:
                    augmented_data = np.vstack((augmented_data, aug_d))
                    augmented_labels = np.vstack((augmented_labels, labels))
            
            # Random Swap
            if self.hparams['rs']:
                aug_d[:,0] = np.array(aug_swap.augment(list(data[:,0])))
                if augmented_data == []:
                    augmented_data = aug_d
                    augmented_labels = labels
                else:
                    augmented_data = np.vstack((augmented_data, aug_d))
                    augmented_labels = np.vstack((augmented_labels, labels))
            
            '''
            # Super Slow
            # Back Translation from en->de->en
            aug_d[:,0] = aug_btrans.augment(list(data[:,0]))
            augmented_data = augmented_data + np.array(aug_d)
            augmented_labels = augmented_labels + labels
            '''        
            # Synonym Replacement using WordNet
            if self.hparams['syn_rep']:
                aug_d[:,0] = np.array(aug_syn.augment(list(data[:,0])))
                if augmented_data == []:
                    augmented_data = aug_d
                    augmented_labels = labels
                else:
                    augmented_data = np.vstack((augmented_data, aug_d))
                    augmented_labels = np.vstack((augmented_labels, labels))
            
            # Antonym Augmentation
            if self.hparams['an_aug']:
                aug_d[:,0] = np.array(aug_an.augment(list(data[:,0])))
                if augmented_data == []:
                    augmented_data = aug_d
                    augmented_labels = labels
                else:
                    augmented_data = np.vstack((augmented_data, aug_d))
                    augmented_labels = np.vstack((augmented_labels, labels))
            
            '''
            # Also rather slow
            # Contextual Word Embedding Augmentation Substitution
            aug_d[:,0] = aug_con_sub.augment(list(data[:,0]))
            augmented_data = augmented_data + np.array(aug_d)
            augmented_labels = augmented_labels + labels
            
            # Contextual Word Embedding Augmentation Insertion
            aug_d[:,0] = aug_con_in.augment(list(data[:,0]))
            augmented_data = augmented_data + np.array(aug_d)
            augmented_labels = augmented_labels + labels
            '''
        print("Augmented sequences: "+str(len(augmented_data)))
        # add augmented sequences to data
        data = np.vstack((data,augmented_data))
        labels = np.vstack((labels,augmented_labels))
        
        return data, labels
        
    
    def augmentation(self, data, labels):
        # remove at random words from input to augment data
        # More ideas for data augmentation:
        # Look into: https://neptune.ai/blog/data-augmentation-nlp
        words_per_seq = self.hparams['aug_prob_words']
        seq_per_data = self.hparams['aug_prob_seq']
        amount = self.hparams['aug_amount']
        rs_prob = self.hparams['aug_rs_prob']
        ri_prob = self.hparams['aug_ri_prob']
        rsr_prob = self.hparams['aug_rsr_prob']
        
        augmented_sequences = []
        augmented_labels = []
        for k in range(0,amount):
            for i in range(len(data)):
                if i%40000==0:
                    print("Iterated throuhg: "+str(i)+" papers")
                sd = random.randint(0,100)
                if sd<=seq_per_data*100:
                    # so far only for title
                    # Random Deletion:
                    for j in range(len(data[i,0])): # change index to 1 for abstract
                        rd = random.randint(0,100)
                        if rd<=words_per_seq*100:
                            aug_seq = copy.deepcopy(data[i])
                            aug_seq[0].pop(j)
                            augmented_sequences.append(aug_seq)
                            augmented_labels.append(labels[i])
                            #print("Random Deletion")
                            #print(aug_seq[0])
                    # Random Swap:
                    for j in range(0,self.hparams['aug_rs_amount']): # amount per seq
                        rs = random.randint(0,100)
                        if rs<=rs_prob*100:
                            aug_seq = copy.deepcopy(data[i])
                            first = random.randint(0,int(len(aug_seq[0])/2))
                            second = random.randint(round(len(aug_seq[0])/2),len(aug_seq[0])-1) # -1 to avoid going out of index
                            aug_seq[0][first] = data[i,0][second]
                            aug_seq[0][second] = data[i,0][first]
                            augmented_sequences.append(aug_seq)
                            augmented_labels.append(labels[i])
                            #print("Random Swap")
                            #print(aug_seq[0])
                    # Random Insertion:
                    for j in range(0,self.hparams['aug_ri_amount']): # amount per seq
                        ri = random.randint(0,100)
                        if ri<=ri_prob*100:
                            aug_seq = copy.deepcopy(data[i])
                            pos = random.randint(0,len(aug_seq[0]))
                            word_pos = random.randint(0,len(aug_seq[0])-1)
                            word = aug_seq[0][word_pos]
                            synonyms = []
                            count=0
                            while (synonyms==[] and count<15):
                                word_pos = random.randint(0,len(aug_seq[0])-1)
                                word = aug_seq[0][word_pos]
                                synonyms = self.get_synonyms(word)
                                count+=1
                            if len(synonyms)>0:
                                synonym = random.choice(synonyms)
                                aug_seq[0] = aug_seq[0][:pos]+[synonym]+aug_seq[0][pos:]
                                augmented_sequences.append(aug_seq)
                                augmented_labels.append(labels[i])
                                #print("Random Insertion")
                                #print(aug_seq[0])
                    # Random Synonym Replacement:
                    for j in range(0,self.hparams['aug_rsr_amount']): # amount per seq
                        rsr = random.randint(0,100)
                        if rsr<=rsr_prob*100:
                            aug_seq = copy.deepcopy(data[i])
                            synonyms = []
                            count=0
                            while (synonyms==[] and count<15):
                                word_pos = random.randint(0, len(aug_seq[0])-1) # -1 to avoid going out of index
                                word = aug_seq[0][word_pos]
                                synonyms = self.get_synonyms(word)
                                count+=1
                            if len(synonyms)>0:
                                synonym = random.choice(synonyms)
                                aug_seq[0][word_pos] = synonym
                                augmented_sequences.append(aug_seq)
                                augmented_labels.append(labels[i])
                                #print("Random Synonym Rep.")
                                #print(aug_seq[0])
                    
                    
        print("Augmented sequences: "+str(len(augmented_sequences)))
        # add augmented sequences to data
        data = np.vstack((data,augmented_sequences))
        labels = np.vstack((labels,augmented_labels))
        
        return data, labels
    
    def get_synonyms(self, word):
    	synonyms = set()
    	for syn in wordnet.synsets(word): 
    		for l in syn.lemmas(): 
    			synonym = l.name().replace("_", " ").replace("-", " ").lower()
    			synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
    			synonyms.add(synonym) 
    	if word in synonyms:
    		synonyms.remove(word)
    	return list(synonyms)
