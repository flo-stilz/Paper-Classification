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


class PaperDataset():
    """Stock dataset class"""

    def __init__(self, data, labels, model, hparams, data_type, input_type):
        self.hparams = hparams
        # Augment Training Data
        if data_type=="train" and hparams['aug_amount']>0:
            start = time.time()
            data, labels = self.augmentation(data, labels)
            print(time.time()-start)
            print("Length of Training Data after Augmentation: "+str(len(data)))
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
        else:
            AssertionError

        self.labels = labels
        self.input_type = input_type

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        if self.input_type=="title":
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
            item['labels'] = torch.tensor(self.labels[idx])
        elif self.input_type=="abstract":
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            item['labels'] = torch.tensor(self.labels[idx])
        elif self.input_type=="title+abstract":
            item = {key+'_t': torch.tensor(val[idx]) for key, val in self.encodings_t.items()}
            item2 = {key+'_a': torch.tensor(val[idx]) for key, val in self.encodings_a.items()}
            item['input_ids_a'] = item2['input_ids_a']
            item['attention_mask_a'] = item2['attention_mask_a']
            item['labels'] = torch.tensor(self.labels[idx])
        else:
            AssertionError

        return item
    '''
    def augmentation_help(self, data, labels):
        # More on: https://nlpaug.readthedocs.io/en/latest/augmenter/word/synonym.html
        # Random Deletion
        aug_del = naw.RandomWordAug(action="delete", aug_min=1, aug_p=0.2)
        # Random Swap
        aug_swap = naw.RandomWordAug(action="swap", aug_min=1, aug_p=0.2)
        # Back Translation from en->de->en (could try different translations)
        aug_btrans = naw.BackTranslationAug()
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
        for i in range(3):
            aug_d = copy.deepcopy(data)
            # Random Deletion
            aug_d[:,0] = aug_del.augment(data[:,0]) # only for title for now!
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
            # Random Swap
            aug_d[:,0] = aug_swap.augment(data[:,0])
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
            # Back Translation from en->de->en
            aug_d[:,0] = aug_btrans.augment(data[:,0])
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
            # Synonym Replacement using WordNet
            aug_d[:,0] = aug_syn.augment(data[:,0])
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
            # Antonym Augmentation
            aug_d[:,0] = aug_an.augment(data[:,0])
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
            # Contextual Word Embedding Augmentation Substitution
            aug_d[:,0] = aug_con_sub.augment(data[:,0])
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
            # Contextual Word Embedding Augmentation Insertion
            aug_d[:,0] = aug_con_in.augment(data[:,0])
            augmented_data = augmented_data + aug_d
            augmented_labels = augmented_labels + labels
        
        print("Augmented sequences: "+str(len(augmented_data)))
        # add augmented sequences to data
        data = np.vstack((data,augmented_data))
        labels = np.vstack((labels,augmented_labels))
        '''
    
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
