# Paper Classification Keywords/Area


from dataset import create_labels_year, create_labels_reg, create_labels_key, over_sampling, match_data_to_images, remove_empty_fig, create_labels_mtl_year_key
from Data_cleaning import pre_tokenize_title, pre_tokenize_abstract
from Paper_dataset import PaperDataset
from hyperparameter_tuning import random_search, grid_search
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.metrics import f1_score
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from transformers import AutoTokenizer
import torch.nn as nn
import nltk
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import warnings
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
import sys
import argparse
import warnings

from models import Bert, CNN
from models import Key_Classifier, MTL_YK_Classifier

nltk.download('punkt')
nltk.download('wordnet') # needed for synonyms in augmentation
nltk.download('omw-1.4')

def load_data(hparams, mtl):
    data_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
    #data_root = os.path.join(data_path, "Data/Data_large_arXiv.csv")
    #data_root = os.path.join(data_path, "Data/Data_large_arXiv_Complete.csv")
    data_root = os.path.join(data_path, "Data/Scientific_Paper_Dataset.csv")
    #data_root = os.path.join(data_path, "Data/Data_large_arXiv_Images_Test.csv")
    dataset = pd.read_csv(data_root)
    
    data = dataset.to_numpy()  
        
    data = remove_empty_fig(data)
                                       
    if mtl:
        labels, data, id2tag_year, id2tag_key = create_labels_mtl_year_key(data)
    else:
        labels, data, id2tag = create_labels_key(data)


    if type(labels[0])==float:
      num_labels = 1
    else:
      num_labels = len(labels[0])
    
    if mtl:
        num_labels_year = len(id2tag_year)
        num_labels_key = len(id2tag_key)
        id2tag = None
    else:
        num_labels_year = 0
        num_labels_key = 0
        id2tag_year = None
        id2tag_key = None
    
    # only use abstracts for now
    #data = pre_tokenize_abstract(data)
    #data = pre_tokenize_title(data)
    data2 = data
    labels2 = np.array(labels)
    
    print("Length of Data: "+ str(len(data)))
    X_train, X_test, y_train, y_test = train_test_split(data2[:,(0,7,16,17)], labels2, test_size=0.2, random_state=1) # set to (0,7,16) for images
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

    X_train = X_train[:int(len(X_train)/1)]
    X_val = X_val[:int(len(X_val)/1)]
    y_train = y_train[:int(len(y_train)/1)]
    y_val = y_val[:int(len(y_val)/1)]
    X_test = X_test[:int(len(X_test)/1)]
    y_test = y_test[:int(len(y_test)/1)]


    print("Length of actually used Data: "+ str(len(X_val)+len(X_train)))
    print("Length of Training Data: "+ str(len(X_train)))
    print("Length of Validation Data: "+ str(len(X_val)))

    """Label Distribution:"""
    print("Label Distribution in Training set")
    print(np.sum(y_train,axis=0)/len(y_train))
    print("Label Distribution in Validation set")
    print(np.sum(y_val,axis=0)/len(y_val))
    
    train_data = PaperDataset(data=X_train, labels=y_train, model = hparams['model'], hparams=hparams, data_type = "train", input_type = hparams['input'])
    val_data = PaperDataset(data=X_val, labels=y_val, model = hparams['model'], hparams=hparams, data_type = "val", input_type = hparams['input'])
    test_data = PaperDataset(data=X_test, labels=y_test, model = hparams['model'], hparams=hparams, data_type = "test", input_type = hparams['input'])
    # num of validation steps for full epoch on val set:
    full_val_epoch = round(len(val_data)/hparams['batch_size'])
    print(train_data[0])
    
    hparams["n_hidden_out"] = num_labels
    hparams["num_labels_year"] = num_labels_year
    hparams["num_labels_key"] = num_labels_key
    hparams["id2tag"] = id2tag
    hparams["id2tag_year"] = id2tag_year
    hparams["id2tag_key"] = id2tag_key
    hparams["Train_data_length"] = len(X_train)
    hparams["Val_data_length"] = len(X_val)
    
    return train_data, val_data, test_data, X_test, y_test, hparams
    
def train_model(hparams, train_data, val_data, test_data, opt, X_test, y_test):
    results = []

    """# Model Architecture:"""
    train = DataLoader(train_data, batch_size=hparams['batch_size'], shuffle=True)
    val = DataLoader(val_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)
    test = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)


    # Pretrained models Initialization:
    if "title" in hparams["input"]:
        bert_t = Bert(hparams)
    else:
        bert_t = nn.Identity()
    if "abstract" in hparams["input"]:
        bert_a = Bert(hparams)
    else:
        bert_a = nn.Identity()
    if "image" in hparams["input"]:
        cnn = CNN(hparams)
    else:
        cnn = nn.Identity()
    if "figures" in hparams["input"]:
        cnn = CNN(hparams)
    else:
        cnn = nn.Identity()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
    
    if hparams["mtl"]:
        classifier = MTL_YK_Classifier(hparams, bert_t, bert_a, cnn, hparams["id2tag_year"], hparams["id2tag_key"], train_data, val_data, test_data)
    else:
        classifier = Key_Classifier(hparams, bert_t, bert_a, cnn, hparams["id2tag"], train_data, val_data, test_data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    classifier = classifier.to(device)
    print(classifier)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=5, verbose=False, mode="min")
    if hparams["mtl"]:
        best_checkpoint = ModelCheckpoint(monitor='val_f1_macro_k', save_top_k=1, mode="max")
    else:
        best_checkpoint = ModelCheckpoint(monitor='val_f1_macro', save_top_k=1, mode="max")
    trainer = None
    if hparams["input"] == "title":
        logger = TensorBoardLogger("tb_logs")
    elif hparams["input"] == "abstract":
        logger = TensorBoardLogger("tb_abs_logs")
    elif hparams["input"] == "title+abstract":
        logger = TensorBoardLogger("tb_t_abs_logs")
    elif hparams["input"] == "image":
        logger = TensorBoardLogger("tb_image_logs")
    elif hparams["input"] == "figures":
        logger = TensorBoardLogger("tb_figures_logs")
    elif hparams["input"] == "title+abstract+image":
        logger = TensorBoardLogger("tb_text_image_logs")
    elif hparams["input"] == "title+abstract+figures":
        logger = TensorBoardLogger("tb_text_figures_logs")
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        #distributed_backend='dp',
        callbacks=[early_stop_callback, best_checkpoint],
        num_sanity_val_steps = 0,
        #logger=True,
        #log_every_n_steps=50,
        logger=logger,
        #accumulate_grad_batches=hparams["acc_grad"],
    )
    trainer.validate(classifier, dataloaders=val)
    trainer.fit(classifier) # train the standard classifier 
    final_model_path = trainer.checkpoint_callback.best_model_path # load best model checkpoint
    result = trainer.validate(ckpt_path=final_model_path, dataloaders=val)
    classifier.load_state_dict(torch.load(final_model_path)['state_dict'], strict=True)
    trainer.validate(classifier, dataloaders=val)
    hp_results = hparams.copy()
    if hparams["mtl"]:
        hp_results['val_f1_macro_y'] = result[0]['val_f1_macro_y']
        hp_results['val_f1_micro_y'] = result[0]['val_f1_micro_y']
        hp_results['val_loss'] = result[0]['val_loss']
        hp_results['val_acc_y'] = result[0]['val_acc_y']
        hp_results['val_f1_macro_k'] = result[0]['val_f1_macro_k']
        hp_results['val_f1_micro_k'] = result[0]['val_f1_micro_k']
        hp_results['val_acc_k'] = result[0]['val_acc_k']
        print(result[0]['val_f1_macro_k'])
    else:
        hp_results['val_f1_macro'] = result[0]['val_f1_macro']
        hp_results['val_f1_micro'] = result[0]['val_f1_micro']
        hp_results['val_loss'] = result[0]['val_loss']
        hp_results['val_acc'] = result[0]['val_acc']
        print(result[0]['val_f1_macro'])
    results.append(hp_results)
    
    results_test = trainer.validate(classifier, dataloaders=test)
    
    # Inference:
    classifier.eval()
    tokenizer = AutoTokenizer.from_pretrained(hparams['model'])
    if "title" in hparams["input"]:
        print(list(X_test[:100,0]))
        encodings_t = tokenizer(list(X_test[:100,0]), is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
        encodings_t.pop("offset_mapping")
        print(encodings_t)
        input_ids_t = torch.LongTensor(encodings_t['input_ids']).reshape(len(encodings_t['input_ids']), len(encodings_t['input_ids'][0]))
        print(input_ids_t.shape)
        masks_t = torch.LongTensor(encodings_t['attention_mask']).reshape(len(encodings_t['input_ids']), len(encodings_t['input_ids'][0]))
        
    if "abstract" in hparams["input"]:
        encodings_a = tokenizer(list(X_test[:100,1]), is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
        encodings_a.pop("offset_mapping")
        input_ids_a = torch.LongTensor(encodings_a['input_ids']).reshape(len(encodings_a['input_ids']), len(encodings_a['input_ids'][0]))
        masks_a = torch.LongTensor(encodings_a['attention_mask']).reshape(len(encodings_a['input_ids']), len(encodings_a['input_ids'][0]))
    else:
        input_ids_a = None
        masks_a = None
        
    label = np.array(y_test[:100])
    out = classifier.forward(input_ids_t, masks_t, input_ids_a, masks_a, None, None, None)
    print(out.shape)
    # for classification:
    preds = torch.tensor(out.detach().clone() > 0.5, dtype=float).detach().clone()
    tar = []
    for i in range(0,len(label)):
      t = []
      for j in range(0,len(label[0])):
        if label[i,j]==1:
          t.append(j)
      tar.append(t)

    pred = []
    for i in range(0,len(preds)):
      a = []
      for j in range(0,len(preds[0])):
        if preds[i,j]==1:
          a.append(j)
      pred.append(a)
    print(pred)
    #print(preds.argmax(-1))
    print(tar)
    
    with open('Keyword_Inference.txt', 'a') as f:
        f.write(str(pred)+"\n")
        f.write(str(tar)+"\n")
        f.write(str(hparams["id2tag"])+"\n")
        f.write(str(results_test[0]))
        
        
    return results

def parser_init():
    
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Keyword/Area Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mtl', default=False, type=bool, required=False,
                        help='Multi-tasking for key and year')
    parser.add_argument('--n_hidden_1', type=int, required=False, default=768,
                        help='Num of hidden units in first layer of classifier')
    parser.add_argument('--model', type=str, required=False, default="allenai/scibert_scivocab_uncased",
                        help="Pytorch description for text model like e.g. distilbert-base-uncased")
    parser.add_argument('--batch_size', type=int, required=False, default=8,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=2e-5,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.2,
                        help='Dropout of classifier')
    parser.add_argument('--bdrop', type=float, required=False, default=0.1,
                        help='Dropout of BERT model')
    parser.add_argument('--freeze_emb', type=bool, required=False, default=True,
                        help='Freeze embedding layer of BERT')
    parser.add_argument('--freeze', type=int, required=False, default=2,
                        help='How many layers of BERT to freeze e.g. 0 means no additional layer frozen and 1 means first attention layer.')
    parser.add_argument('--reg', type=float, required=False, default=1e-7,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=1,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=1,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--input', type=str, required=False, default="title+abstract",
                        help='Input features: options: title, abstract, image, and figures -> combinations should be combined via + e.g. title+abstract')
    parser.add_argument('--img_sz_ratio', type=float, required=False, default=1.0,
                        help='Ratio for initial image size e.g. 0.5 would resize image to half the width and height.')
    parser.add_argument('--epochs', type=int, required=False, default=10,
                        help='Set maxs number of epochs.')
    parser.add_argument('--gpu', type=int, required=True,
                        help='Set the name of the GPU in the system')
    opt = parser.parse_args()
    hparams = { # title ... # abstract: ...
        "model": opt.model,
        "n_hidden_1": opt.n_hidden_1, # 170000 for both so far
        "2_layer": False,
        "3_layer": False, # False # abs: ?
        "n_hidden_2": 0,
        "mtl": opt.mtl,
        "lstm_hidden": 200,
        "batch_size": opt.batch_size, # 256 # abs: 16
        "all_emb": False, # use all BERT embeddings and pass through LSTM->MLP->output
        "cattn": False,
        "c_head": 2,
        "c_depth": 1,
        "lr": opt.lr,#1.5e-4 for title # abs: 1e-4 -> try lower as well
        "drop": opt.drop, # 0.1 # abs: ?
        "reg":opt.reg,
        "freeze":opt.freeze,
        "freeze_emb": opt.freeze_emb, # freezing embedding layer in bert # False # abs: ?
        "cnn_freeze": 0, # if the first 6 layers shall be frozen then set to 7
        "b_drop":opt.bdrop, # 0.2 # abs: 0.3
        "gamma": opt.gamma,#0.82 #abs: 0.85 -> try more
        "aug_prob_words": 0.0,
        "aug_prob_seq": 1.0,
        "aug_rs_prob": 1.0,
        "aug_ri_prob": 1.0,
        "aug_rsr_prob": 1.0,
        "aug_rs_amount": 0,
        "aug_ri_amount": 0,
        "aug_rsr_amount": 0,
        "aug_amount": 0, # 0 for no augmentation
        "syn_rep": False,
        "an_aug": False,
        "rs": False,
        "rd": False,
        "input": opt.input,
        "acc_grad": opt.acc_grad,
        "img_sz_ratio": opt.img_sz_ratio,
        "max_seq": 4,
        "fig_lstm": False,
        "lstm_num_l": 1,
        "fig_crop_size": 400,
        "t+i_encoder": False,
        "num_enc_lay": 1,
        "fig_enc": False,
        "year_lw": 0.1,
        "key_lw": 1,
    }
    
    return hparams, opt
    
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    hparams, opt = parser_init()
    train_data, val_data, test_data, X_test, y_test, hparams = load_data(hparams, hparams['mtl'])
    results = []
    results_sub = train_model(hparams,train_data, val_data, test_data, opt, X_test, y_test)
    results = results + results_sub
    tuning_value = 1
    res_frame = pd.DataFrame(results)
    res_frame.to_csv(str(os.getcwd()) + '/Tuning_results_'+str(hparams['input'])+'_'+str(hparams['batch_size'])+'_'+str(tuning_value)+'.csv', index=False)
