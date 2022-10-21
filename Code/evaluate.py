# Evaluation of Paper Classification models

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
from pytorch_lightning.loggers import TensorBoardLogger
import warnings
from sklearn.model_selection import ShuffleSplit # or StratifiedShuffleSplit
import sys
import argparse
import warnings
from argparse import Namespace

from models import Bert, CNN
from models import Key_Classifier, MTL_YK_Classifier, Cite_Classifier, Year_Classifier, Baseline

def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)

def load_data(hparams, mtl, opt):
    data_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
    data_root = os.path.join(data_path, "Data/Scientific_Paper_Dataset.csv")
    dataset = pd.read_csv(data_root)
    
    data = dataset.to_numpy()  
        
    data = remove_empty_fig(data)
                                       
    if mtl:
        labels, data, id2tag_year, id2tag_key = create_labels_mtl_year_key(data)
    elif opt.task=="key":
        labels, data, id2tag = create_labels_key(data)
    elif opt.task=="cite":
        labels, data, id2tag = create_labels_reg(data)
    elif opt.task=="year":
        labels, data, id2tag = create_labels_year(data)


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

    # Max Class label for baseline based on train set:
    print(np.sum(y_train,axis=0))
    max_class = largest_indices(np.sum(y_train,axis=0),1)
    print(max_class)

    # Train data not needed
    X_train = X_train[:int(len(X_train)/1000)]
    X_val = X_val[:int(len(X_val)/1)]
    y_train = y_train[:int(len(y_train)/1000)]
    y_val = y_val[:int(len(y_val)/1)]
    # small Test for now:
    X_test = X_test[:int(len(X_test)/1)]
    y_test = y_test[:int(len(y_test)/1)]

    print("Length of actually used Data: "+ str(len(X_val)+len(X_train)))
    print("Length of Training Data: "+ str(len(X_train)))
    print("Length of Validation Data: "+ str(len(X_val)))
    print("Length of Test Data: "+ str(len(X_test)))

    """Label Distribution:"""
    print("Label Distribution in Test set")
    print(np.sum(y_test,axis=0)/len(y_test))
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
    hparams["Train_data_length"] = len(X_train)
    hparams["Val_data_length"] = len(X_val)
    
    
    return train_data, val_data, test_data, hparams, id2tag, id2tag_year, id2tag_key, X_test, y_test, max_class
    
def evaluate_model(hparams, train_data, val_data, test_data, opt, id2tag, id2tag_year, id2tag_key, X_test, y_test, max_class):

    PATH = os.path.join(os.getcwd(), opt.model_path)

    """# Model Architecture:"""
    train = DataLoader(train_data, batch_size=hparams['batch_size'], shuffle=True)
    val = DataLoader(val_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)
    test = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=False, drop_last=True)

    # Pretrained models Initialization:
    print(hparams["input"])
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
        fig_cnn = CNN(hparams)
    else:
        fig_cnn = nn.Identity()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    os.environ["CUDA_VISIBLE_DEVICES"]="3"

    if hparams["mtl"] and opt.task=="key":
        classifier = MTL_YK_Classifier(hparams, bert_t=bert_t, bert_a=bert_a, cnn=cnn, id2tag_year=id2tag_year, id2tag_key=id2tag_key, train_set=train_data, val_set=val_data, test_set=test_data)
    elif opt.task=="key":
        classifier = Key_Classifier(hparams, bert_t=bert_t, bert_a=bert_a, cnn=cnn, id2tag=id2tag, train_set=train_data, val_set=val_data, test_set=test_data)
    elif opt.task=="year":
        classifier = Year_Classifier(hparams, bert_t=bert_t, bert_a=bert_a, cnn=cnn, cnn_fig=fig_cnn, id2tag=id2tag, train_set=train_data, val_set=val_data, test_set=test_data)
    elif opt.task=="cites":
        classifier = Cite_Classifier(hparams, bert_t=bert_t, bert_a=bert_a, cnn=cnn, fig_cnn=fig_cnn, id2tag=id2tag, train_set=train_data, val_set=val_data, test_set=test_data)
        
    if opt.baseline:
        classifier = Baseline(hparams, max_class, opt.task, train_data, val_data, test_data)
        
    classifier = classifier.to(device)
    print(classifier)
    # Create Trainer
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
    trainer = pl.Trainer(weights_summary=None,
                         gpus=1 if torch.cuda.is_available() else None,
                         logger=logger,
                         checkpoint_callback=False)
    
    checkpoint = torch.load(PATH)
    model_weights = checkpoint["state_dict"]
    
    if not opt.baseline:
        print(classifier.state_dict()['classifier.1.weight'])
        print(model_weights['classifier.1.weight'])
        print(model_weights['bert_t.bert.encoder.layer.6.output.dense.weight'])
        print(classifier.state_dict()['bert_t.bert.encoder.layer.6.output.dense.weight'])
        #classifier = Key_Classifier.load_from_checkpoint(PATH, bert_t=bert_t, bert_a=bert_a, cnn=cnn)
        classifier.load_state_dict(model_weights,strict=True)
        print(classifier.state_dict()['classifier.1.weight'])
        print(classifier.state_dict()['bert_t.bert.encoder.layer.6.output.dense.weight'])
    
    classifier.eval()
    
    print("Performance on Validation set")
    trainer.validate(classifier, dataloaders=val)
    print("Performance on Test set")
    trainer.validate(classifier, dataloaders=test)


def parser_init():
    
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Keyword/Area Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mtl', default=False, type=bool, required=False,
                        help='Multi-tasking for key and year')
    parser.add_argument('--task', type=str, required=True,
                        help='key, year, or cites')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model to evaluate')
    parser.add_argument('--n_hidden_1', type=int, required=False, default=768,
                        help='Num of hidden units in first layer of classifier')
    parser.add_argument('--model', type=str, required=False, default="allenai/scibert_scivocab_uncased",
                        help="Pytorch description for text model like e.g. distilbert-base-uncased")
    parser.add_argument('--batch_size', type=int, required=False, default=2,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=1e-4,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of classifier')
    parser.add_argument('--bdrop', type=int, required=False, default=0.1,
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
    parser.add_argument('--baseline', type=bool, required=False, default=False,
                        help='Use basic baseline for given task')
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
    train_data, val_data, test_data, hparams, id2tag, id2tag_year, id2tag_key, X_test, y_test, max_class = load_data(hparams, hparams['mtl'], opt)

    evaluate_model(hparams, train_data, val_data, test_data, opt, id2tag, id2tag_year, id2tag_key, X_test, y_test, max_class)