# Paper Classification -> Cites/Year

from dataset import create_labels_year, create_labels_reg, create_labels_key, match_data_to_images, remove_empty_fig
from Data_cleaning import pre_tokenize_title, pre_tokenize_abstract
from Paper_dataset import PaperDataset
from hyperparameter_tuning import random_search, grid_search
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot
from sklearn.metrics import f1_score
from pytorch_lightning.callbacks import ModelCheckpoint
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
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
nltk.download('punkt')
nltk.download('wordnet') # needed for synonyms in augmentation
nltk.download('omw-1.4') # needed for synonyms in augmentation
nltk.download('averaged_perceptron_tagger') # needed for synonyms in augmentation
import os
import torch
import warnings
import argparse
import math
from PIL import Image
from torchvision import transforms

from models import Bert, CNN
from models import Regressor, Cite_Classifier

    
def load_data(hparams):
    data_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
    #data_root = os.path.join(data_path, "Data/Data_large_arXiv.csv")
    #data_root = os.path.join(data_path, "Data/Profile_Paper_Data.csv")
    #data_root = os.path.join(data_path, "Data/Data_large_arXiv_Complete.csv")
    #data_root = os.path.join(data_path, "Data/Data_large_arXiv_Images_Test.csv")
    data_root = os.path.join(data_path, "Data/Scientific_Paper_Dataset.csv")
    dataset = pd.read_csv(data_root)
    
    data = dataset.to_numpy()      
    
    data = remove_empty_fig(data)
    
    # Labels:
    labels, data, id2tag = create_labels_reg(data)
    if type(labels[0])==float:
      num_labels = 1
    else:
      num_labels = len(labels[0])
    
    # Input Features:
    data2 = data
    labels2 = labels
    #data = pre_tokenize_title(data2)
    #data = pre_tokenize_abstract(data2)
    print("Length of Data: "+ str(len(data2)))
    
    X_train, X_test, y_train, y_test = train_test_split(data2[:,(0,7,16,17)], labels2, test_size=0.2, random_state=1) # set to (0,7,16) for images
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
    
    X_train = X_train[:int(len(X_train)/1)]
    X_val = X_val[:int(len(X_val)/1)]
    y_train = y_train[:int(len(y_train)/1)]
    y_val = y_val[:int(len(y_val)/1)]
    X_test = X_test[:int(len(X_test)/1)]
    y_test = y_test[:int(len(y_test)/1)]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print("Size of Training Data: "+ str(len(X_train)))
    print("Size of Validation Data: "+ str(len(X_val)))
    print("Size of Test Data: "+ str(len(X_test)))
    
    """Label Distribution:"""
    print("Label Distribution in Total")
    print(np.sum(labels2,axis=0)/len(labels2))
    print("Label Distribution in Training")
    print(np.sum(y_train,axis=0)/len(y_train))
    print("Label Distribution in Validation")
    print(np.sum(y_val,axis=0)/len(y_val))
    
    train_data = PaperDataset(data=X_train, labels=y_train, model = hparams['model'], hparams=hparams, data_type = "train", input_type = hparams['input'])
    val_data = PaperDataset(data=X_val, labels=y_val, model = hparams['model'], hparams=hparams, data_type = "val", input_type = hparams['input'])
    test_data = PaperDataset(data=X_test, labels=y_test, model = hparams['model'], hparams=hparams, data_type = "test", input_type = hparams['input'])
    
    hparams["n_hidden_out"] = num_labels
    hparams["id2tag"] = id2tag
    hparams["Train_data_length"] = len(X_train)
    hparams["Val_data_length"] = len(X_val)
    
    return train_data, val_data, test_data, X_test, y_test, hparams

def train_model(hparams, train_data, val_data, test_data, opt, X_test, y_test):
    results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
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
        cnn_fig = CNN(hparams)
    else:
        cnn_fig = nn.Identity()
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(opt.gpu)
    # num of validation steps for full epoch on val set:
    full_val_epoch = round(len(val_data)/hparams['batch_size'])
    print(train_data[0])


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
        fig_cnn = CNN(hparams)
    else:
        fig_cnn = nn.Identity()
        
    classifier = Cite_Classifier(hparams=hparams, bert_t=bert_t, bert_a=bert_a, cnn=cnn, fig_cnn=fig_cnn, id2tag=hparams["id2tag"], train_set=train_data, val_set=val_data, test_set=test_data)
    
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    logger = TensorBoardLogger("cite_logs")
    
    results = []

    warnings.filterwarnings('ignore')
    classifier = classifier.to(device)
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=3, verbose=False, mode="min")
    best_checkpoint = ModelCheckpoint(monitor='val_f1_macro', save_top_k=1, mode="max")
    logger = TensorBoardLogger("cite_logs")
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        gpus=1 if torch.cuda.is_available() else None,
        #distributed_backend='dp',
        callbacks=[early_stop_callback, best_checkpoint],
        num_sanity_val_steps = 0,
        #logger=True,
        #log_every_n_steps=50,
        logger=logger,
        accumulate_grad_batches=hparams["acc_grad"],
    )
    trainer.validate(classifier, dataloaders=val)
    trainer.fit(classifier) # train the standard classifier 
    final_model_path = trainer.checkpoint_callback.best_model_path # load best model checkpoint
    result = trainer.validate(ckpt_path=final_model_path, dataloaders=val) 
    classifier.load_state_dict(torch.load(final_model_path)['state_dict'], strict=True)
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
    else:
        input_ids_t = None
        masks_a = None
    if "abstract" in hparams["input"]:
        encodings_a = tokenizer(list(X_test[:100,1]), is_split_into_words=False, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
        encodings_a.pop("offset_mapping")
        input_ids_a = torch.LongTensor(encodings_a['input_ids']).reshape(len(encodings_a['input_ids']), len(encodings_a['input_ids'][0]))
        masks_a = torch.LongTensor(encodings_a['attention_mask']).reshape(len(encodings_a['input_ids']), len(encodings_a['input_ids'][0]))
    else:
        input_ids_a = None
        masks_a = None
    if "image" in hparams["input"]:
        image_paths_file = os.path.join(Path(os.path.dirname(os.path.abspath(os.getcwd()))), "Data/Images/")
        root_dir_name = os.path.dirname(image_paths_file) # need to define image_paths
        to_tensor = transforms.ToTensor()
        for i in range(len(X_test[:100])):
            location = str(image_paths_file)+str(X_test[i,2])
        
            img = Image.open(location).convert('RGB')
            img = resize_img(hparams, img, downscale=True)

            img = to_tensor(img)
            # remove arXiv description to avoid passing the labels via the input:
            img = remove_arXiv_text(hparams, img)
            img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
            if i==0:
                imgs = img
            else:
                imgs = torch.cat((imgs, img))
    else:
        imgs = None
        
    label = np.array(y_test[:100])
    out = classifier.forward(input_ids_t, masks_t, input_ids_a, masks_a, imgs, None, None)
    print(out.shape)
    # for classification:
    preds = out.argmax(axis=1)
    predictions = torch.zeros((out.shape[0], out.shape[1]))
    for j in range(preds.shape[0]):
        val = preds[j]
        predictions[j,val] = 1
    
    preds = predictions
        
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
    
    with open('Cite_Inference.txt', 'a') as f:
        f.write(str(pred)+"\n")
        f.write(str(tar)+"\n")
        f.write(str(hparams["id2tag"])+"\n")
        f.write(str(results_test[0]))
        
                    
    return results
            

def resize_img(hparams, img, downscale=True):
    
    if downscale:
        new_size_ratio = hparams["img_sz_ratio"] # reduce size
        img = img.resize((int(img.size[0] * new_size_ratio), int(img.size[1] * new_size_ratio)), Image.ANTIALIAS)
        
        #center_crop = transforms.CenterCrop(842)
        width, height = img.size   # Get current dimensions
        new_width, new_height = int(612*new_size_ratio), int(842*new_size_ratio) # ensures that no information is left out
        
        left = math.ceil((width - new_width)/2)
        top = math.ceil((height - new_height)/2)
        right = math.ceil((width + new_width)/2)
        bottom = math.ceil((height + new_height)/2)
        
    img = img.crop((left, top, right, bottom))    
    
    return img
    
def remove_arXiv_text(hparams, img):
    #trans = transforms.ToPILImage() # for converting back to Image from Tensor
    new_size_ratio = hparams["img_sz_ratio"]
    if new_size_ratio==1.0:
        img[:,120:700,13:37] = torch.ones([3,580,24])
    else:
        img_h_s = int(new_size_ratio*120)
        img_h_e = int(new_size_ratio*700)
        img_w_s = int(new_size_ratio*12)
        img_w_e = int(new_size_ratio*38)
        cover_h = int(new_size_ratio*580)
        cover_w = int(new_size_ratio*26)
        img[:,img_h_s:img_h_e,img_w_s:img_w_e] = torch.ones([3,cover_h,cover_w])
    
    return img

def parser_init():
    
    # Settings
    parser = argparse.ArgumentParser(
        description='Hyperparameters for Publication Year Prediction',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mtl', default=False, type=bool, required=False,
                        help='Multi-tasking for key and year')
    parser.add_argument('--n_hidden_1', type=int, required=False, default=768,
                        help='Num of hidden units in first layer of classifier')
    parser.add_argument('--model', type=str, required=False, default="allenai/scibert_scivocab_uncased",
                        help="Pytorch description for text model like e.g. distilbert-base-uncased")
    parser.add_argument('--batch_size', type=int, required=False, default=2,
                        help='Batch size!')
    parser.add_argument('--lr', type=float, required=False, default=4e-5,
                        help='Learning Rate!')
    parser.add_argument('--drop', type=float, required=False, default=0.1,
                        help='Dropout of classifier')
    parser.add_argument('--bdrop', type=float, required=False, default=0.1,
                        help='Dropout of BERT model')
    parser.add_argument('--freeze_emb', type=bool, required=False, default=False,
                        help='Freeze embedding layer of BERT')
    parser.add_argument('--freeze', type=int, required=False, default=0,
                        help='How many layers of BERT to freeze e.g. 0 means no additional layer frozen and 1 means first attention layer.')
    parser.add_argument('--reg', type=float, required=False, default=1e-7,
                        help='Weight decay')
    parser.add_argument('--gamma', type=float, required=False, default=1,
                        help='Gamma value for Exponential LR-Decay')
    parser.add_argument('--acc_grad', type=int, required=False, default=64,
                        help='Set to 1 if no gradient accumulation is desired else enter desired value for accumulated batches before gradient step')
    parser.add_argument('--input', type=str, required=False, default="title+abstract+image",
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
    train_data, val_data, test_data, X_test, y_test, hparams = load_data(hparams)
    results = []
    results_sub = train_model(hparams,train_data, val_data, test_data, opt, X_test, y_test)
    results = results + results_sub
    tuning_value = 1
    res_frame = pd.DataFrame(results)
    res_frame.to_csv(str(os.getcwd()) + '/Tuning_results_'+str(hparams['input'])+'_'+str(hparams['batch_size'])+'_'+str(tuning_value)+'.csv', index=False)
   