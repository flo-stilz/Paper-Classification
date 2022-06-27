# Paper Classification -> Cites/Year

from dataset import create_labels_year, create_labels_reg, create_labels_key
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
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import TensorBoardLogger
nltk.download('punkt')

data_path = Path(os.path.dirname(os.path.abspath(os.getcwd())))
data_root = os.path.join(data_path, "Data/Data_large.csv")
dataset = pd.read_csv(data_root)

data = dataset.to_numpy()

# Labels:
labels = create_labels_reg(data)
if type(labels[0])==float:
  num_labels = 1
else:
  num_labels = len(labels[0])

# Input Features:
# only use abstracts for now
#data = pre_tokenize_abstract(data)
data = pre_tokenize_title(data)
data2 = data
labels2 = labels
print("Length of Data: "+ str(len(data)))
X_train, X_test, y_train, y_test = train_test_split(data2[:,(0,7)], labels2, test_size=0.2, random_state=1)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("Length of Training Data: "+ str(len(X_train)))


"""Label Distribution:"""
print("Label Distribution")
print(np.sum(labels,axis=0)/len(labels))

########################################################################
# TODO: Define your hyper parameters here!                             #
########################################################################
#'allenai/longformer-base-4096'  # has 4096 input tokens -> good for long input sequences
#'allenai/scibert_scivocab_uncased' #for scientific papers especially computer science related
hparams = {
    "model": 'distilbert-base-cased',
    "n_hidden_1": 768,
    "n_hidden_2": 0,
    "n_hidden_out": num_labels,
    "batch_size": 128,
    "lr": 1e-5,
    "drop":0.0,
    "reg":1e-7,
    "freeze":0,
    "b_drop":0.0,
    "gamma":0.9,
    "data_length": len(data2),
    "aug_prob_words": 0.3,
    "aug_prob_seq": 1.0,
    "aug_rs_prob": 0.7,
    "aug_ri_prob": 0.7,
    "aug_rsr_prob": 0.7,
    "aug_rs_amount": 5,
    "aug_ri_amount": 5,
    "aug_rsr_amount": 5,
    "aug_amount": 0, # 0 for no augmentation
    "input": "title",
}

train_data = PaperDataset(data=X_train, labels=y_train, model = hparams['model'], hparams=hparams, data_type = "train", input_type = hparams['input'])
val_data = PaperDataset(data=X_val, labels=y_val, model = hparams['model'], hparams=hparams, data_type = "val", input_type = hparams['input'])
test_data = PaperDataset(data=X_test, labels=y_test, model = hparams['model'], hparams=hparams, data_type = "test", input_type = hparams['input'])
# num of validation steps for full epoch on val set:
full_val_epoch = round(len(val_data)/hparams['batch_size'])
print(train_data[0])

"""Tensorboard setup:"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs --port 6005

"""# Model Architecture:"""

from models import Bert
from models import Regressor, Cite_Classifier

########################################################################
#                           END OF YOUR CODE                           #
########################################################################
train = DataLoader(train_data, batch_size=hparams['batch_size'], shuffle=True)
val = DataLoader(val_data, batch_size=hparams['batch_size'], shuffle=True, drop_last=True)
test = DataLoader(test_data, batch_size=hparams['batch_size'], shuffle=True)


bert = Bert(hparams)
classifier = Cite_Classifier(hparams, bert, train_data, val_data, test_data)
#classifier = nn.DataParallel(classifier)

os.environ["CUDA_VISIBLE_DEVICES"]="2"
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

print(classifier)

"""Fit Classifier"""

classifier = classifier.to(device)
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=2, verbose=False, mode="min")
trainer = None
logger = TensorBoardLogger("cite_logs")
trainer = pl.Trainer(
    max_epochs=500,
    gpus=1 if torch.cuda.is_available() else None,
    #distributed_backend='dp',
    callbacks=[early_stop_callback],
    num_sanity_val_steps = 0,
    #logger=True,
    #log_every_n_steps=50,
    logger=logger
)
trainer.validate(classifier, dataloaders=val)
trainer.fit(classifier) # train the standard classifier
#print("Validation accuracy when training from scratch: {}%".format(regressor.getAcc(regressor.val_dataloader())[1]*100))
'''
best_model, results = random_search(
    train_data, val_data, test_data, 
    random_search_spaces = {
        "lr": ([8e-5, 8e-5], 'log'),
        "reg": ([1e-7, 1e-7], 'log'),
        "freeze": ([0, 5], 'int'),
        "drop": ([0.0, 0.0], 'float'),
        #"batch_size": ([8, 16, 32, 64, 128], "item"),
        "b_drop": ([0.0,0.0], 'float'),
        #"n_hidden_1": ([100, 1500], 'int'),
        "gamma": ([0.4, 0.7], 'float'),
        #"loss_func": ([nn.CrossEntropyLoss()], "item")
    },
    hparams=hparams,
    num_search = 30, epochs=500, patience=5)

'''
'''
def objective(trial: optuna.trial.Trial) -> float:

    # We optimize the number of layers, hidden units in each layer and dropouts.
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    output_dims = [
        trial.suggest_int("n_units_l{}".format(i), 4, 128, log=True) for i in range(n_layers)
    ]

    trainer = pl.Trainer(
        logger=True,
        checkpoint_callback=False,
        max_epochs=500,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_loss")],
    )
    hyperparameters = dict(n_layers=n_layers, dropout=dropout, output_dims=output_dims)
    trainer.logger.log_hyperparams(hyperparameters)
    trainer.fit(classifier)

    return trainer.callback_metrics["val_acc"].item()
objective()
'''
"""Inference:"""

classifier.eval()
tokenizer = AutoTokenizer.from_pretrained(hparams['model'])
encodings = tokenizer(list(X_test[:25,0]), is_split_into_words=True, return_offsets_mapping=True, padding=True, truncation=True, max_length=512)
encodings.pop("offset_mapping")
print(encodings)
input_ids = torch.LongTensor(encodings['input_ids']).reshape(len(encodings['input_ids']), len(encodings['input_ids'][0]))
print(input_ids.shape)
masks = torch.LongTensor(encodings['attention_mask']).reshape(len(encodings['input_ids']), len(encodings['input_ids'][0]))
label = np.array(y_test[:25])
out = classifier.forward(input_ids, masks)
print(out.shape)
# for classification:
#print(out.argmax(1))
'''
preds = []
for i in range(0,len(out)):
  pred = []
  for j in range(0,len(out[0])):
    if out[i,j]>0.5:
      pred.append(j)
  preds.append(pred)
'''
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
'''
# for regression:
#print(out)
#print(label.argmax(1))
targets = torch.LongTensor(label).reshape(len(label[0]), 1)
print(targets)
loss_func = nn.MSELoss()
loss = loss_func(out, targets)
print(loss)
'''
