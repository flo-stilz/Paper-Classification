import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from transformers import BertModel, DistilBertModel, AutoModel, AutoConfig, AutoModelForSequenceClassification
import json
import time
import sys
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from attention import MultiHeadAttention
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, PackedSequence

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score, precision_recall_fscore_support
from argparse import Namespace


class Bert(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        #set Bert
        configuration = AutoConfig.from_pretrained(hparams["model"])
        # For BERT
        if "distilbert-base" in hparams["model"]:
            # For DistilBERT
            configuration.dropout = hparams["b_drop"]
            configuration.attention_dropout = hparams["b_drop"]
        else:
            # For BERT
            configuration.hidden_dropout_prob = hparams["b_drop"]
            configuration.attention_probs_dropout_prob = hparams["b_drop"]
        #configuration.num_labels=hparams["n_hidden_out"] # For SequenceClassificationModel
        
        self.bert = AutoModel.from_pretrained(hparams["model"], config = configuration)
        #self.bert = AutoModelForSequenceClassification.from_pretrained(hparams["model"], config = configuration)
        
        # decide on how many layers to use
        # For BERT
        #self.bert.encoder.layer = self.bert.encoder.layer[:hparams['num_layers']]
        # For DistilBERT
        #self.bert.transformer.layer = self.bert.transformer.layer[:hparams['num_layers']]
        # freeze some of the BERT weights:
        if hparams['freeze_emb'] and "distil" in hparams["model"]:
            modules = [self.bert.embeddings, *self.bert.transformer.layer[:hparams["freeze"]]] 
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        elif hparams['freeze_emb']:
            modules = [self.bert.embeddings, *self.bert.encoder.layer[:hparams["freeze"]]] 
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        
    def forward(self, input_ids, mask):
        # feed x into BERT model!
        if "distilbert-base" in self.hparams["model"]:
            # For DistilBERT
            hidden_state = self.bert(input_ids=input_ids, attention_mask=mask,return_dict=False)
            output = hidden_state[0][:,0]
        else:
            # For BERT uses final hidden_state for now
            hidden_state, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask,return_dict=False)
            #output = pooled_output
            if self.hparams["all_emb"]:
                output = hidden_state[:,:]
            else:
                output = hidden_state[:,0]
        
        return output
    

class CNN(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        #self.cnn = models.vgg16(pretrained=True) # VGG 16 # 71.592% on ImageNet 1K
        #self.cnn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # ResNet50 #80.858% on ImageNet 1K
        self.cnn = models.resnet50(pretrained=True) # ResNet50 #80.858% on ImageNet 1K
        #self.cnn = model.efficientnet_v2_m(weights=models.EfficientNet_V2_M_Weights) # EfficientNet_V2_M #	85.112% on ImageNet 1K
        # remove the last three layers:
        #self.cnn = nn.Sequential(*list(self.cnn.children())[:-3])
        # freeze weights:
        if self.hparams["cnn_freeze"]>0:
            layer_c = 0
            for layer in self.cnn.children():
                layer_c += 1
                if layer_c < self.hparams["cnn_freeze"]:
                    for param in layer.parameters():
                        param.requires_grad = False
        
        
    def forward(self, x):
        # feed x into CNN model!
        
        N = len(x)
        output = self.cnn(x)
        #print(x.size())
        
        
        return output
    

class Baseline(pl.LightningModule):
    
    # Simple baseline for comparison
    # Only predicts biggest class in dataset for a task
    def __init__(self, hparams, max_class, task, train_set, val_set, test_set):
        super().__init__()
        self.hparams.update(hparams)
        self.max_class = max_class
        self.task = task
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        
        
    def forward(self, bs):
        
        
        output = torch.zeros((bs,self.hparams["n_hidden_out"])).cuda()
        output[:,self.max_class] = 1
        
        return output
    
    def general_step(self, batch, batch_idx, mode):

        # forward pass
        targets = batch['labels']
        bs = len(targets)
        out = self.forward(bs)
        
        # loss
        #targets = targets.argmax(axis=1) # only needed for binary classification

        if self.task=="key":
            
            targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])

            out = out.to(torch.float32)
            # for multi-label classification:
            ######################
            preds = torch.as_tensor(out.detach().clone() > 0.5, dtype=float).detach().clone()
            
            n_correct = ((preds == targets).sum(axis=1)==len(preds[0])).sum()
            #n_correct = torch.Tensor([n_correct])
            #n_correct = torch.Tensor([0])
            ######################
            loss_func = nn.BCELoss()
            loss = loss_func(out, targets.type(torch.float))
            
            f1_micro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='micro')
            f1_macro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='macro')
            labels = targets
        else:
            targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])
            
            out = out.to(torch.float32)
            
            labels = targets
            targets = targets.argmax(axis=1)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(out, targets)
            preds = out.argmax(axis=1) # for multiclass classification
            predictions = torch.zeros((out.shape[0], out.shape[1]))
            for j in range(preds.shape[0]):
                val = preds[j]
                predictions[j,val] = 1
    
            n_correct = (targets == preds).sum()
    
            f1_micro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='micro')
            f1_macro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='macro')

        return loss, n_correct, torch.as_tensor(f1_micro).cuda(), torch.as_tensor(f1_macro).cuda(), preds, labels

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1_micro = torch.stack([x[mode + '_f1_micro'] for x in outputs]).mean()
        f1_macro = torch.stack([x[mode + '_f1_macro'] for x in outputs]).mean()
        
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()

        return avg_loss, acc, f1_micro, f1_macro, preds, targets

    def training_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro, 'val_preds': preds, 'val_targets': targets}

    def test_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct, 'test_f1_micro': f1_micro, 'test_f1_macro': f1_macro}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1_micro, f1_macro, preds, targets = self.general_end(outputs, "val")
         
        
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1-Micro={}".format(f1_micro))
        print("Val-F1-Macro={}".format(f1_macro))
        #cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print("Learning-Rate: "+str(cur_lr))
        self.log("val_loss", avg_loss)
        self.log("val_f1_micro", f1_micro)
        self.log("val_f1_macro", f1_macro)
        self.log("val_acc", acc)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1_micro", f1_micro, self.global_step)
        self.logger.experiment.add_scalar("val_f1_macro", f1_macro, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle=False, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True, pin_memory=True)
    
    def configure_optimizers(self):

        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert_t.parameters()) + list(self.bert_a.parameters()) + list(self.cnn.parameters()) + list(self.fig_cnn.parameters()) + list(self.lstm.parameters())+ list(self.cross_attn.parameters()) + list(self.linear_1.parameters()) + list(self.transformer_encoder.parameters()) + list(self.classifier.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.lr, weight_decay=self.hparams['reg'])
        #optim = torch.optim.SGD(params=params, lr = self.hparams['lr'], weight_decay=self.hparams['reg'],momentum=0.9)
        #optim = nn.optim.RMSprop(params, lr=self.hparams['lr'], alpha=0.99, eps=1e-08, weight_decay=self.hparams['reg'], momentum=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(self.train_dataloader), epochs=8) # epoch number unkown
        return [optim], [scheduler]



class Regressor(pl.LightningModule):

    def __init__(self, hparams, bert, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.bert = bert
        self.model = nn.Identity()
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        
        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        
        self.model = nn.Sequential(
            nn.Dropout(self.hparams["drop"]),
            nn.Linear(768, self.hparams["n_hidden_out"]),
            #nn.Sigmoid(),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_3"])
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, input_ids, masks):
        
        x = self.bert(input_ids, masks)
        # For DistilBert:
        x = x[0]
        x = self.model(x[:,0])
        #x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        
        masks = batch['attention_mask']
        input_ids = batch['input_ids'].squeeze(1)
        targets = batch['labels']

        # forward pass
        out = self.forward(input_ids, masks)
        # loss
        
        targets = targets.reshape(len(targets), 1)

        out = out.to(torch.float32)
        targets = targets.to(torch.float32)
        loss_func = nn.MSELoss()
        loss = loss_func(out, targets)

        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        
        
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        should_log = False
        if batch_idx % self.trainer.accumulate_grad_batches == 0:
            should_log = (
                    (self.global_step + 1) % self.hparams['acc_grad'] == 0
                    )
        if should_log:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)

        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_epoch_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        self.log("val_loss", avg_loss)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle = False, batch_size=self.hparams['batch_size'])
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert.parameters()) + list(self.model.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])

        return optim

class Cite_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert_t, bert_a, cnn, fig_cnn, id2tag, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.id2tag = id2tag
        self.bert_t = bert_t
        self.bert_a = bert_a
        self.cnn = cnn
        self.fig_cnn = fig_cnn
        self.lstm = nn.Identity()
        self.cross_attn = nn.Identity()
        self.linear_1 = nn.Identity()
        self.transformer_encoder = nn.Identity()
        self.id2tag = id2tag
        #self.classifier = nn.Identity()
        self.input_type = hparams["input"]
        self.seq_amount = (hparams["input"]=="title+abstract")+1
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        if "large" in self.hparams['model']:
            self.emb_dim = 1024
        else:
            self.emb_dim = 768
            
        if hparams["input"]=="image": # needs further adjustments later on
            self.emb_dim = 1000
            self.seq_amount = 1
        if hparams["input"]=="figures":
            self.emb_dim = 1000
            self.seq_amount = 1
            if hparams["fig_lstm"]:
                self.emb_dim = self.hparams["lstm_hidden"]*self.hparams["lstm_num_l"]*2

                self.lstm = nn.LSTM(input_size=1000, hidden_size=self.hparams["lstm_hidden"], num_layers=self.hparams["lstm_num_l"], bidirectional=True, batch_first=True)
        if hparams["input"]=="title+abstract+image":
            self.emb_dim = 1000+768*2
            self.seq_amount=1
            
        if hparams["input"]=="title+abstract+image+figures":
            self.emb_dim = 1000*2+768*2
            self.seq_amount=1
        
        if hparams["t+i_encoder"]:
            self.emb_dim = 3*768
            self.linear_1 = nn.Sequential(
                    nn.Dropout(self.hparams["drop"]),
                    nn.Linear(1000, 768),
                    )
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        elif hparams["fig_enc"]:
            encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        
        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        
        self.classifier = nn.Sequential(
            nn.Dropout(self.hparams["drop"]),
            nn.Linear(self.emb_dim*self.seq_amount, self.hparams["n_hidden_out"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_2"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_2"], self.hparams["n_hidden_3"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_3"], self.hparams["n_hidden_4"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_4"], self.hparams["n_hidden_5"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_5"], self.hparams["n_hidden_6"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_6"], self.hparams["n_hidden_7"]),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_7"], self.hparams["n_hidden_out"]),
            #nn.Sigmoid(),
        )
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward_title(self, input_ids, masks):
        
        x1 = self.bert_t(input_ids, masks)
        x = x1
        if self.hparams["all_emb"]:
            x, hidden_state = self.lstm(x)
            hidden_state = hidden_state[0]
            # Flatten hidden state with respect to batch size

            hidden = hidden_state.transpose(1,0).contiguous().view(self.hparams["batch_size"], -1)
            
            x = self.classifier(hidden)
        else:
            x = self.classifier(x)
            
        return x
    
    def forward_abstract(self, input_ids, masks):
        
        x1 = self.bert_a(input_ids, masks)
        x = x1
        x = self.classifier(x)
        #x = self.model(x)
        return x
    
    def forward_title_abstract(self, input_ids_t, masks_t, input_ids_a, masks_a):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        if self.hparams["cattn"]: # use cross-attention to fuse embeddings
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = x2.clone()
            for c in range(self.hparams["c_depth"]):
                x = self.cross_attn[c](x, x1, x1) # x2: abstract emb, x1: title emb
            x = x.reshape(x.shape[0], x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2),1)
        x = self.classifier(x)
        #x = self.model(x)
        return x
    
    def forward_img(self, img):
        
        x = self.cnn(img)
        x = self.classifier(x)
        
        return x
    
    def forward_fig(self, fig, seq_len):
        
        pad_seq = pack_padded_sequence(fig, seq_len.cpu(), batch_first = True, enforce_sorted=False)
        fig = pad_seq.data
        x = self.fig_cnn(fig)
        if self.hparams["fig_lstm"]:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = self.lstm(seq)
            x,_ = pad_packed_sequence(x, batch_first=True)
            # take final hidden states output
            x = x[:,-1]
        elif self.hparams["fig_enc"]:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = pad_packed_sequence(seq, batch_first=True)
            x = self.transformer_encoder(x)
            # pick last seq embedding for output -> might want to try to run LSTM over entire seq
            x = x[:,-1]
        else:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = pad_packed_sequence(seq, batch_first=True)
            # Max Pool accross entire seq for now
            x = torch.max(x, axis=1).values
            
        x = self.classifier(x)
        
        return x
    
    def forward_text_img(self, input_ids_t, masks_t, input_ids_a, masks_a, img):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        x3 = self.cnn(img)
        
        if self.hparams["t+i_encoder"]:
            #x = torch.cat((x1, x2, x3),1)
            x3 = self.linear_1(x3)
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x3 = x3.reshape(x3.shape[0], 1, x3.shape[1])
            x = torch.cat((x1,x2,x3),1)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = self.transformer_encoder(x)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2, x3),1)
        
        x = self.classifier(x)
        
        return x
        
    def forward_text_img_fig(self, input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        x3 = self.cnn(img)
        pad_seq = pack_padded_sequence(fig, seq_len.cpu(), batch_first = True, enforce_sorted=False)
        fig = pad_seq.data
        x4 = self.fig_cnn(fig)
        seq = PackedSequence(x4, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
        x4,_ = pad_packed_sequence(seq, batch_first=True)
        # Max Pool accross entire seq for now
        x4 = torch.max(x4, axis=1).values
        
        if self.hparams["t+i_encoder"]:
            #x = torch.cat((x1, x2, x3),1)
            x3 = self.linear_1(x3)
            x3 = self.linear_1(x4)
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x3 = x3.reshape(x3.shape[0], 1, x3.shape[1])
            x4 = x4.reshape(x4.shape[0], 1, x4.shape[1])
            x = torch.cat((x1,x2,x3,x4),1)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = self.transformer_encoder(x)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2, x3, x4),1)

        x = self.classifier(x)
        
        return x
        
    
    def forward(self, input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len):
        
        if self.input_type=="title":
            x = self.forward_title(input_ids_t, masks_t)
        elif self.input_type=="abstract":
            x = self.forward_abstract(input_ids_a, masks_a)
        elif self.input_type=="title+abstract":
            x = self.forward_title_abstract(input_ids_t, masks_t, input_ids_a, masks_a)
        elif self.input_type=="image":
            x = self.forward_img(img)
        elif self.input_type=="figures":
            x = self.forward_fig(fig, seq_len)
        elif self.input_type=="title+abstract+image":
            x = self.forward_text_img(input_ids_t, masks_t, input_ids_a, masks_a, img)
        elif self.input_type=="title+abstract+image+figures":
            x = self.forward_text_img_fig(input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len)
        
        else:
            AssertionError
            
        return x

    def general_step(self, batch, batch_idx, mode):
        
        if "title" in self.input_type:
            masks_t = batch['attention_mask_t']
            input_ids_t = batch['input_ids_t'].squeeze(1)
        else:
            masks_t = None
            input_ids_t = None
        if "abstract" in self.input_type:
            masks_a = batch['attention_mask_a']
            input_ids_a = batch['input_ids_a'].squeeze(1)
        else:
            masks_a = None
            input_ids_a = None
        if "image" in self.input_type:
            img = batch['img']
        else:
            img = None
        if "figures" in self.input_type:
            fig = batch['fig']
            seq_len = batch['seq_len']
        else:
            fig = None
            seq_len = None
        
        targets = batch['labels']

        # forward pass
        out = self.forward(input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len)
        
        # loss
        #targets = targets.argmax(axis=1) # only needed for binary classification

        targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])
        
        out = out.to(torch.float32)
        
        labels = targets
        targets = targets.argmax(axis=1)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(out, targets)
        #loss_func = nn.BCELoss()
        #loss = loss_func(out, targets.type(torch.float))

        preds = out.argmax(axis=1) # for multiclass classification
        predictions = torch.zeros((out.shape[0], out.shape[1]))
        for j in range(preds.shape[0]):
            val = preds[j]
            predictions[j,val] = 1

        #preds = (out>0.5).float() # for binary classification
        #targets = targets.argmax(axis=1)  # only for binary classification
        n_correct = (targets == preds).sum()

        f1_micro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='micro')
        f1_macro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='macro')

        return loss, n_correct, torch.as_tensor(f1_micro).cuda(), torch.as_tensor(f1_macro).cuda(), predictions, labels

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1_micro = torch.stack([x[mode + '_f1_micro'] for x in outputs]).mean()
        f1_macro = torch.stack([x[mode + '_f1_macro'] for x in outputs]).mean()
        
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()

        return avg_loss, acc, f1_micro, f1_macro, preds, targets

    def training_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro, 'val_preds': preds, 'val_targets': targets}

    def test_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct, 'test_f1_micro': f1_micro, 'test_f1_macro': f1_macro}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1_micro, f1_macro, preds, targets = self.general_end(outputs, "val")
         
        preds = preds.reshape(preds.shape[0]*preds.shape[1],preds.shape[2])
        targets = targets.reshape(targets.shape[0]*targets.shape[1],targets.shape[2])
        
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1-Micro={}".format(f1_micro))
        print("Val-F1-Macro={}".format(f1_macro))
        #cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print("Learning-Rate: "+str(cur_lr))
        self.log("val_loss", avg_loss)
        self.log("val_f1_micro", f1_micro)
        self.log("val_f1_macro", f1_macro)
        self.log("val_acc", acc)
        if self.current_epoch>=(1-1): # currently only print from epoch 4 onwards -> probably needs adjustment later on
            file = 'cite_logs/'
            with open(file+'lightning_logs/version_'+str(self.logger.version)+'/classification_report.txt', 'a') as f:
                f.write("Epoch "+str(self.current_epoch+1)+":\n")
                f.write("Validation-Loss: "+str(avg_loss)+"\n")
                f.write("Validation-F1-Macro: "+str(f1_macro)+"\n")
                f.write("Validation-F1-Micro: "+str(f1_micro)+"\n")
                f.write("Validation-Acc: "+str(acc)+"\n")
                #t = [[self.id2tag[p] for p in pre] for pre in targets]
                #p = [[self.id2tag[p] for p in pre] for pre in preds]
                label_names = []
                for i in range(len(self.id2tag)):
                    label_names.append(self.id2tag[i])
                f.write(classification_report(targets, preds, target_names=label_names))
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1_micro", f1_micro, self.global_step)
        self.logger.experiment.add_scalar("val_f1_macro", f1_macro, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True, pin_memory=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle=False, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True, pin_memory=True)
    
    def configure_optimizers(self):

        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert_t.parameters()) + list(self.bert_a.parameters()) + list(self.cnn.parameters()) + list(self.fig_cnn.parameters()) + list(self.lstm.parameters())+ list(self.cross_attn.parameters()) + list(self.linear_1.parameters()) + list(self.transformer_encoder.parameters()) + list(self.classifier.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.lr, weight_decay=self.hparams['reg'])
        #optim = torch.optim.SGD(params=params, lr = self.hparams['lr'], weight_decay=self.hparams['reg'],momentum=0.9)
        #optim = nn.optim.RMSprop(params, lr=self.hparams['lr'], alpha=0.99, eps=1e-08, weight_decay=self.hparams['reg'], momentum=0)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(self.train_dataloader), epochs=8) # epoch number unkown
        return [optim], [scheduler]


class Year_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert_t, bert_a, cnn, cnn_fig, id2tag, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.id2tag = id2tag
        self.bert_t = bert_t
        self.bert_a = bert_a
        self.cnn = cnn
        self.cnn_fig = cnn_fig
        self.lstm = nn.Identity()
        self.cross_attn = nn.Identity()
        self.linear_1 = nn.Identity()
        self.transformer_encoder = nn.Identity()
        self.id2tag = id2tag
        #self.classifier = nn.Identity()
        self.input_type = hparams["input"]
        self.seq_amount = (hparams["input"]=="title+abstract")+1
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        if "large" in self.hparams['model']:
            self.emb_dim = 1024
        else:
            self.emb_dim = 768
            
        if hparams["input"]=="image": # needs further adjustments later on
            self.emb_dim = 1000
            self.seq_amount = 1
        if hparams["input"]=="figures":
            self.emb_dim = 1000
            self.seq_amount = 1
            if hparams["fig_lstm"]:
                self.emb_dim = self.hparams["lstm_hidden"]*self.hparams["lstm_num_l"]*2

                self.lstm = nn.LSTM(input_size=1000, hidden_size=self.hparams["lstm_hidden"], num_layers=self.hparams["lstm_num_l"], bidirectional=True, batch_first=True)
        if hparams["input"]=="title+abstract+image":
            self.emb_dim = 1000+768*2
            self.seq_amount=1
            
        if hparams["input"]=="image+figures":
            self.emb_dim = 1000*2
            self.seq_amount=1
        
        if hparams["t+i_encoder"]:
            self.emb_dim = 3*768
            self.linear_1 = nn.Sequential(
                    nn.Dropout(self.hparams["drop"]),
                    nn.Linear(1000, 768),
                    )
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        elif hparams["fig_enc"]:
            encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        
        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        # start with 768 when bert base design
        # else for bert large 1024 embedding dim
        # so far only for simple concatenation when combining multiple input features
        # try more advanced combinations like attention or a full transformer encoder
        self.classifier = nn.Sequential(
            nn.Dropout(self.hparams["drop"]),
            nn.Linear(self.emb_dim*self.seq_amount, self.hparams["n_hidden_out"]),
            #nn.PReLU(),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_2"]),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Sigmoid(),
            #nn.PReLU(),
            #nn.Linear(self.hparams["n_hidden_2"], self.hparams["n_hidden_out"]),
            #nn.Dropout(self.hparams["drop"]),
        )
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    
    def forward_title(self, input_ids, masks):
        
        x1 = self.bert_t(input_ids, masks)
        # For DistilBert:
        #x1 = x1[0][:,0]
        x = x1
        if self.hparams["all_emb"]:
            x, hidden_state = self.lstm(x)
            hidden_state = hidden_state[0]
            # Flatten hidden state with respect to batch size

            hidden = hidden_state.transpose(1,0).contiguous().view(self.hparams["batch_size"], -1)
            
            x = self.classifier(hidden)
        else:
            x = self.classifier(x)
            
        return x
    
    def forward_abstract(self, input_ids, masks):
        
        x1 = self.bert_a(input_ids, masks)
        # For DistilBert:
        #x1 = x1[0][:,0]
        x = x1
        x = self.classifier(x)
        #x = self.model(x)
        return x
    
    def forward_title_abstract(self, input_ids_t, masks_t, input_ids_a, masks_a):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        # For DistilBert:
        #x1 = x1[0][:,0]
        #x2 = x2[0][:,0]
        if self.hparams["cattn"]: # use cross-attention to fuse embeddings
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = x2.clone()
            for c in range(self.hparams["c_depth"]):
                x = self.cross_attn[c](x, x1, x1) # x2: abstract emb, x1: title emb
            x = x.reshape(x.shape[0], x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2),1)
        x = self.classifier(x)
        #x = self.model(x)
        return x
    
    def forward_img(self, img):
        
        x = self.cnn(img)
        x = self.classifier(x)
        
        return x
    
    def forward_fig(self, fig, seq_len):
        
        pad_seq = pack_padded_sequence(fig, seq_len.cpu(), batch_first = True, enforce_sorted=False)
        fig = pad_seq.data
        x = self.cnn_fig(fig)
        if self.hparams["fig_lstm"]:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = self.lstm(seq)
            x,_ = pad_packed_sequence(x, batch_first=True)
            # take final hidden states output
            x = x[:,-1]
        elif self.hparams["fig_enc"]:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = pad_packed_sequence(seq, batch_first=True)
            x = self.transformer_encoder(x)
            # pick last seq embedding for output -> might want to try to run LSTM over entire seq
            x = x[:,-1]
        else:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = pad_packed_sequence(seq, batch_first=True)
            # Max Pool accross entire seq for now
            x = torch.max(x, axis=1).values
            
        x = self.classifier(x)
        
        return x
        
    def forward_text_img(self, input_ids_t, masks_t, input_ids_a, masks_a, img):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        x3 = self.cnn(img)
        # For DistilBert:
        #x1 = x1[0][:,0]
        #x2 = x2[0][:,0]
        if self.hparams["cattn"]: # use cross-attention to fuse text embeddings
            # TODO: handle images -> however probably not wort pursuing
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = x2.clone()
            for c in range(self.hparams["c_depth"]):
                x = self.cross_attn[c](x, x1, x1) # x2: abstract emb, x1: title emb
            x = x.reshape(x.shape[0], x.shape[2])
        elif self.hparams["t+i_encoder"]:
            #x = torch.cat((x1, x2, x3),1)
            x3 = self.linear_1(x3)
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x3 = x3.reshape(x3.shape[0], 1, x3.shape[1])
            x = torch.cat((x1,x2,x3),1)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = self.transformer_encoder(x)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2, x3),1)
        
        x = self.classifier(x)
        
        return x
    
    def forward_img_fig(self, img, fig, seq_len):
        
        x1 = self.cnn(img)
        pad_seq = pack_padded_sequence(fig, seq_len.cpu(), batch_first = True, enforce_sorted=False)
        fig = pad_seq.data
        x2 = self.cnn(fig)
        seq = PackedSequence(x2, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
        x2,_ = pad_packed_sequence(seq, batch_first=True)
        # Max Pool accross entire seq for now
        x2 = torch.max(x2, axis=1).values
        
        if self.hparams["t+i_encoder"]:
            #x = torch.cat((x1, x2, x3),1)
            x1 = self.linear_1(x1)
            x2 = self.linear_1(x2)
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = torch.cat((x1,x2),1)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = self.transformer_encoder(x)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2),1)
        
        x = self.classifier(x)
        
        return x
    
    def forward(self, input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len):
        
        if self.input_type=="title":
            x = self.forward_title(input_ids_t, masks_t)
        elif self.input_type=="abstract":
            x = self.forward_abstract(input_ids_a, masks_a)
        elif self.input_type=="title+abstract":
            x = self.forward_title_abstract(input_ids_t, masks_t, input_ids_a, masks_a)
        elif self.input_type=="image":
            x = self.forward_img(img)
        elif self.input_type=="figures":
            x = self.forward_fig(fig, seq_len)
        elif self.input_type=="title+abstract+image":
            x = self.forward_text_img(input_ids_t, masks_t, input_ids_a, masks_a, img)
        elif self.input_type=="image+figures":
            x = self.forward_img_fig(img, fig, seq_len)
        else:
            AssertionError
            
        return x

    def general_step(self, batch, batch_idx, mode):
        
        if "title" in self.input_type:
            masks_t = batch['attention_mask_t']
            input_ids_t = batch['input_ids_t'].squeeze(1)
        else:
            masks_t = None
            input_ids_t = None
        if "abstract" in self.input_type:
            masks_a = batch['attention_mask_a']
            input_ids_a = batch['input_ids_a'].squeeze(1)
        else:
            masks_a = None
            input_ids_a = None
        if "image" in self.input_type:
            img = batch['img']
        else:
            img = None
        if "figures" in self.input_type:
            fig = batch['fig']
            seq_len = batch['seq_len']
        else:
            fig = None
            seq_len = None
        
        targets = batch['labels']

        # forward pass
        out = self.forward(input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len)
        
        # loss
        targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])

        out = out.to(torch.float32)

        # For Multi-Class Classification
        labels = targets
        targets = targets.argmax(axis=1)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(out, targets)
        # For Binary Classification:
        #loss_func = nn.BCELoss()
        #loss = loss_func(out, targets.type(torch.float))

        #targets = targets.argmax(axis=1)
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        predictions = torch.zeros((out.shape[0], out.shape[1]))
        for j in range(preds.shape[0]):
            val = preds[j]
            predictions[j,val] = 1
        #acc = n_correct / len(batch)

        f1_micro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='micro')
        f1_macro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='macro')

        return loss, n_correct, torch.as_tensor(f1_micro).cuda(), torch.as_tensor(f1_macro).cuda(), predictions, labels

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1_micro = torch.stack([x[mode + '_f1_micro'] for x in outputs]).mean()
        f1_macro = torch.stack([x[mode + '_f1_macro'] for x in outputs]).mean()
        
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()

        return avg_loss, acc, f1_micro, f1_macro, preds, targets

    def training_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        n_correct = n_correct.double()
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1_micro':f1_micro, 'val_f1_macro':f1_macro, 'val_preds': preds, 'val_targets': targets}

    def test_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1_micro, f1_macro, preds, targets = self.general_end(outputs, "val")
        
        preds = preds.reshape(preds.shape[0]*preds.shape[1],preds.shape[2])
        targets = targets.reshape(targets.shape[0]*targets.shape[1],targets.shape[2])
        
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1-Micro={}".format(f1_micro))
        print("Val-F1-Macro={}".format(f1_macro))
        self.log("val_loss", avg_loss) 
        self.log("val_f1_micro", f1_micro)
        self.log("val_f1_macro", f1_macro)
        self.log("val_acc", acc)
        
        if self.current_epoch>=(1-1): # currently only print from epoch 4 onwards -> probably needs adjustment later on
            file = 'year_logs/'
            with open(file+'lightning_logs/version_'+str(self.logger.version)+'/classification_report.txt', 'a') as f:
                f.write("Epoch "+str(self.current_epoch+1)+":\n")
                f.write("Validation-Loss: "+str(avg_loss)+"\n")
                f.write("Validation-F1-Macro: "+str(f1_macro)+"\n")
                f.write("Validation-F1-Micro: "+str(f1_micro)+"\n")
                f.write("Validation-Acc: "+str(acc)+"\n")
                #t = [[self.id2tag[p] for p in pre] for pre in targets]
                #p = [[self.id2tag[p] for p in pre] for pre in preds]
                label_names = []
                for i in range(len(self.id2tag)):
                    label_names.append(self.id2tag[i])
                f.write(classification_report(targets, preds, target_names=label_names))
        
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1_micro", f1_micro, self.global_step)
        self.logger.experiment.add_scalar("val_f1_macro", f1_macro, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro,'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.hparams['batch_size'],num_workers=12, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle=False, batch_size=self.hparams['batch_size'],num_workers=12, drop_last=True)
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert_t.parameters()) + list(self.bert_a.parameters()) + list(self.cnn.parameters()) + list(self.cnn_fig.parameters()) + list(self.lstm.parameters())+ list(self.cross_attn.parameters()) + list(self.linear_1.parameters()) + list(self.transformer_encoder.parameters()) + list(self.classifier.parameters())
        #optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        optim = torch.optim.AdamW(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #optim = torch.optim.SGD(params=params, lr = self.hparams['lr'], weight_decay=self.hparams['reg'],momentum=0.9)
        
        # Define LR-Scheduler:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda= lambda epoch: 0.95**epoch)
        #num_training_steps = int(len(self.data['train'])/self.hparams['batch_size'])*6
        #num_warmup_steps = num_training_steps*0.1
        #scheduler = torch.optim.get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        return [optim], [scheduler]

    def MAcc(out, targets):
      preds = np.zeros(len(out[0]))
      for i in range(0, len(out)):
        for j in range(0, len(out[i])):
          if out[i,j]>=0.5:
            preds[j] = 1
      n_correct = (targets == preds).sum()

      return n_correct
      

    def getAcc(self, loader=None):
        self.eval()
        self = self.to(self.device)

        if not loader:
            loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
    

class Key_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert_t=nn.Identity(), bert_a=nn.Identity(), cnn=nn.Identity(), id2tag=None, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(Namespace(**hparams))
        self.bert_t = bert_t
        self.bert_a = bert_a
        self.cnn = cnn
        self.lstm = nn.Identity()
        self.cross_attn = nn.Identity()
        self.linear_1 = nn.Identity()
        self.transformer_encoder = nn.Identity()
        self.id2tag = id2tag
        #self.classifier = nn.Identity()
        self.input_type = hparams["input"]
        self.seq_amount = (hparams["input"]=="title+abstract")+1
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        if "large" in self.hparams['model']:
            self.emb_dim = 1024
        else:
            self.emb_dim = 768
            
        if hparams["input"]=="image": # needs further adjustments later on
            self.emb_dim = 1000
            self.seq_amount = 1
        if hparams["input"]=="figures":
            self.emb_dim = 1000
            self.seq_amount = 1
            if hparams["fig_lstm"]:
                self.emb_dim = self.hparams["lstm_hidden"]*self.hparams["lstm_num_l"]*2

                self.lstm = nn.LSTM(input_size=1000, hidden_size=self.hparams["lstm_hidden"], num_layers=self.hparams["lstm_num_l"], bidirectional=True, batch_first=True)
        if hparams["input"]=="title+abstract+image":
            self.emb_dim = 1000+768*2
            self.seq_amount = 1
        if hparams["input"]=="title+abstract+figures":
            self.emb_dim = 1000+768*2
            self.seq_amount = 1
        if hparams["t+i_encoder"]:
            self.emb_dim = 3*768
            self.linear_1 = nn.Sequential(
                    nn.Dropout(self.hparams["drop"]),
                    nn.Linear(1000, 768),
                    )
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        
        elif hparams["fig_enc"]:
            encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        self.classifier = nn.Sequential(
                nn.Dropout(self.hparams["drop"]),            
                nn.Linear(self.seq_amount*self.emb_dim, self.hparams["n_hidden_out"]),
                nn.Sigmoid(),
            )
        
        if self.hparams["2_layer"]:
            self.classifier = nn.Sequential(
                nn.Dropout(self.hparams["drop"]),            
                nn.Linear(self.seq_amount*self.emb_dim, self.hparams["n_hidden_1"]),
                #nn.BatchNorm1d(self.hparams["n_hidden_1"]),
                #nn.LayerNorm(self.hparams["n_hidden_1"]),
                #nn.InstanceNorm1d(self.hparams["n_hidden_1"]), # without learnable parameters
                #nn.InstanceNorm1d(self.hparams["n_hidden_1"], affine=True), # with learnable parameters
                #nn.GroupNorm(3, 6) # not sure how to use it yet!
                #nn.GELU(),
                #nn.GLU(),
                nn.PReLU(),
                nn.Dropout(self.hparams["drop"]),
                nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out"]),
                nn.Sigmoid(),
            )
        if self.hparams["3_layer"]:
            self.classifier = nn.Sequential(
                nn.Dropout(self.hparams["drop"]),            
                nn.Linear(self.seq_amount*self.emb_dim, self.hparams["n_hidden_1"]),
                nn.PReLU(),
                nn.Dropout(self.hparams["drop"]),
                nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_2"]),
                nn.PReLU(),
                nn.Dropout(self.hparams["drop"]),
                nn.Linear(self.hparams["n_hidden_2"], self.hparams["n_hidden_out"]),
                nn.Sigmoid(),
            )

        if self.hparams["all_emb"]:
            self.lstm = nn.LSTM(768, hidden_size=self.hparams["lstm_hidden_size"], num_layers=2, bidirectional=True, batch_first=True)
            self.classifier = nn.Sequential(
                    nn.Linear(self.hparams["lstm_hidden_size"]*4 , self.hparams["n_hidden_1"]),
                    nn.PReLU(),
                    nn.Dropout(self.hparams["drop"]),
                    nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out"]),
                    nn.Sigmoid()
                    )
        if self.hparams["cattn"]:
            self.cross_attn = nn.ModuleList(
                    MultiHeadAttention(d_model=self.emb_dim, d_k=self.emb_dim // self.hparams["c_head"], d_v=self.emb_dim // self.hparams["c_head"], h=self.hparams["c_head"]) for i in range(self.hparams["c_depth"]))
            '''
            self.self_attn = nn.ModuleList(
                    MultiHeadAttention(d_model=hidden_size, d_k=hidden_size // head, d_v=hidden_size // head, h=head) for i in range(depth))  # k, q, v
            '''
            self.seq_amount = 1
            self.classifier = nn.Sequential(
                    nn.Dropout(self.hparams["drop"]),            
                    nn.Linear(self.seq_amount*self.emb_dim, self.hparams["n_hidden_out"]),
                    nn.Sigmoid(),
                    )
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    def forward_title(self, input_ids, masks):
        
        x1 = self.bert_t(input_ids, masks)
        # For DistilBert:
        #x1 = x1[0][:,0]
        x = x1
        if self.hparams["all_emb"]:
            x, hidden_state = self.lstm(x)
            hidden_state = hidden_state[0]
            # Flatten hidden state with respect to batch size

            hidden = hidden_state.transpose(1,0).contiguous().view(self.hparams["batch_size"], -1)
            
            x = self.classifier(hidden)
        else:
            x = self.classifier(x)
            
        return x
    
    def forward_abstract(self, input_ids, masks):
        
        x1 = self.bert_a(input_ids, masks)
        # For DistilBert:
        #x1 = x1[0][:,0]
        x = x1
        x = self.classifier(x)
        #x = self.model(x)
        return x
    
    def forward_title_abstract(self, input_ids_t, masks_t, input_ids_a, masks_a):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        # For DistilBert:
        #x1 = x1[0][:,0]
        #x2 = x2[0][:,0]
        if self.hparams["cattn"]: # use cross-attention to fuse embeddings
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = x2.clone()
            for c in range(self.hparams["c_depth"]):
                x = self.cross_attn[c](x, x1, x1) # x2: abstract emb, x1: title emb
            x = x.reshape(x.shape[0], x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2),1)
        x = self.classifier(x)
        #x = self.model(x)
        return x
    
    def forward_img(self, img):
        
        x = self.cnn(img)
        x = self.classifier(x)
        
        return x
    
    def forward_fig(self, fig, seq_len):
        
        pad_seq = pack_padded_sequence(fig, seq_len.cpu(), batch_first = True, enforce_sorted=False)
        fig = pad_seq.data
        x = self.cnn(fig)
        if self.hparams["fig_lstm"]:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = self.lstm(seq)
            x,_ = pad_packed_sequence(x, batch_first=True)
            # take final hidden states output
            x = x[:,-1]
        elif self.hparams["fig_enc"]:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = pad_packed_sequence(seq, batch_first=True)
            x = self.transformer_encoder(x)
            # pick last seq embedding for output -> might want to try to run LSTM over entire seq
            x = x[:,-1]
        else:
            seq = PackedSequence(x, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
            x,_ = pad_packed_sequence(seq, batch_first=True)
            # Max Pool accross entire seq for now
            x = torch.max(x, axis=1).values
            
        x = self.classifier(x)
        
        return x
        
    def forward_text_img(self, input_ids_t, masks_t, input_ids_a, masks_a, img):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        x3 = self.cnn(img)
        # For DistilBert:
        #x1 = x1[0][:,0]
        #x2 = x2[0][:,0]
        if self.hparams["cattn"]: # use cross-attention to fuse text embeddings
            # TODO: handle images -> however probably not wort pursuing
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = x2.clone()
            for c in range(self.hparams["c_depth"]):
                x = self.cross_attn[c](x, x1, x1) # x2: abstract emb, x1: title emb
            x = x.reshape(x.shape[0], x.shape[2])
        elif self.hparams["t+i_encoder"]:
            #x = torch.cat((x1, x2, x3),1)
            x3 = self.linear_1(x3)
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x3 = x3.reshape(x3.shape[0], 1, x3.shape[1])
            x = torch.cat((x1,x2,x3),1)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = self.transformer_encoder(x)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2, x3),1)
        
        x = self.classifier(x)
        
        return x
    
    def forward_text_fig(self, input_ids_t, masks_t, input_ids_a, masks_a, fig, seq_len):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        # handle figures with max pool:
        pad_seq = pack_padded_sequence(fig, seq_len.cpu(), batch_first = True, enforce_sorted=False)
        fig = pad_seq.data
        x3 = self.cnn(fig)
        seq = PackedSequence(x3, pad_seq.batch_sizes, pad_seq.sorted_indices, pad_seq.unsorted_indices)
        x3,_ = pad_packed_sequence(seq, batch_first=True)
        # Max Pool accross entire seq for now
        x3 = torch.max(x3, axis=1).values
        
        if self.hparams["t+i_encoder"]:
            #x = torch.cat((x1, x2, x3),1)
            x3 = self.linear_1(x3)
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x3 = x3.reshape(x3.shape[0], 1, x3.shape[1])
            x = torch.cat((x1,x2,x3),1)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            x = self.transformer_encoder(x)
            x = x.reshape(x.shape[1], x.shape[0], x.shape[2])
            
            x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2, x3),1)
        
        x = self.classifier(x)
        
        return x
    
    def forward(self, input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len):
        
        if self.input_type=="title":
            x = self.forward_title(input_ids_t, masks_t)
        elif self.input_type=="abstract":
            x = self.forward_abstract(input_ids_a, masks_a)
        elif self.input_type=="title+abstract":
            x = self.forward_title_abstract(input_ids_t, masks_t, input_ids_a, masks_a)
        elif self.input_type=="image":
            x = self.forward_img(img)
        elif self.input_type=="figures":
            x = self.forward_fig(fig, seq_len)
        elif self.input_type=="title+abstract+image":
            x = self.forward_text_img(input_ids_t, masks_t, input_ids_a, masks_a, img)
        elif self.input_type=="title+abstract+figures":
            x = self.forward_text_fig(input_ids_t, masks_t, input_ids_a, masks_a, fig, seq_len)
        else:
            AssertionError
            
        return x

    def general_step(self, batch, batch_idx, mode):
        
        if "title" in self.input_type:
            masks_t = batch['attention_mask_t']
            input_ids_t = batch['input_ids_t'].squeeze(1)
        else:
            masks_t = None
            input_ids_t = None
        if "abstract" in self.input_type:
            masks_a = batch['attention_mask_a']
            input_ids_a = batch['input_ids_a'].squeeze(1)
        else:
            masks_a = None
            input_ids_a = None
        if "image" in self.input_type:
            img = batch['img']
        else:
            img = None
        if "figures" in self.input_type:
            fig = batch['fig']
            seq_len = batch['seq_len']
        else:
            fig = None
            seq_len = None
        
        targets = batch['labels']

        # forward pass
        out = self.forward(input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len)
        # loss
        targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])

        out = out.to(torch.float32)
        # for multi-label classification:
        ######################
        preds = torch.as_tensor(out.detach().clone() > 0.5, dtype=float).detach().clone()
  
        n_correct = ((preds == targets).sum(axis=1)==len(preds[0])).sum()
        #n_correct = torch.Tensor([n_correct])
        #n_correct = torch.Tensor([0])
        ######################
        loss_func = nn.BCELoss()
        loss = loss_func(out, targets.type(torch.float))
        
        f1_micro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='micro')
        f1_macro = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='macro')

        return loss, n_correct, torch.as_tensor(f1_micro).cuda(), torch.as_tensor(f1_macro).cuda(), preds, targets

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1_micro = torch.stack([x[mode + '_f1_micro'] for x in outputs]).mean()
        f1_macro = torch.stack([x[mode + '_f1_macro'] for x in outputs]).mean()
        
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()

        return avg_loss, acc, f1_micro, f1_macro, preds, targets

    def training_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "train")
        #self.log("train_loss", loss)
        #self.log("loss", {"train": loss})
        tensorboard_logs = {'loss': loss}
        if self.global_step%self.hparams['acc_grad']==0:
            self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'train_f1_micro': f1_micro, 'train_f1_macro': f1_macro, 'train_preds': preds, 'train_targets': targets, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro}
        #self.log("val_loss", loss)     # try this line
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro, 'val_preds': preds, 'val_targets': targets, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, n_correct, f1_micro, f1_macro, preds, targets = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct, 'f1_micro': f1_micro, 'f1_macro':f1_macro}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1_micro, f1_macro, preds, targets = self.general_end(outputs, "val")
        
        preds = preds.reshape(preds.shape[0]*preds.shape[1],preds.shape[2])
        targets = targets.reshape(targets.shape[0]*targets.shape[1],targets.shape[2])
        
        f1_macro = f1_score(targets, preds,average='macro')
        f1_micro = f1_score(targets, preds,average='micro')
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1-Micro={}".format(f1_micro))
        print("Val-F1-Macro={}".format(f1_macro))
        
        self.log('val_loss', avg_loss) 
        self.log('val_acc', acc)
        self.log('val_f1_micro', f1_micro)
        self.log('val_f1_macro', f1_macro)
        if self.current_epoch>=(1-1): # currently only print from epoch 4 onwards -> probably needs adjustment later on
            if self.hparams['input'] == "title":
                file = 'tb_logs/'
            elif self.hparams['input'] == "abstract":
                file = 'tb_abs_logs/'
            elif self.hparams['input'] == "title+abstract":
                file = 'tb_t_abs_logs/'
            elif self.hparams['input'] == "image":
                file = 'tb_image_logs/'
            elif self.hparams['input'] == "figures":
                file = 'tb_figures_logs/'
            elif self.hparams['input'] == "title+abstract+image":
                file = 'tb_text_image_logs/'
            elif self.hparams['input'] == "title+abstract+figures":
                file = 'tb_text_figures_logs/'
            with open(file+'lightning_logs/version_'+str(self.logger.version)+'/classification_report.txt', 'a') as f:
                f.write("Epoch "+str(self.current_epoch+1)+":\n")
                f.write("Validation-Loss: "+str(avg_loss)+"\n")
                f.write("Validation-F1-Macro: "+str(f1_macro)+"\n")
                f.write("Validation-F1-Micro: "+str(f1_micro)+"\n")
                f.write("Validation-Acc: "+str(acc)+"\n")
                #t = [[self.id2tag[p] for p in pre] for pre in targets]
                #p = [[self.id2tag[p] for p in pre] for pre in preds]
                label_names = []
                for i in range(len(self.id2tag)):
                    label_names.append(self.id2tag[i])
                f.write(classification_report(targets, preds, target_names=label_names))
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1_micro", f1_micro, self.global_step)
        self.logger.experiment.add_scalar("val_f1_macro", f1_macro, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1_micro': f1_micro, 'val_f1_macro': f1_macro, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], shuffle=False, batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], shuffle=False, batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert_t.parameters()) + list(self.bert_a.parameters()) + list(self.cnn.parameters()) + list(self.lstm.parameters())+ list(self.cross_attn.parameters()) + list(self.linear_1.parameters()) + list(self.transformer_encoder.parameters()) + list(self.classifier.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda= lambda epoch: 0.95**epoch)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(self.data['train']), epochs=500)

        return [optim], [scheduler]
    
class MTL_YK_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert_t, bert_a, cnn, id2tag_year, id2tag_key, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.bert_t = bert_t
        self.bert_a = bert_a
        self.cnn = cnn
        self.lstm = nn.Identity()
        self.cross_attn = nn.Identity()
        self.linear_1 = nn.Identity()
        self.transformer_encoder = nn.Identity()
        self.id2tag_year = id2tag_year
        self.id2tag_key = id2tag_key
        #self.classifier = nn.Identity()
        self.input_type = hparams["input"]
        self.seq_amount = (hparams["input"]=="title+abstract")+1
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        if "large" in self.hparams['model']:
            self.emb_dim = 1024
        else:
            self.emb_dim = 768
            
        if hparams["input"]=="image": # needs further adjustments later on
            self.emb_dim = 1000
            self.seq_amount = 1
        if hparams["input"]=="figures":
            self.emb_dim = 1000
            self.seq_amount = 1
            if hparams["fig_lstm"]:
                self.emb_dim = self.hparams["lstm_hidden"]*self.hparams["lstm_num_l"]*2

                self.lstm = nn.LSTM(input_size=1000, hidden_size=self.hparams["lstm_hidden"], num_layers=self.hparams["lstm_num_l"], bidirectional=True, batch_first=True)
        if hparams["input"]=="title+abstract+image":
            self.emb_dim = 1000+768*2
            self.seq_amount=1
        
        if hparams["t+i_encoder"]:
            self.emb_dim = 3*768
            self.linear_1 = nn.Sequential(
                    nn.Dropout(self.hparams["drop"]),
                    nn.Linear(1000, 768),
                    )
            encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        elif hparams["fig_enc"]:
            encoder_layer = nn.TransformerEncoderLayer(d_model=1000, nhead=8, dim_feedforward=2048)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=hparams["num_enc_lay"])
        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        
        self.y_classifier = nn.Sequential(
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(768, self.hparams["n_hidden_1"]),
            #nn.PReLU(),
            nn.Linear(self.emb_dim*self.seq_amount, self.hparams["num_labels_year"]),
            nn.Dropout(self.hparams["drop"]),
            #nn.Sigmoid(),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_3"])
        )
        
        self.k_classifier = nn.Sequential(
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(768, self.hparams["n_hidden_1"]),
            #nn.PReLU(),
            nn.Linear(self.emb_dim*self.seq_amount, self.hparams["num_labels_key"]),
            nn.Dropout(self.hparams["drop"]),
            nn.Sigmoid(),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_3"])
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward_title(self, input_ids, masks):
        
        x1 = self.bert_t(input_ids, masks)
        # For DistilBert:
        #x1 = x1[0][:,0]
        x = x1
        xy = self.y_classifier(x)
        xk = self.k_classifier(x)
            
        return xy, xk
    
    def forward_abstract(self, input_ids, masks):
        
        x1 = self.bert_a(input_ids, masks)
        # For DistilBert:
        #x1 = x1[0][:,0]
        x = x1
        xy = self.y_classifier(x)
        xk = self.k_classifier(x)
        #x = self.model(x)
        return xy, xk
    
    def forward_title_abstract(self, input_ids_t, masks_t, input_ids_a, masks_a):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        # For DistilBert:
        #x1 = x1[0][:,0]
        #x2 = x2[0][:,0]
        if self.hparams["cattn"]: # use cross-attention to fuse embeddings
            x1 = x1.reshape(x1.shape[0], 1, x1.shape[1])
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[1])
            x = x2.clone()
            for c in range(self.hparams["c_depth"]):
                x = self.cross_attn[c](x, x1, x1) # x2: abstract emb, x1: title emb
            x = x.reshape(x.shape[0], x.shape[2])
        else: # use simple concatenation to fuse embeddings
            x = torch.cat((x1, x2),1)
        xy = self.y_classifier(x)
        xk = self.k_classifier(x)
        #x = self.model(x)
        return xy, xk
    
    def forward(self, input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len):
        
        if self.input_type=="title":
            x = self.forward_title(input_ids_t, masks_t)
        elif self.input_type=="abstract":
            x = self.forward_abstract(input_ids_a, masks_a)
        elif self.input_type=="title+abstract":
            x = self.forward_title_abstract(input_ids_t, masks_t, input_ids_a, masks_a)
        elif self.input_type=="image":
            x = self.forward_img(img)
        elif self.input_type=="figures":
            x = self.forward_fig(fig, seq_len)
        elif self.input_type=="title+abstract+image":
            x = self.forward_text_img(input_ids_t, masks_t, input_ids_a, masks_a, img)
        else:
            AssertionError
            
        return x

    def general_step(self, batch, batch_idx, mode):
        
        if "title" in self.input_type:
            masks_t = batch['attention_mask_t']
            input_ids_t = batch['input_ids_t'].squeeze(1)
        else:
            masks_t = None
            input_ids_t = None
        if "abstract" in self.input_type:
            masks_a = batch['attention_mask_a']
            input_ids_a = batch['input_ids_a'].squeeze(1)
        else:
            masks_a = None
            input_ids_a = None
        if "image" in self.input_type:
            img = batch['img']
        else:
            img = None
        if "figures" in self.input_type:
            fig = batch['fig']
            seq_len = batch['seq_len']
        else:
            fig = None
            seq_len = None
        
        targets_y = batch['labels_year']
        targets_k = batch['labels_key']

        # forward pass
        out_y, out_k = self.forward(input_ids_t, masks_t, input_ids_a, masks_a, img, fig, seq_len)
        
        # loss
        targets_y = targets_y.reshape(len(targets_y), self.hparams["num_labels_year"])
        targets_k = targets_k.reshape(len(targets_k), self.hparams["num_labels_key"])

        out_y = out_y.to(torch.float32)
        out_k = out_k.to(torch.float32)

        # For Multi-Class Classification
        targets_y = torch.max(targets_y, 1)[1].long()
        loss_func = nn.CrossEntropyLoss()
        loss_y = loss_func(out_y, targets_y)
        
        # for multi-label classification:
        ######################
        preds_k = torch.as_tensor(out_k.detach().clone() > 0.5, dtype=float).detach().clone()
  
        n_correct_k = ((preds_k == targets_k).sum(axis=1)==len(preds_k[0])).sum()
        #n_correct = torch.Tensor([n_correct])
        #n_correct = torch.Tensor([0])
        ######################
        loss_func = nn.BCELoss()
        loss_k = loss_func(out_k, targets_k.type(torch.float))
        
        f1_micro_k = f1_score(preds_k.cpu().numpy(), targets_k.cpu().numpy(),average='micro')
        f1_macro_k = f1_score(preds_k.cpu().numpy(), targets_k.cpu().numpy(),average='macro')
        
        
        preds_y = out_y.argmax(axis=1)
        n_correct_y = (targets_y == preds_y).sum()
        
        f1_micro_y = f1_score(preds_y.cpu().numpy(), targets_y.cpu().numpy(),average='micro')
        f1_macro_y = f1_score(preds_y.cpu().numpy(), targets_y.cpu().numpy(),average='macro')
        
        # Final Loss:
        loss = self.hparams["year_lw"]*loss_y + self.hparams["key_lw"]*loss_k
        
        return loss, n_correct_y, n_correct_k, torch.as_tensor(f1_micro_k).cuda(), torch.as_tensor(f1_macro_k).cuda(), torch.as_tensor(f1_micro_y).cuda(), torch.as_tensor(f1_macro_y).cuda(), preds_y, preds_k, targets_y, targets_k

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct_y = torch.stack(
            [x[mode + '_n_correct_y'] for x in outputs]).sum().cpu().numpy()
        total_correct_c = torch.stack(
            [x[mode + '_n_correct_k'] for x in outputs]).sum().cpu().numpy()
        f1_micro_y = torch.stack([x[mode + '_f1_micro_y'] for x in outputs]).mean()
        f1_macro_y = torch.stack([x[mode + '_f1_macro_y'] for x in outputs]).mean()
        f1_micro_k = torch.stack([x[mode + '_f1_micro_k'] for x in outputs]).mean()
        f1_macro_k = torch.stack([x[mode + '_f1_macro_k'] for x in outputs]).mean()
        preds_y = torch.stack(
            [x[mode + '_preds_y'] for x in outputs]).cpu().numpy()
        preds_k = torch.stack(
            [x[mode + '_preds_k'] for x in outputs]).cpu().numpy()
        targets_y = torch.stack(
            [x[mode + '_targets_y'] for x in outputs]).cpu().numpy()
        targets_k = torch.stack(
            [x[mode + '_targets_k'] for x in outputs]).cpu().numpy()
        acc_y = total_correct_y / len(self.data[mode])
        acc_k = total_correct_c / len(self.data[mode])
        #f1_y = f1_score(preds_y[0], targets_y[0],average='weighted')
        #f1_c = f1_score(preds_c[0], targets_c[0],average='weighted')
        return avg_loss, acc_y, acc_k, f1_micro_y, f1_macro_y, f1_micro_k, f1_macro_k

    def training_step(self, batch, batch_idx):
        loss, n_correct_y, n_correct_k, f1_micro_k, f1_macro_k, f1_micro_y, f1_macro_y, preds_y, preds_k, targets_y, targets_k = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        n_correct_y = n_correct_y.double()
        n_correct_k = n_correct_k.double()
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct_y': n_correct_y, 'train_n_correct_k': n_correct_k,'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct_y, n_correct_k, f1_micro_k, f1_macro_k, f1_micro_y, f1_macro_y, preds_y, preds_k, targets_y, targets_k = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'val_loss': loss, 'val_n_correct_k': n_correct_k, 'val_f1_micro_k': f1_micro_k, 'val_f1_macro_k': f1_macro_k, 'val_n_correct_y': n_correct_y, 'val_f1_micro_y': f1_micro_y, 'val_f1_macro_y': f1_macro_y}
        return {'val_loss': loss, 'val_n_correct_y': n_correct_y, 'val_n_correct_k': n_correct_k, 'val_f1_micro_y': f1_micro_y, 'val_f1_micro_k': f1_micro_k, 'val_f1_macro_y': f1_macro_y, 'val_f1_macro_k': f1_macro_k, 'val_preds_y': preds_y, 'val_preds_k': preds_k, 'val_targets_y': targets_y, 'val_targets_k': targets_k, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, n_correct_y, n_correct_k, preds_y, preds_k, targets_y, targets_k = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct_y': n_correct_y, 'test_n_correct_k': n_correct_k}

    def validation_epoch_end(self, outputs):
        avg_loss, acc_y, acc_k, f1_micro_y, f1_macro_y, f1_micro_k, f1_macro_k = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc-Year={}".format(acc_y))
        print("Val-Acc-Key{}".format(acc_k))
        print("Val-F1-Micro-Year={}".format(f1_micro_y))
        print("Val-F1-Macro-Year={}".format(f1_macro_y))
        print("Val-F1-Micro-Key={}".format(f1_micro_k))
        print("Val-F1-Macro-Key={}".format(f1_macro_k))
        self.log("val_loss", avg_loss) 
        self.log("val_acc_y", acc_y)
        self.log("val_acc_k", acc_k)
        self.log("val_f1_micro_y", f1_micro_y)
        self.log("val_f1_micro_k", f1_micro_k)
        self.log("val_f1_macro_y", f1_macro_y)
        self.log("val_f1_macro_k", f1_macro_k)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc_y", acc_y, self.global_step)
        self.logger.experiment.add_scalar("val_f1_micro_y", f1_micro_y, self.global_step)
        self.logger.experiment.add_scalar("val_acc_k", acc_k, self.global_step)
        self.logger.experiment.add_scalar("val_f1_micro_k", f1_micro_k, self.global_step)
        self.logger.experiment.add_scalar("val_f1_macro_y", f1_macro_y, self.global_step)
        self.logger.experiment.add_scalar("val_f1_macro_k", f1_macro_k, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc_y': acc_y, 'val_acc_k': acc_k, 'val_f1_micro_y': f1_micro_y, 'val_f1_micro_k': f1_micro_k, 'val_f1_macro_y': f1_macro_y, 'val_f1_macro_k': f1_macro_k}
        return {'val_loss': avg_loss, 'val_acc_y': acc_y, 'val_acc_k': acc_k, 'val_f1_micro_y': f1_micro_y, 'val_f1_micro_k': f1_micro_k, 'val_f1_macro_y': f1_macro_y, 'val_f1_macro_k': f1_macro_k, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'],num_workers=12, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'],num_workers=12, drop_last=True)
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert_t.parameters()) + list(self.bert_a.parameters()) + list(self.k_classifier.parameters()) + list(self.y_classifier.parameters())
        #optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        optim = torch.optim.AdamW(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #optim = torch.optim.SGD(params=params, lr = self.hparams['lr'], weight_decay=self.hparams['reg'],momentum=0.9)
        
        # Define LR-Scheduler:
        #num_training_steps = int(len(self.data['train'])/self.hparams['batch_size'])*6
        #num_warmup_steps = num_training_steps*0.1
        #scheduler = torch.optim.get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        return optim
     
