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

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report, accuracy_score



class Bert(nn.Module):
    
    def __init__(self, hparams):
        super().__init__()
        
        #set Bert
        configuration = AutoConfig.from_pretrained(hparams["model"])
        # For BERT
        #configuration.hidden_dropout_prob = hparams["b_drop"]
        #configuration.attention_probs_dropout_prob = hparams["b_drop"]
        # For DistilBERT
        configuration.dropout = hparams["b_drop"]
        configuration.attention_dropout = hparams["b_drop"]
        #configuration.num_labels=hparams["n_hidden_out"] # For SequenceClassificationModel
        
        self.bert = AutoModel.from_pretrained(hparams["model"], config = configuration)
        
        
        # freeze some of the BERT weights:
        
        modules = [self.bert.embeddings, *self.bert.transformer.layer[:hparams["freeze"]]] 
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        
        
    def forward(self, input_ids, mask):
        # feed x into encoder!
        # For BERT
        #_, pooled_output = self.bert(input_ids=input_ids, attention_mask=mask,return_dict=False)
        # For DistilBERT
        pooled_output = self.bert(input_ids=input_ids, attention_mask=mask,return_dict=False)
        
        return pooled_output



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

        #loss = F.cross_entropy(out, targets)
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
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc}
        return {'val_loss': avg_loss, 'val_acc': acc, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'])

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'])
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert.parameters()) + list(self.model.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])

        return optim

class Cite_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
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
            #nn.Linear(768, self.hparams["n_hidden_1"]),
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out"]),
            #nn.Dropout(self.hparams["drop"]),
            nn.Sigmoid(),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_3"])
        )
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, input_ids, masks):
        
        x = self.bert(input_ids, masks)
        # For SequenceClassificationModel
        #x = x[0]
        # For DistilBERT:
        x = x[0]
        x = self.model(x[:,0])
        # For BERT
        #x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        
        masks = batch['attention_mask']
        input_ids = batch['input_ids'].squeeze(1)
        targets = batch['labels']

        # forward pass
        out = self.forward(input_ids, masks)
        
        # loss
        targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])

        out = out.to(torch.float32)

        #targets = torch.max(targets, 1)[1].long()
        #loss_func = nn.CrossEntropyLoss()
        loss_func = nn.BCELoss()
        loss = loss_func(out, targets.type(torch.float))

        preds = out.argmax(axis=1)
        targets = targets.argmax(axis=1)
        n_correct = (targets == preds).sum()
        
        return loss, n_correct, preds, targets

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1 = f1_score(preds[0], targets[0],average='weighted')
        return avg_loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, n_correct, preds, targets = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, preds, targets = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_preds': preds, 'val_targets': targets}

    def test_step(self, batch, batch_idx):
        loss, n_correct, preds, targets = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1 = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1={}".format(f1))
        #cur_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        #print("Learning-Rate: "+str(cur_lr))
        self.log("val_loss", avg_loss)
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1", f1, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'], num_workers=12, drop_last=True)
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert.parameters()) + list(self.model.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(self.data['train']), epochs=500)

        return [optim], [scheduler]


class Year_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.bert = bert
        self.model = nn.Identity()
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        
        self.model = nn.Sequential(
            #nn.Dropout(self.hparams["drop"]),
            nn.Linear(768, self.hparams["n_hidden_1"]),
            nn.PReLU(),
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out"]),
            nn.Dropout(self.hparams["drop"]),
            nn.Sigmoid(),
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
        # For BERT:
        #x = self.model(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        
        masks = batch['attention_mask']
        input_ids = batch['input_ids'].squeeze(1)
        targets = batch['labels']
        '''
        input_ids = input_ids.to(0)
        masks = masks.to(0)
        self = self.to(0)
        '''
        # forward pass
        out = self.forward(input_ids, masks)
        # loss
        targets = targets.reshape(len(targets), self.hparams["n_hidden_out"])

        out = out.to(torch.float32)

        # For Multi-Class Classification
        #targets = torch.max(targets, 1)[1].long()
        #loss_func = nn.CrossEntropyLoss()
        #loss = loss_func(out, targets)
        # For Binary Classification:
        loss_func = nn.BCELoss()
        loss = loss_func(out, targets.type(torch.float))
        targets = targets.argmax(axis=1)
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        
        return loss, n_correct, preds, targets

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1 = f1_score(preds[0], targets[0],average='micro')
        return avg_loss, acc, f1

    def training_step(self, batch, batch_idx):
        loss, n_correct, preds, targets = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        n_correct = n_correct.double()
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, preds, targets = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_preds': preds, 'val_targets': targets}

    def test_step(self, batch, batch_idx):
        loss, n_correct, preds, targets = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1 = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1={}".format(f1))
        self.log("val_loss", avg_loss) 
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1", f1, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1, 'log': tensorboard_logs}
    
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
        params = list(self.bert.parameters()) + list(self.model.parameters())
        #optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        optim = torch.optim.AdamW(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #optim = torch.optim.SGD(params=params, lr = self.hparams['lr'], weight_decay=self.hparams['reg'],momentum=0.9)
        
        # Define LR-Scheduler:
        #num_training_steps = int(len(self.data['train'])/self.hparams['batch_size'])*6
        #num_warmup_steps = num_training_steps*0.1
        #scheduler = torch.optim.get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        return optim

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

    def __init__(self, hparams, bert, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
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
            #nn.Linear(768, self.hparams["n_hidden_1"]),
            #nn.PReLU(),
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out"]),
            #nn.Dropout(self.hparams["drop"]),
            nn.Sigmoid(),
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
        
        f1 = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='micro')
        return loss, n_correct, torch.tensor(f1).cuda()

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1 = torch.stack([x[mode + '_f1'] for x in outputs]).mean()
        '''
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()
        f1 = f1_score(preds[0], targets[0],average='micro')
        '''
        return avg_loss, acc, f1 

    def training_step(self, batch, batch_idx):
        loss, n_correct, f1 = self.general_step(batch, batch_idx, "train")
        #self.log("train_loss", loss)
        #self.log("loss", {"train": loss})
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'train_f1': f1, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, f1 = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1': f1}
        #self.log("val_loss", loss)     # try this line
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1': f1, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, n_correct, f1 = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct, 'f1': f1}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1 = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1={}".format(f1))
        #self.global_step=self.gst
        self.log('val_loss', avg_loss) 
        #self.log('val_acc', acc)
        self.log('val_f1', f1)
        #self.log("loss", {"val": avg_loss})
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1", f1, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert.parameters()) + list(self.model.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(self.data['train']), epochs=500)

        return [optim], [scheduler]
    

class Key_Classifier2(pl.LightningModule):

    def __init__(self, hparams, bert_t, bert_a, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.bert_t = bert_t
        self.bert_a = bert_a
        self.model = nn.Identity()
        self.input_type = hparams["input"]
        self.seq_amount = (hparams["input"]=="title+abstract")+1
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}
        
        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        
        self.model = nn.Sequential(
            nn.Dropout(self.hparams["drop"]),            
            #nn.Linear(768, self.hparams["n_hidden_1"]),
            #nn.PReLU(),
            nn.Linear(self.seq_amount*768, self.hparams["n_hidden_out"]),
            #nn.Dropout(self.hparams["drop"]),
            nn.Sigmoid(),
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
    def forward_title(self, input_ids, masks):
        
        x1 = self.bert_t(input_ids, masks)
        # For DistilBert:
        x1 = x1[0][:,0]
        x = x1
        x = self.model(x)
        #x = self.model(x)
        return x
    
    def forward_abstract(self, input_ids, masks):
        
        x1 = self.bert_a(input_ids, masks)
        # For DistilBert:
        x1 = x1[0][:,0]
        x = x1
        x = self.model(x)
        #x = self.model(x)
        return x
    
    def forward_title_abstract(self, input_ids_t, masks_t, input_ids_a, masks_a):
        
        x1 = self.bert_t(input_ids_t, masks_t)
        x2 = self.bert_a(input_ids_a, masks_a)
        # For DistilBert:
        x1 = x1[0][:,0]
        x2 = x2[0][:,0]
        x = torch.cat((x1, x2),1)
        x = self.model(x)
        #x = self.model(x)
        return x
        
    
    def forward(self, input_ids_t, masks_t, input_ids_a, masks_a):
        
        if self.input_type=="title":
            x = self.forward_title(input_ids_t, masks_t)
        elif self.input_type=="abstract":
            x = self.forward_abstract(input_ids_a, masks_a)
        elif self.input_type=="title+abstract":
            x = self.forward_title_abstract(input_ids_t, masks_t, input_ids_a, masks_a)
        else:
            AssertionError
            
        return x

    def general_step(self, batch, batch_idx, mode):
        
        if self.input_type=="title":
            masks_t = batch['attention_mask']
            input_ids_t = batch['input_ids'].squeeze(1)
            masks_a = None
            input_ids_a = None
        elif self.input_type=="abstract":
            masks_t = None
            input_ids_t = None
            masks_a = batch['attention_mask']
            input_ids_a = batch['input_ids'].squeeze(1)
        elif self.input_type=="title+abstract":
            masks_t = batch['attention_mask_t']
            input_ids_t = batch['input_ids_t'].squeeze(1)
            masks_a = batch['attention_mask_a']
            input_ids_a = batch['input_ids_a'].squeeze(1)
        else:
            AssertionError
        
        targets = batch['labels']

        # forward pass
        out = self.forward(input_ids_t, masks_t, input_ids_a, masks_a)
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
        
        f1 = f1_score(preds.cpu().numpy(), targets.cpu().numpy(),average='weighted')
        return loss, n_correct, torch.tensor(f1).cuda()

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack(
            [x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.data[mode])
        f1 = torch.stack([x[mode + '_f1'] for x in outputs]).mean()
        '''
        preds = torch.stack(
            [x[mode + '_preds'] for x in outputs]).cpu().numpy()
        targets = torch.stack(
            [x[mode + '_targets'] for x in outputs]).cpu().numpy()
        f1 = f1_score(preds[0], targets[0],average='micro')
        '''
        return avg_loss, acc, f1 

    def training_step(self, batch, batch_idx):
        loss, n_correct, f1 = self.general_step(batch, batch_idx, "train")
        #self.log("train_loss", loss)
        #self.log("loss", {"train": loss})
        tensorboard_logs = {'loss': loss}
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct': n_correct, 'train_f1': f1, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct, f1 = self.general_step(batch, batch_idx, "val")
        tensorboard_logs = {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1': f1}
        #self.log("val_loss", loss)     # try this line
        return {'val_loss': loss, 'val_n_correct': n_correct, 'val_f1': f1, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        loss, n_correct, f1 = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct, 'f1': f1}

    def validation_epoch_end(self, outputs):
        avg_loss, acc, f1 = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc={}".format(acc))
        print("Val-F1={}".format(f1))
        #self.global_step=self.gst
        self.log('val_loss', avg_loss) 
        #self.log('val_acc', acc)
        self.log('val_f1', f1)
        #self.log("loss", {"val": avg_loss})
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc, self.global_step)
        self.logger.experiment.add_scalar("val_f1", f1, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1}
        return {'val_loss': avg_loss, 'val_acc': acc, 'val_f1': f1, 'log': tensorboard_logs}
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.data['train'], shuffle=True, batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.data['val'], batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.data['test'], batch_size=self.hparams['batch_size'], drop_last=True, num_workers=12)
    
    def configure_optimizers(self):

        optim = None
        ########################################################################
        # Define  optimizer:                                                   #
        ########################################################################
        params = list(self.bert_t.parameters()) + list(self.bert_a.parameters()) + list(self.model.parameters())
        optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=self.hparams['gamma'])
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=0.01, steps_per_epoch=len(self.data['train']), epochs=500)

        return [optim], [scheduler]
    
class MTL_YC_Classifier(pl.LightningModule):

    def __init__(self, hparams, bert, train_set=None, val_set=None, test_set=None):
        super().__init__()
        # set hyperparams
        self.hparams.update(hparams)
        self.save_hyperparameters(self.hparams)
        self.bert = bert
        self.y_classifier = nn.Identity()
        self.c_classifier = nn.Identity()
        
        self.data = {'train': train_set,
                     'val': val_set,
                     'test': test_set}

        ########################################################################
        # Initialize classifier:                                               #
        ########################################################################
        
        self.y_classifier = nn.Sequential(
            #nn.Dropout(self.hparams["drop"]),
            nn.Linear(768, self.hparams["n_hidden_1"]),
            nn.PReLU(),
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out"]),
            nn.Dropout(self.hparams["drop"]),
            #nn.Sigmoid(),
            #nn.PReLU(),
            #nn.Dropout(self.hparams["drop"]),
            #nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_3"])
        )
        
        self.c_classifier = nn.Sequential(
            #nn.Dropout(self.hparams["drop"]),
            nn.Linear(768, self.hparams["n_hidden_1"]),
            nn.PReLU(),
            nn.Linear(self.hparams["n_hidden_1"], self.hparams["n_hidden_out_c"]),
            nn.Dropout(self.hparams["drop"]),
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
        xy = self.y_classifier(x[:,0])
        xc = self.c_classifier(x[:,0])
        # For BERT:
        #x = self.model(x)
        return xy, xc

    def general_step(self, batch, batch_idx, mode):
        
        masks = batch['attention_mask']
        input_ids = batch['input_ids'].squeeze(1)
        targets_y = batch['labels_y']
        targets_c = batch['labels_c']
        
        '''
        input_ids = input_ids.to(0)
        masks = masks.to(0)
        self = self.to(0)
        '''
        # forward pass
        out_y, out_c = self.forward(input_ids, masks)
        # loss
        targets_y = targets_y.reshape(len(targets_y), self.hparams["n_hidden_out"])
        targets_c = targets_c.reshape(len(targets_c), self.hparams["n_hidden_out_c"])

        out_y = out_y.to(torch.float32)
        out_c = out_c.to(torch.float32)

        # For Multi-Class Classification
        targets_y = torch.max(targets_y, 1)[1].long()
        targets_c = torch.max(targets_c, 1)[1].long()
        loss_func = nn.CrossEntropyLoss()
        loss_y = loss_func(out_y, targets_y)
        loss_c = loss_func(out_c, targets_c)
        # For Binary Classification:
        #loss_func = nn.BCELoss()
        #loss = loss_func(out, targets.type(torch.float))
        #targets = targets.argmax(axis=1)
        preds_y = out_y.argmax(axis=1)
        preds_c = out_c.argmax(axis=1)
        n_correct_y = (targets_y == preds_y).sum()
        n_correct_c = (targets_c == preds_c).sum()
        
        # Final Loss:
        loss = 1*loss_y + 1*loss_c
        
        return loss, n_correct_y, n_correct_c, preds_y, preds_c, targets_y, targets_c

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct_y = torch.stack(
            [x[mode + '_n_correct_y'] for x in outputs]).sum().cpu().numpy()
        total_correct_c = torch.stack(
            [x[mode + '_n_correct_c'] for x in outputs]).sum().cpu().numpy()
        preds_y = torch.stack(
            [x[mode + '_preds_y'] for x in outputs]).cpu().numpy()
        preds_c = torch.stack(
            [x[mode + '_preds_c'] for x in outputs]).cpu().numpy()
        targets_y = torch.stack(
            [x[mode + '_targets_y'] for x in outputs]).cpu().numpy()
        targets_c = torch.stack(
            [x[mode + '_targets_c'] for x in outputs]).cpu().numpy()
        acc_y = total_correct_y / len(self.data[mode])
        acc_c = total_correct_c / len(self.data[mode])
        f1_y = f1_score(preds_y[0], targets_y[0],average='weighted')
        f1_c = f1_score(preds_c[0], targets_c[0],average='weighted')
        return avg_loss, acc_y, acc_c, f1_y, f1_c

    def training_step(self, batch, batch_idx):
        loss, n_correct_y, n_correct_c, preds_y, preds_c, targets_y, targets_c = self.general_step(batch, batch_idx, "train")
        tensorboard_logs = {'loss': loss}
        n_correct_y = n_correct_y.double()
        n_correct_c = n_correct_c.double()
        self.logger.experiment.add_scalar("train_loss", loss, self.global_step)
        return {'loss': loss, 'train_n_correct_y': n_correct_y, 'train_n_correct_c': n_correct_c,'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss, n_correct_y, n_correct_c, preds_y, preds_c, targets_y, targets_c = self.general_step(batch, batch_idx, "val")
        return {'val_loss': loss, 'val_n_correct_y': n_correct_y, 'val_n_correct_c': n_correct_c,'val_preds_y': preds_y, 'val_preds_c': preds_c, 'val_targets_y': targets_y, 'val_targets_c': targets_c}

    def test_step(self, batch, batch_idx):
        loss, n_correct_y, n_correct_c, preds_y, preds_c, targets_y, targets_c = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct_y': n_correct_y, 'test_n_correct_c': n_correct_c}

    def validation_epoch_end(self, outputs):
        avg_loss, acc_y, acc_c, f1_y, f1_c = self.general_end(outputs, "val")
        print("Val-Loss={}".format(avg_loss))
        print("Val-Acc-Year={}".format(acc_y))
        print("Val-Acc-Cite={}".format(acc_c))
        print("Val-F1-Year={}".format(f1_y))
        print("Val-F1-Cite={}".format(f1_c))
        self.log("val_loss", avg_loss) 
        self.logger.experiment.add_scalar("val_loss", avg_loss, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc_y, self.global_step)
        self.logger.experiment.add_scalar("val_f1", f1_y, self.global_step)
        self.logger.experiment.add_scalar("val_acc", acc_c, self.global_step)
        self.logger.experiment.add_scalar("val_f1", f1_c, self.global_step)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc_y': acc_y, 'val_acc_c': acc_c, 'val_f1_y': f1_y, 'val_f1_c': f1_c}
        return {'val_loss': avg_loss, 'val_acc_y': acc_y, 'val_acc_c': acc_c, 'val_f1_y': f1_y, 'val_f1_c': f1_c, 'log': tensorboard_logs}
    
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
        params = list(self.bert.parameters()) + list(self.model.parameters())
        #optim = torch.optim.Adam(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        optim = torch.optim.AdamW(params=params,betas=(0.9, 0.999),lr=self.hparams['lr'], weight_decay=self.hparams['reg'])
        #optim = torch.optim.SGD(params=params, lr = self.hparams['lr'], weight_decay=self.hparams['reg'],momentum=0.9)
        
        # Define LR-Scheduler:
        #num_training_steps = int(len(self.data['train'])/self.hparams['batch_size'])*6
        #num_warmup_steps = num_training_steps*0.1
        #scheduler = torch.optim.get_linear_schedule_with_warmup(optim, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        return optim

    def MAcc(out, targets):
      preds = np.zeros(len(out[0]))
      for i in range(0, len(out)):
        for j in range(0, len(out[i])):
          if out[i,j]>=0.5:
            preds[j] = 1
      n_correct = (targets == preds).sum()

      return n_correct
     
