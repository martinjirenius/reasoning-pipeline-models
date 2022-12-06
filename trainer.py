from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm

import sampler
import datasets
import models

class CLS_Trainer():
    
    def __init__(self,
                 data_path,
                 save_path,
                 split = [0,1],
                 model_path = None,
                 num_labels = 2,
                 stratify = None,
                 sample_weights = None,
                 batch_size = 16,
                 learning_rate = 5e-5,
                 num_epochs = 3,
                 warmup_percent = 1,
                ):
        
        self.dataset = datasets.CLS_Dataset(data_path, split)
        self.save_path = save_path
        
        if model_path is None:
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels = num_labels)
        else:
            self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.wrs = sampler.WeightedRandomSampler(self.dataset.encodings['labels'], stratify = stratify, weights = sample_weights)
        sampler = torch.utils.data.sampler.BatchSampler(self.wrs, batch_size=batch_size,drop_last=False)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast = True)
        
        dataCollator = DataCollatorWithPadding(tokenizer = self.tokenizer)
        self.train_loader = DataLoader(dataset = self.dataset, 
                                       batch_size = None, 
                                       collate_fn = lambda x: dataCollator(x[0]), 
                                       sampler = sampler)
        
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.num_warmup_steps = round(num_epochs * len(self.train_loader) * warmup_percent / (batch_size * 100)) * batch_size
        self.num_training_steps = num_epochs * len(self.train_loader)
        self.lr_scheduler = get_scheduler("linear", 
                                     optimizer = self.optimizer, 
                                     num_warmup_steps = self.num_warmup_steps, 
                                     num_training_steps = self.num_training_steps)
    
    def train(self, save = True, evaluator = None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        
        progress_bar_train = tqdm(range(self.num_training_steps))
        
        losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                enc = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**enc)
                loss = outputs.loss
                losses += [loss.detach().cpu()]
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar_train.update(1)
                
                if i % round(len(self.train_loader)/3) == 1:
                    print(f'Average loss: {sum(losses)/len(losses)}')
                    losses = []
            if evaluator is not None:
                self.model.eval()
                evaluator.evaluate(self.model)
        if save:
            self.model.save_pretrained(self.save_path)

class LM_Trainer():
    
    def __init__(self,
                 data_path,
                 save_path,
                 split = [0,1],
                 model_path = 'gpt2',
                 num_labels = 2,
                 stratify = None,
                 sample_weights = None,
                 batch_size = 16,
                 learning_rate = 5e-5,
                 num_epochs = 3,
                 warmup_percent = 1,
                ):
        
        encodings =  {'attention_mask': 'gold_mask', 'input_ids': 'gold_ids', 'labels': 'gold_ids'}
        self.dataset = datasets.LM_Dataset(data_path, encodings, split)
        self.save_path = save_path
        
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        self.wrs = sampler.WeightedRandomSampler(self.dataset.others['labels'], stratify = stratify, weights = sample_weights)
        sampler = torch.utils.data.sampler.BatchSampler(self.wrs, batch_size=batch_size,drop_last=False)
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', use_fast = True)

        dataCollator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm = False)
        self.train_loader = DataLoader(dataset = self.dataset, 
                                       batch_size = None, 
                                       collate_fn = self.train_collator, 
                                       sampler = sampler)
        
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.num_warmup_steps = round(num_epochs * len(self.train_loader) * warmup_percent / (batch_size * 100)) * batch_size
        self.num_training_steps = num_epochs * len(self.train_loader)
        self.lr_scheduler = get_scheduler("linear", 
                                     optimizer = self.optimizer, 
                                     num_warmup_steps = self.num_warmup_steps, 
                                     num_training_steps = self.num_training_steps)
    
    def train_collator(self, batch):
        encs = batch[0]
        max_len = max([len(l) for l in encs['input_ids']])
        
        pad = {'input_ids': self.tokenizer.eos_token_id, 'labels': -100, 'attention_mask': 0}
        for key in encs:
            for sample in encs[key]:
                encs[key] = torch.tensor(sample + [pad[key]] * (max_len - len(sample)))

        return encs
    
    def train(self, save = True, evaluator = None):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        
        progress_bar_train = tqdm(range(self.num_training_steps))
        
        losses = []
        for epoch in range(self.num_epochs):
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                enc = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**enc)
                loss = outputs.loss
                losses += [loss.detach().cpu()]
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar_train.update(1)
                
                if i % round(len(self.train_loader)/3) == 1:
                    print(f'Average loss: {sum(losses)/len(losses)}')
                    losses = []
            if evaluator is not None:
                self.model.eval()
                evaluator.evaluate(self.model)
        if save:
            self.model.save_pretrained(self.save_path)

class MT_Trainer():
    
    def __init__(self,
                 data_paths,
                 save_path,
                 eval_paths = None,
                 split = [0,1],
                 model_path = 'gpt2',
                 stratify = None,
                 sample_weights = None,
                 batch_size = 16,
                 learning_rate = 5e-5,
                 num_epochs = 3,
                 warmup_percent = 1,
                ):
        
        self.dataset = datasets.DatasetForMultiTaskLearning(data_paths, split)
        self.save_path = save_path
        
        wrs = sampler.WeightedRandomSampler(self.dataset.encodings['cls_labels'], stratify = stratify, weights = sample_weights)
        sampler = torch.utils.data.sampler.BatchSampler(wrs, batch_size=batch_size,drop_last=False)
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', use_fast = True)
        
        self.train_loader = DataLoader(dataset = self.dataset, 
                                       batch_size = None, 
                                       collate_fn = self.train_collator, 
                                       sampler = sampler)
        self.eval_loader = None
        self.eval_dataset = None
        if eval_paths is not None:
            self.eval_dataset = DatasetForMultiTaskLearning(eval_paths, split)
            wrs = WeightedRandomSampler(self.eval_dataset.encodings['cls_labels'], stratify = stratify, weights = sample_weights)
            sampler = torch.utils.data.sampler.BatchSampler(wrs, batch_size=batch_size,drop_last=False)
            self.eval_loader = DataLoader(dataset = self.eval_dataset, 
                                       batch_size = None, 
                                       collate_fn = self.train_collator, 
                                       sampler = sampler)
        num_labels = len(pd.unique(self.dataset.encodings['cls_labels']))
        config = models.GPT2ForMultiTaskConfig(n_labels = num_labels, pad_token_id = self.tokenizer.eos_token_id)
        self.model = models.GPT2ForMultiTaskLearning.from_pretrained(model_path, config = config)
        
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.num_epochs = num_epochs
        self.num_warmup_steps = round(num_epochs * len(self.train_loader) * warmup_percent / (batch_size * 100)) * batch_size
        self.num_training_steps = num_epochs * len(self.train_loader)
        self.lr_scheduler = get_scheduler("linear", 
                                     optimizer = self.optimizer, 
                                     num_warmup_steps = self.num_warmup_steps, 
                                     num_training_steps = self.num_training_steps)
    
    def train_collator(self, batch):
        max_len = max([len(l) for l in batch[0]['gold_ids']])
        
        pads = {'input_ids': self.tokenizer.eos_token_id, 'lm_labels': -100, 'attention_mask': 0}
        keys = {'gold_ids': 'input_ids', 'gold_mask': 'attention_mask', 'lm_labels': 'lm_labels'}
        out = {}
        for key, val in keys.items():
            out[val] = torch.tensor([sample + [pads[val]] * (max_len - len(sample)) for sample in batch[0][key]])
        out['cls_labels'] = torch.tensor(batch[0]['cls_labels'])
        return out
    
    def train(self, save = True):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        
        progress_bar_train = tqdm(range(self.num_training_steps))
        self.losses = []
        for epoch in range(self.num_epochs):
            if self.eval_loader is not None:
                self.evaluate()
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                enc = {k: v.to(device) for k, v in batch.items()}
                
                outputs = self.model(**enc)
                loss = outputs.loss + outputs.cls_loss
                #loss = outputs.cls_loss
                self.losses += [loss.detach().cpu()]
                loss.backward()

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar_train.update(1)
                
                if i % round(len(self.train_loader)/3) == 1:
                    print(f'Average training loss: {sum(self.losses)/len(self.losses)}')
                    
            if save:
                self.model.save_pretrained(f'{self.save_path}_{epoch}')
                
        if self.eval_loader is not None:
            self.evaluate()
    
    def evaluate(self):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.eval()
        lm_losses = []
        cls_losses = []
        cls_preds = []
        cls_true = []
        progress_bar_eval = tqdm(range(len(self.eval_loader)))
        for i, batch in enumerate(self.eval_loader):
            enc = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**enc)
            lm_losses += [outputs.loss.detach().cpu()]
            cls_losses += [outputs.cls_loss.detach().cpu()]
            vals, predictions = torch.max(outputs.cls_logits, dim=-1)
            cls_preds += predictions.detach().cpu().tolist()#outputs.cls_logits.detach().cpu().tolist()
            cls_true += enc['cls_labels'].cpu().tolist()
            progress_bar_eval.update(1)

        print(f'Average language modeling loss: {sum(lm_losses)/len(lm_losses)}')
        print(f'Average classification loss: {sum(cls_losses)/len(cls_losses)}')
        print(f'Average classification accuracy: {sum([x == y for x, y in zip(cls_preds, cls_true)])/len(cls_true)}')
        losses = []
        alpha = (sum(lm_losses)/len(lm_losses)) / (sum(cls_losses)/len(cls_losses))
        print(f'Alpha = {alpha}')
        return alpha