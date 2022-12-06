from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import DataCollatorWithPadding
from transformers import GPT2LMHeadModel
from transformers import GPT2TokenizerFast
from transformers import DataCollatorForLanguageModeling
from transformers import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from datasets import load_metric
from sklearn import metrics
import json

import sampler
import datasets
import models

class CLS_Evaluator():
    
    def __init__(self,
                 data_path,
                 split = [0,1],
                 stratify = None,
                 sample_weights = None,
                 batch_size = 16
                ):
        
        self.dataset = datasets.CLS_Dataset(data_path, split)
        wrs = sampler.WeightedRandomSampler(self.dataset.encodings['labels'], stratify = stratify, weights = sample_weights)
        sampler = torch.utils.data.sampler.BatchSampler(wrs, batch_size=batch_size,drop_last=False)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', use_fast = True)
        dataCollator = DataCollatorWithPadding(tokenizer = self.tokenizer)
        self.eval_loader = DataLoader(dataset = self.dataset, 
                                       batch_size = None, 
                                       collate_fn = lambda x: (dataCollator(x[0]), x[1]), 
                                       sampler = sampler)
        
    
    def evaluate(self, model):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        groups = []
        lgits = []
        lgits_bin = []
        preds = []
        refs = []
        opts = []

        model.eval()
        self.progress_bar_eval = tqdm(range(len(self.eval_loader)))
        self.progress_bar_eval.reset()
        for i, (enc, others) in enumerate(self.eval_loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = model(**enc)
            logits = outputs.logits
            vals, predictions = torch.max(logits, dim=-1)
            groups += others['groups'].tolist()
            lgits_bin += logits[:,1+addit].detach().cpu()
            lgits += vals.detach().cpu()
            preds += predictions.detach().cpu() - addit
            refs += enc['labels'].detach().cpu() - addit

            self.progress_bar_eval.update(1)
        
        df = pd.DataFrame({'groups': groups, 'logits': lgits, 'logits_bin': lgits_bin, 'preds': preds, 'refs': refs})
        print("-"*20 + "----------" + "-"*20)
        print(df.head())
        prec, rec, f1, dist = metrics.precision_recall_fscore_support(refs, preds, average=None)
        print("-"*20 + "EVALUATION" + "-"*20)
        print('Precision: \t\t{}'.format(prec))
        print('Recall: \t\t{}'.format(rec))
        print('F1: \t\t\t{}'.format(f1))
        print('Distribution: \t\t{}'.format(dist))
        
        print("Original task: \t\t{}".format(df[(df.groupby('groups'))['logits_bin'].transform(max) == df['logits_bin']]['refs'].mean()))
        
        tmp = df[(df.groupby('groups'))['logits'].transform(max) == df['logits']]
        print("Original task v2: \t{}".format(len(tmp[tmp['preds'] == tmp['refs']].groupby('groups').first())/len(df.groupby('groups').first())))
        
        print("-"*20 + "----------" + "-"*20)

class LM_Evaluator():
    
    def __init__(self,
                 data_path,
                 split = [0,1],
                 stratify = None,
                 sample_weights = None,
                 batch_size = 16,
                 s1 = "",
                 s2 = "",
                 save_path = "evaluator_output",
                 gold_explanations = True
                ):
        
        encodings =  {'attention_mask': 'attention_mask', 'input_ids': 'input_ids'}
        self.dataset = datasets.LM_Dataset(data_path, encodings, split)
        
        self.wrs = sampler.WeightedRandomSampler(self.dataset.others['labels'], stratify = stratify, weights = sample_weights)
        sampler = torch.utils.data.sampler.BatchSampler(self.wrs, batch_size=batch_size,drop_last=False)
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', use_fast = True)
        dataCollator = DataCollatorForLanguageModeling(tokenizer = self.tokenizer, mlm = False)
        self.test_loader = DataLoader(dataset = self.dataset, 
                                       batch_size = None, 
                                       collate_fn = self.test_collator, 
                                       sampler = sampler)
        
        self.bertscore = load_metric('bertscore')
        
        self.s1 = s1
        self.s2 = s2
        self.save_path = save_path
        self.gold_explanations = gold_explanations
        
    def test_collator(self, batch):
        max_len = max([len(l) for l in batch[0]['input_ids']])
        
        pad = {'input_ids': self.tokenizer.eos_token_id, 'attention_mask': 0}
        for key in batch[0]:
            batch[0][key] = torch.tensor([[pad[key]] * (max_len - len(sample)) + sample for sample in batch[0][key]])
                
        return batch[0], batch[1]
    
    def evaluate(self, model, save = True):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        progress_bar = tqdm(range(len(self.test_loader)))
        
        inputs = []
        explanations = []
        gold_exps = []
        groups = []
        s1s = []
        cls_labels = []
        s2s = []
        model.eval()
        for i, batch in enumerate(self.test_loader):
            enc = {k: v.to(device) for k, v in batch[0].items()}
            with torch.no_grad():
                generation = model.generate(
                    input_ids = enc['input_ids'],
                    attention_mask = enc['attention_mask'],
                    do_sample = True,
                    max_length = 100,
                    temperature = 0.7,
                    top_k = 50,
                    top_p = 0.7
                )
                batch_input = self.tokenizer.batch_decode(enc['input_ids'].detach(), skip_special_tokens=True)
                groups += batch[1]['groups'].tolist() #[int(id) for id in batch['other']['id']]
                s1s += batch[1][self.s1].tolist()
                s2s += batch[1][self.s2].tolist()
                cls_labels += [int(cls) for cls in batch[1]['labels']]
                preds = []
                for i, e in zip(batch_input, self.tokenizer.batch_decode(generation.detach(), skip_special_tokens=True)):
                    preds += [e[len(i):].strip()]
                
                if self.gold_explanations: 
                    golds = batch[1]['explanations'].tolist()
                    self.bertscore.add_batch(predictions=preds, references=golds)
                    gold_exps += golds
                    
                inputs += batch_input
                explanations += preds

            progress_bar.update(1)
        if save:
            if self.gold_explanations:
                with open('../../data/generated/{}.json'.format(self.save_path), 'w') as f:
                    json.dump({"groups": groups, 
                               self.s1: s1s, 
                               self.s2: s2s, 
                               "generated": explanations, 
                               "gold_explanations": gold_exps,
                               "labels": cls_labels}, 
                              f)

                res = self.bertscore.compute(lang = 'en')
                print(f'{round(sum(res["precision"])/len(res["precision"]), 2)}\t\t|\t'+
                      f'{round(sum(res["recall"])/len(res["recall"]), 2)}\t\t|\t'+
                      f'{round(sum(res["f1"])/len(res["f1"]), 2)}')

                with open('../../data/generated/{}_bertscores.json'.format(self.save_path), 'w') as f:
                    json.dump({'Precision': round(sum(res["precision"])/len(res["precision"]), 2),
                               'Recall': round(sum(res["recall"])/len(res["recall"]), 2),
                               'F1-score': round(sum(res["f1"])/len(res["f1"]), 2)}, 
                              f)
            else:
                with open('../../data/generated/{}.json'.format(save_path), 'w') as f:
                    json.dump({"groups": groups, 
                               self.s1: s1s, 
                               self.s2: s2s, 
                               "generated": explanations, 
                               "labels": cls_labels}, 
                              f)

            with open('../../data/generated/{}.json'.format(self.save_path), 'r') as f:
                json_explanations = json.load(f)

class MT_Evaluator():
    
    def __init__(self,
                 data_paths,
                 split = [0,1],
                 stratify = None,
                 sample_weights = None,
                 batch_size = 16,
                ):
        
        self.dataset = datasets.DatasetForMultiTaskLearning(data_paths, split)
        
        self.wrs = sampler.WeightedRandomSampler(self.dataset.encodings['cls_labels'], stratify = stratify, weights = sample_weights)
        sampler = torch.utils.data.sampler.BatchSampler(self.wrs, batch_size=batch_size,drop_last=False)
        
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', use_fast = True)
        self.eval_loader = DataLoader(dataset = self.dataset, 
                                       batch_size = None, 
                                       collate_fn = self.eval_collator, 
                                       sampler = sampler)
    
    def eval_collator(self, batch):
        max_len = max([len(l) for l in batch[0]['attention_mask']])
        
        pads = {'input_ids': self.tokenizer.eos_token_id, 'attention_mask': 0}
        keys = {'input_ids': 'input_ids', 'attention_mask': 'attention_mask'}
        out = {}
        for key, val in keys.items():
            out[val] = torch.tensor([sample + [pads[val]] * (max_len - len(sample)) for sample in batch[0][key]])
        out['cls_labels'] = torch.tensor(batch[0]['cls_labels'])
        return out, batch[1]
    
    def evaluate(self, model):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)
        
        groups = []
        lgits = []
        lgits_bin = []
        preds = []
        refs = []
        opts = []
        data_ids = []

        model.eval()
        self.progress_bar_eval = tqdm(range(len(self.eval_loader)))
        self.progress_bar_eval.reset()
        for i, (enc, others) in enumerate(self.eval_loader):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                outputs = model(**enc)
            logits = outputs.cls_logits
            vals, predictions = torch.max(logits, dim=-1)
            groups += others['groups'].tolist()
            lgits_bin += logits[:,1].detach().cpu()
            lgits += vals.detach().cpu()
            preds += predictions.detach().cpu()
            refs += enc['cls_labels'].tolist()
            data_ids += others['data_id'].tolist()

            self.progress_bar_eval.update(1)
            
        df = pd.DataFrame({'data_ids': data_ids, 'groups': groups, 'logits': lgits, 'logits_bin': lgits_bin, 'preds': preds, 'refs': refs})
        
        print("-"*20 + "----------" + "-"*20)
        prec, rec, f1, dist = metrics.precision_recall_fscore_support(refs, preds, average=None)
        print("-"*20 + "EVALUATION" + "-"*20)
        print('Precision: \t\t{}'.format(prec))
        print('Recall: \t\t{}'.format(rec))
        print('F1: \t\t\t{}'.format(f1))
        print('Distribution: \t\t{}'.format(dist))
        
        ecqa_task = df[(df.groupby('groups'))['logits_bin'].transform(max) == df['logits_bin']]
        ecqa_task = ecqa_task[ecqa_task['data_ids'] == 'ecqa']
        print("Original task ecqa: \t\t{}".format(ecqa_task['refs'].mean()))
        
        tmp = df[(df.groupby('groups'))['logits'].transform(max) == df['logits']]
        tmp = tmp[tmp['data_ids'] == 'esnli']
        print("Original task v2: \t{}".format(len(tmp[tmp['preds'] == tmp['refs']].groupby('groups').first())/len(df.groupby('groups').first())))
        
        print("-"*20 + "----------" + "-"*20)