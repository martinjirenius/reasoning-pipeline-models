import torch
import pandas as pd
from ast import literal_eval
from tqdm import tqdm
import numpy as np

class CLS_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, split = [0,1]):
        
        encs = ['attention_mask', 'input_ids', 'token_type_ids']
        dataset = pd.read_csv(f'{data_path}', header = 0, index_col = 0).reset_index()
        print(pd.unique(dataset['labels']))
        dataset = dataset[round(len(dataset)*split[0]): round(len(dataset)*split[1])]
        dataset['labels'] = dataset['labels'].apply(lambda lbl: int(lbl))
        tmp = dataset[dataset['labels'] == 1].groupby(['groups']).first().reset_index()
        dataset = dataset[dataset['labels'] == 0].append(tmp).reset_index()
        
        dataset['labels'] += addit
        
        tqdm.pandas()
        
        self.encodings = dataset[encs].progress_applymap(literal_eval)
        self.encodings['labels'] = dataset['labels']
        self.others = dataset.drop(encs + ['labels'], axis=1)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings.loc[idx].to_dict('list'), self.others.loc[idx]

class LM_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_path, encoding_map, split = [0,1]):
        
        dataset = pd.read_csv(f'{data_path}', header = 0, index_col = 0).reset_index()
        dataset = dataset[round(len(dataset)*split[0]): round(len(dataset)*split[1])]
        tqdm.pandas()
        
        self.encodings = pd.DataFrame({key: dataset[val].progress_map(literal_eval) for key, val in encoding_map.items()})
        self.others = dataset.drop(encoding_map.values(), axis=1)
        self.others['labels'] = dataset['labels'].apply(lambda lbl: int(lbl))

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings.loc[idx].to_dict('list'), self.others.loc[idx]

class MT_Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_paths, split = [0,1]):
        
        self.encodings = pd.DataFrame(columns = ['cls_labels'])
        self.others = pd.DataFrame()
        for (data_path, data_id) in data_paths:
            dataset = pd.read_csv(f'{data_path}', header = 0, index_col = 0)
            dataset = dataset[round(len(dataset)*split[0]): round(len(dataset)*split[1])]
            #dataset = dataset.dropna().groupby([sen1, sen2]).first().reset_index()
            dataset['data_id'] = data_id
            lit_evals = ['input_ids', 'attention_mask']
            if 'gold_ids' in dataset:
                lit_evals += ['gold_ids', 'gold_mask']
    
            tqdm.pandas()        
            dataset[lit_evals] = dataset[lit_evals].progress_applymap(literal_eval)
            
            
            num_labels = len(pd.unique(self.encodings['cls_labels']))
            
            lbls = np.sort(pd.unique(dataset['labels']))
            
            dataset['cls_labels'] = dataset['labels'].apply(lambda lbl: np.where(lbls == lbl)[0][0] + num_labels)
            tmp = dataset[dataset['cls_labels'] == 1].groupby(['groups']).first().reset_index()
            dataset = dataset[dataset['cls_labels'] == 0].append(tmp).reset_index()
            
            dataset['cls_labels'] += addit
        
            self.encodings = self.encodings.append(dataset[lit_evals + ['cls_labels']])
            self.others = self.others.append(dataset.drop(lit_evals + ['cls_labels'], axis=1))
        if 'gold_ids' in self.encodings:
            lm_labels = lambda x: [-100]*len(x['input_ids']) + x['gold_ids'][(len(x['input_ids'])):]
            self.encodings['lm_labels'] = self.encodings.apply(lm_labels, axis=1)
        self.encodings = self.encodings.reset_index()
        self.others = self.others.reset_index()
        
        
    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings.loc[idx].to_dict('list'), self.others.loc[idx]