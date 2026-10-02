import json
import random
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def metrics(y_true, y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    return {'accuracy':accuracy_score(y_true,y_pred),'precision':p,'recall':r,'f1':f,'confusion_matrix':confusion_matrix(y_true,y_pred)}

def save_json(obj,path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    def conv(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,(np.float32,np.float64)): return float(x)
        if isinstance(x,(np.int32,np.int64)): return int(x)
        return x
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,indent=2,default=conv)
