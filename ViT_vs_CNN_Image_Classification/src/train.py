from pathlib import Path
import sys, json, random
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Subset,random_split
from torchvision import datasets,transforms,models
from torchvision.models import ViT_B_16_Weights
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from tqdm.auto import tqdm
from src.utils import set_seed,count_parameters,metrics,save_json

SEED=42; TRAIN_SAMPLES=10000; TEST_SAMPLES=2000; EPOCHS_CNN=5; EPOCHS_VIT=3; BATCH_SIZE_CNN=64; BATCH_SIZE_VIT=8; NUM_WORKERS=2
DATA_DIR=ROOT/'data'; MODEL_DIR=ROOT/'models'; OUTPUT_DIR=ROOT/'outputs'; NUM_CLASSES=10

class CNNClassifier(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(3,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),
            nn.Conv2d(64,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),nn.BatchNorm2d(256),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d((1,1)))
        self.classifier=nn.Sequential(nn.Flatten(),nn.Dropout(.3),nn.Linear(256,num_classes))
    def forward(self,x): return self.classifier(self.features(x))

def build_vit():
    m=models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    m.heads.head=nn.Linear(m.heads.head.in_features,NUM_CLASSES)
    return m

def make_loaders():
    cm=(.4914,.4822,.4465); cs=(.2470,.2435,.2616); im=(.485,.456,.406); ist=(.229,.224,.225)
    ctrain=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(cm,cs)])
    ctest=transforms.Compose([transforms.ToTensor(),transforms.Normalize(cm,cs)])
    vtrain=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.8,1.0)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(im,ist)])
    vtest=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize(im,ist)])
    ct=datasets.CIFAR10(DATA_DIR,train=True,download=True,transform=ctrain); cte=datasets.CIFAR10(DATA_DIR,train=False,download=True,transform=ctest)
    vt=datasets.CIFAR10(DATA_DIR,train=True,download=True,transform=vtrain); vte=datasets.CIFAR10(DATA_DIR,train=False,download=True,transform=vtest)
    if TRAIN_SAMPLES: ct=Subset(ct,range(TRAIN_SAMPLES)); vt=Subset(vt,range(TRAIN_SAMPLES))
    if TEST_SAMPLES: cte=Subset(cte,range(TEST_SAMPLES)); vte=Subset(vte,range(TEST_SAMPLES))
    common=dict(num_workers=NUM_WORKERS,pin_memory=torch.cuda.is_available())
    
    # Create a validation split from the training data; keep test data untouched.
    g=torch.Generator().manual_seed(SEED)
    n_val=max(1,int(0.1*len(ct))); n_train=len(ct)-n_val
    ctrain,cval=random_split(ct,[n_train,n_val],generator=g)
    g=torch.Generator().manual_seed(SEED)
    vtrain,vval=random_split(vt,[n_train,n_val],generator=g)
    return (DataLoader(ctrain,batch_size=BATCH_SIZE_CNN,shuffle=True,**common),DataLoader(cval,batch_size=BATCH_SIZE_CNN,shuffle=False,**common),DataLoader(cte,batch_size=BATCH_SIZE_CNN,shuffle=False,**common),DataLoader(vtrain,batch_size=BATCH_SIZE_VIT,shuffle=True,**common),DataLoader(vval,batch_size=BATCH_SIZE_VIT,shuffle=False,**common),DataLoader(vte,batch_size=BATCH_SIZE_VIT,shuffle=False,**common))

def run_epoch(model,loader,criterion,device,optimizer=None,scaler=None):
    train=optimizer is not None; model.train(train); loss_sum=0.; ys=[]; ps=[]
    for x,y in tqdm(loader,leave=False):
        x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
        if train: optimizer.zero_grad(set_to_none=True)
        amp=scaler is not None
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp): out=model(x); loss=criterion(out,y)
        if train:
            if scaler: scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            else: loss.backward(); optimizer.step()
        loss_sum+=loss.item()*x.size(0); ys.extend(y.cpu().numpy()); ps.extend(out.argmax(1).detach().cpu().numpy())
    m=metrics(ys,ps); m['loss']=loss_sum/len(loader.dataset); m['y_true']=ys; m['y_pred']=ps; return m

def train_model(name,model,train_loader,test_loader,epochs,lr,weight_decay,device):
    criterion=nn.CrossEntropyLoss(); opt=optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay); sched=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs); scaler=torch.amp.GradScaler('cuda') if device.type=='cuda' else None
    hist={'train_loss':[],'train_accuracy':[],'test_loss':[],'test_accuracy':[]}; best=-1; path=MODEL_DIR/name.lower().replace('-','_')
    path.mkdir(parents=True,exist_ok=True)
    for e in range(epochs):
        tr=run_epoch(model,train_loader,criterion,device,opt,scaler); te=run_epoch(model,test_loader,criterion,device); sched.step()
        for k,v in [('train_loss',tr['loss']),('train_accuracy',tr['accuracy']),('test_loss',te['loss']),('test_accuracy',te['accuracy'])]: hist[k].append(v)
        print(f'{name} | Epoch {e+1}/{epochs} | Train Acc {tr["accuracy"]:.4f} | Test Acc {te["accuracy"]:.4f}')
        if te['accuracy']>best: best=te['accuracy']; torch.save(model.state_dict(),path/f'{name.lower().replace("/", "_")}_cifar10.pth')
    save_json(hist,OUTPUT_DIR/'metrics'/f'{name.lower().replace("/","_")}_history.json'); return hist

def main():
    set_seed(SEED); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device); print('GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU')
    cl,cval,cte,vl,vval,vte=make_loaders()
    cnn=CNNClassifier().to(device); vit=build_vit().to(device)
    print('CNN parameters:',f'{count_parameters(cnn):,}'); print('ViT parameters:',f'{count_parameters(vit):,}')
    train_model('CNN',cnn,cl,cval,EPOCHS_CNN,1e-3,1e-4,device)
    train_model('ViT-B16',vit,vl,vval,EPOCHS_VIT,2e-5,.01,device)
    criterion=nn.CrossEntropyLoss()
    cnn.load_state_dict(torch.load(MODEL_DIR/'cnn'/'cnn_cifar10.pth',map_location=device,weights_only=True)); vit.load_state_dict(torch.load(MODEL_DIR/'vit_b16'/'vit-b16_cifar10.pth',map_location=device,weights_only=True))
    cm=run_epoch(cnn,cte,criterion,device); vm=run_epoch(vit,vte,criterion,device)
    comp={'CNN':{k:cm[k] for k in ['accuracy','precision','recall','f1']},'ViT-B/16':{k:vm[k] for k in ['accuracy','precision','recall','f1']}}
    save_json(cm,OUTPUT_DIR/'metrics'/'cnn_test_metrics.json'); save_json(vm,OUTPUT_DIR/'metrics'/'vit_test_metrics.json'); save_json(comp,OUTPUT_DIR/'metrics'/'model_comparison.json')
    print('\nFINAL COMPARISON'); print(json.dumps(comp,indent=2))
if __name__=='__main__': main()
