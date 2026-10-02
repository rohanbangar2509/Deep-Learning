import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms,models
from src.train import CNNClassifier
CLASSES=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def load_vit():
    m=models.vit_b_16(weights=None); m.heads.head=nn.Linear(m.heads.head.in_features,10); m.load_state_dict(torch.load(ROOT/'models'/'vit_b16'/'vit-b16_cifar10.pth',map_location=DEVICE,weights_only=True)); return m.to(DEVICE).eval()
def pred(m,img,t):
    x=t(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): p=torch.softmax(m(x),1)[0]; i=p.argmax().item()
    return CLASSES[i],float(p[i])
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--image',required=True); a=ap.parse_args(); img=Image.open(a.image).convert('RGB')
    cm=transforms.Compose([transforms.ToTensor(),transforms.Normalize((.4914,.4822,.4465),(.247,.2435,.2616))]); vm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
    cnn=CNNClassifier().to(DEVICE).eval(); cnn.load_state_dict(torch.load(ROOT/'models'/'cnn'/'cnn_cifar10.pth',map_location=DEVICE,weights_only=True)); vit=load_vit()
    a1,c1=pred(cnn,img,cm); a2,c2=pred(vit,img,vm); print('Device:',DEVICE); print(f'CNN Prediction : {a1} ({c1:.4f})'); print(f'ViT Prediction : {a2} ({c2:.4f})')
if __name__=='__main__': main()
