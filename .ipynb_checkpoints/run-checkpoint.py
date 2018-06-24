import numpy as np
import torch
from torch.autograd import Variable,grad
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import seaborn as sns
import glob
from torchvision.transforms import Compose,ToTensor,ToPILImage,Resize,Normalize,CenterCrop
from torch.utils.data import Dataset, DataLoader

from models import D_net,G_net
import visdom
viz = visdom.Visdom()
data_dir ='Data/'
image_size=[3,128,128]
small_image_size=[3,57,57]
code = 'beta'
use_GPU=torch.cuda.is_available()
batch_size=64
device=1 # GPU device
latent_dim = 128
imgs_dir = glob.glob(data_dir+'*')
imgs = [Image.open(fil).resize(image_size[1:]) for fil in imgs_dir]
class PK_DATASET(Dataset):
    def __init__(self,imgs):
        self.data=imgs
        self.trans=Compose([
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.trans(self.data[idx])
def get_noise(batch_size=batch_size):
    return Variable(torch.rand(batch_size,latent_dim))
Pk_dataset=PK_DATASET(imgs)
label='WGan'
img_graph = viz.image(torch.ones(image_size[1:]),
                opts=dict(title=label+' generated img '+str(device)))
loss_graph=viz.line(
            Y=np.zeros((1,3)),
            opts=dict(
                fillarea=False,
                showlegend=True,
                legend=['D_loss','G_loss','W-distance'],
                width=400,
                height=400,
                xlabel='Iter',
                ylabel='Loss',
#                 ytype='log',
                title=label+' - loss curve  '+str(device),
            ))

d_iter = 1
g_iter = 1
epoch = 5000
Pk_dataloader=DataLoader(Pk_dataset,batch_size=batch_size,num_workers=1,shuffle=True)

d_model = D_net()
g_model = G_net(latent_dim)
d_model.apply(weights_init)
g_model.apply(weights_init)

d_optimizer = Adam(d_model.parameters(),lr=1e-4,betas=[0.01,.9])
g_optimizer = Adam(g_model.parameters(),lr=1e-4,betas=[0.01,.9])
if use_GPU:
    d_model.cuda(device)
    g_model.cuda(device)
for e in range(pause,epoch):
    for data in Pk_dataloader:
        
        for i in range(d_iter):
            #real data
            for p in d_model.parameters():
                p.requires_grad = True
            d_model.zero_grad()
            true_data = Variable(data)
            if use_GPU:
                true_data=true_data.cuda(device)
            d_true_score = d_model(true_data)
            true_loss = -d_true_score.mean()
#             true_loss.backward()
#             d_optimizer.step()
            #fake data
#             d_model.zero_grad()
            noise = get_noise(true_data.size()[0])
            if use_GPU:
                noise = noise.cuda(device)
            fake_data = g_model(noise)
            d_fake_score = d_model(fake_data)
            fake_loss = d_fake_score.mean()
#             fake_loss.backward()
            w_loss = loss_with_penalty(d_model,true_data.data,fake_data.data)
#             w_loss.backward()
            loss =true_loss+fake_loss+w_loss
            loss.backward()
            d_optimizer.step()
        for i in range(g_iter):
            #train G
            g_model.zero_grad()
            for p in d_model.parameters():
                p.requires_grad = False
            noise = get_noise()
            if use_GPU:
                noise = noise.cuda(device)
            fake_data = g_model(noise)
            g_score = d_model(fake_data)
            g_loss = -g_score.mean()
            g_loss.backward()
            g_optimizer.step()
    dloss=float(true_loss) + float(fake_loss) + float(w_loss)
    gloss=float(g_loss)
    w_distance=-float(true_loss)-float(fake_loss)
    viz.line(Y=np.array((dloss,gloss,w_distance)).reshape(1,-1),
                X=np.array([[e,e,e]]),
                win=loss_graph,
                update='append')
    fake_imgs=fake_data.cpu().data*0.5 +0.5
    viz.image(fake_imgs[3],win=img_graph)
    if e%10==0:
        img = ToPILImage()(fake_imgs[3])
        img.save('result/G_result/iter_%d.png'%e)
        torch.save(d_model,'result/D_checkpoint/iter_d_%d.pt'%e)
        torch.save(g_model,'result/G_checkpoint/iter_g_%d.pt'%e)
