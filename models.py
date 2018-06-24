import torch
from torch.autograd import Variable,grad
import torch.nn as nn
import torch.nn.functional as F

units=[32,64,128,256,512,1024]
padding=[2,2,2,2,2]
k_size=[5,5,5,5,5]#kernal size
strides=[2,2,2,2,2]
fs=[8,8]
class G_net(nn.Module):
    def __init__(self,in_dim=latent_dim):
        super(G_net,self).__init__()
        self.fc1= nn.Sequential(
            nn.Linear(in_dim,units[4]),
            nn.ReLU(),
            nn.Linear(units[4],units[3]*fs[0]*fs[1]),
            nn.ReLU(),
            nn.BatchNorm1d(units[3]*fs[0]*fs[1])
        )
        self.ct1 = nn.Sequential(
            nn.ConvTranspose2d(units[3],units[2],k_size[3],stride=strides[3],padding=padding[3],output_padding=strides[3]/2),
            nn.BatchNorm2d(units[2]),
            nn.ReLU()
        )#[64,12,12]
        self.ct2 = nn.Sequential(
            nn.ConvTranspose2d(units[2],units[1],k_size[2],stride=strides[2],padding=padding[2],output_padding=strides[2]/2),
            nn.BatchNorm2d(units[1]),
            nn.ReLU()
        )#[32,27,27]
        self.ct3 = nn.Sequential(
            nn.ConvTranspose2d(units[1],units[0],k_size[1],stride=strides[1],padding=padding[1],output_padding=strides[1]/2),
            nn.BatchNorm2d(units[0]),
            nn.ReLU()
        )#[3,57,57]
        self.ct4 = nn.Sequential(
            nn.ConvTranspose2d(units[0],small_image_size[0],k_size[0],stride=strides[0],padding=padding[0],output_padding=strides[0]/2),
            nn.Tanh()
        )#[3,289,289]
    def forward(self,X):
        X = self.fc1(X)
        X = self.ct1(X.view(-1,units[3],fs[0],fs[1]))
        X = self.ct2(X)
        X = self.ct3(X)
        return self.ct4(X)
class D_net(nn.Module):
    def __init__(self):
        super(D_net,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(image_size[0],units[0],k_size[0],strides[0],padding=padding[0]),
            nn.BatchNorm2d(units[0]),
            nn.LeakyReLU(0.2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(units[0],units[1],k_size[1],strides[1],padding=padding[1]),
            nn.BatchNorm2d(units[1]),
            nn.LeakyReLU(0.2)
        )
        self.conv3=nn.Sequential(
            nn.Conv2d(units[1],units[2],k_size[2],strides[2],padding=padding[2]),
            nn.BatchNorm2d(units[2]),
            nn.LeakyReLU(0.2)
        )
        self.conv4=nn.Sequential(
            nn.Conv2d(units[2],units[3],k_size[3],strides[3],padding=padding[3]),
            nn.BatchNorm2d(units[3]),
            nn.LeakyReLU(0.2)
        )
        self.fc1 = nn.Linear(units[3]*fs[0]*fs[1],units[4])
        self.dp = nn.Dropout(0.5)
        self.d_out = nn.Linear(units[4],1)
    def forward(self,X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = self.conv4(X)
        X = X.view((-1,units[3]*fs[0]*fs[1]))
        X = self.dp(F.leaky_relu(self.fc1(X)))
        out = self.d_out(X)
        return out