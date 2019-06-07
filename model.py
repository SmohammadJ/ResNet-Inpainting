import torch
import torch.nn as nn
import torch.nn.functional as F

class resBlock(nn.Module): 
    def __init__(self, inchannel, outchannel, downsample=False, upsample=False):
        super(resBlock, self).__init__()

        self.inchannel = inchannel
        self.outchannel = outchannel
        self.downsample = downsample
        self.upsample = upsample

        if(self.downsample == False and self.upsample == False):
            self.standard = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(inchannel,outchannel,3,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(outchannel,outchannel,3,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(outchannel))

        if self.downsample :
            self.conv0_D = nn.Conv2d(inchannel,outchannel,1,stride=2,padding=0,bias=False)
            self.Dsample = nn.Sequential(nn.ReflectionPad2d(1),
                nn.Conv2d(inchannel,outchannel,3,stride=2,padding=0,bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(outchannel,outchannel,3,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(outchannel))

        if self.upsample :
            self.conv0_U = nn.Conv2d(inchannel,outchannel,1,stride=1,padding=0,bias=False)
            self.Usample = nn.Sequential(nn.ReflectionPad2d(1),
                nn.ConvTranspose2d(inchannel,outchannel,3,stride=2,padding=3,output_padding=1,bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(),
                nn.ReflectionPad2d(1),
                nn.Conv2d(outchannel,outchannel,3,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(outchannel))


    def forward(self, x):

        if self.downsample:
            out = self.Dsample(x)
            out = out + self.conv0_D(x)

        if self.upsample:
            out = self.Usample(x)
            out1 = self.conv0_U(x)
            out = out + F.interpolate(out1,scale_factor=2)

        if(self.downsample == False and self.upsample == False):
            out = self.standard(x)
            out = out + x

        return F.relu(out)



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.reflectpad1 = nn.ReflectionPad2d(3)
        self.conv1 = nn.Conv2d(4,32,7,stride=1,padding=0,bias=False)
        self.reflectpad2 = nn.ReflectionPad2d(2)
        self.conv2 = nn.Conv2d(32,64,5,stride=2,padding=0,bias=False)

        self.downsample_ = resBlock(64,128,downsample=True,upsample=False)

        self.standard = nn.ModuleList([resBlock(128,128,downsample=False,upsample=False ) for i in range(6)])

        self.upsample_ = resBlock(128,64,downsample=False,upsample=True)

        self.reflectpad3 = nn.ReflectionPad2d(1)
        self.deconve1 = nn.ConvTranspose2d(64,32,5,stride=2,padding=4,output_padding=1,bias=False)
        self.reflectpad4 = nn.ReflectionPad2d(3)
        self.conv3 = nn.Conv2d(32,3,7,stride=1,padding=0,bias=True)
        self.sigmpid = nn.Sigmoid()


    def forward(self, x):

        out = self.reflectpad1(x)
        out = F.relu(self.conv1(out))
        out = self.reflectpad2(out)
        out = F.relu(self.conv2(out))
        
        out = self.downsample_(out)

        for i in range(6):
            out = self.standard[i](out)

        out = self.upsample_(out)

        out = self.reflectpad3(out)
        out = F.relu(self.deconve1(out))
        out = self.reflectpad4(out)
        out = self.conv3(out)
        out = self.sigmpid(out)

        return out





            






        


        
