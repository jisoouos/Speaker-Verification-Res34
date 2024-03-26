from torch import nn
import torch 


class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1000):
        super().__init__()

        self.inplanes =64
        self.stem= nn.Sequential(
            nn.Conv2d(3,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        )
        self.stage1=self._make_layer(block,64,layers[0],stride=1)
        self.stage2=self._make_layer(block,128,layers[1],stride=2)
        self.stage3=self._make_layer(block,256,layers[2],stride=2)
        self.stage4=self._make_layer(block,512,layers[3],stride=2)

        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes)

    def _make_layer(self,block,planes,num_blocks,stride):
        layers=[]
        layers.append(block(self.inplanes,planes,stride))
        self.inplanes=planes*block.expansion
        for _ in range(num_blocks-1):
            layers.append(block(self.inplanes,planes,1))

        return nn.Sequential(*layers)
    
    def forward(self,x):
        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.avgpool(out)
        out = torch.flatten(out,1)
        out=self.fc(out)
        return out
        