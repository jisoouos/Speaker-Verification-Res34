from torch import nn

class BasicBlock(nn.Module):
    expansion=1
    
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False
        )
        self.bn1=nn.BatchNorm2d(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(
            planes,planes,kernel_size=3,stride=1,padding=1,bias=False
        )
        self.bn2=nn.BatchNorm2d(planes)

        self.shortcut=nn.Sequential() #residual 할때 차원 맞추기
        if stride !=1 or inplanes!=self.expansion*planes:
            self.shortcut= nn.Sequential(
                nn.Conv2d(inplanes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)       
        out=self.bn2(out)
        out+=self.shortcut(x)
        out=self.relu(out)
        
        return out
