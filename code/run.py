import torch
import torch.optim as optim
from loss import AAMsoftmax
from Data_loader import train_loader,test_loader

from torch.utils.data import DataLoader
from train import train
from Resnet  import ResNet18


model = ResNet18().cuda()
#model.load_state_dict(torch.load('./best_model_MFCC_16.882290562036058.pt'))
lossfunc = AAMsoftmax(n_class = 1211, m=0.2, s=30).cuda()

optimizer = optim.AdamW([
    {'params': model.parameters()},
    {'params': lossfunc.parameters(), 'weight_decay': 2e-4}
], lr=1e-3, weight_decay=2e-5)
TrainingSet = train_loader("/data/VoxCeleb1/train_list.txt", "/data/VoxCeleb1/train", 200)
TrainDatasetLoader = DataLoader(TrainingSet, batch_size = 128, shuffle = True, num_workers = 10, drop_last = True)
ValidSet = test_loader('/data/VoxCeleb1/trials.txt','/data/VoxCeleb1/test', 200,10)
ValidDatasetLoader = DataLoader(ValidSet, batch_size = 1, shuffle = False, num_workers = 10, drop_last = True)

num_epochs = 48
train(model, optimizer, lossfunc ,TrainDatasetLoader,ValidDatasetLoader, num_epochs,'/eer')