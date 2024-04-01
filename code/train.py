import numpy as np
import torch, random,os
import numpy.typing as NumpyType
import torch.nn as nn
from tqdm import tqdm
from loss import computeEER
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CyclicLR


def train(model:torch.nn.Module, optimizer, loss_function, train_loader,valid_loader, epoch,model_link):
    model.train()
    lowest_eer = 999
    scheduler = CyclicLR(optimizer,base_lr=1e-8,max_lr=1e-3,step_size_up=6500,mode='triangular2',cycle_momentum=False)
    #valid_eer = valid(model, valid_loader)
    #print(valid_eer)
    for iteration in range(epoch):
        train_loss, correct, count =0,0,0
        with tqdm(iter(train_loader)) as pbar:# dataloader로 data와 label 가져옴
            for inputs, label in pbar:
                optimizer.zero_grad()
                inputs, label = inputs.cuda(), label.cuda()
                #예측 오류 계산
                pred = model(inputs) 
                loss, _ = loss_function(pred, label)
                #역전파
                
                loss.backward() 
                optimizer.step()
                scheduler.step()
                train_loss += loss.item()
        for name, param in model.named_parameters():
            count+=1
            if count==1:
                print(param)
                print(name)
        valid_eer = valid(model, valid_loader)
        
        if valid_eer < lowest_eer:
            lowest_eer = valid_eer
            torch.save(model.state_dict(), model_link+"_"+str(lowest_eer)+'.pt')

        print(f'Epoch: {iteration} | Train Loss: {train_loss/len(train_loader):.3f} ')
        print(f'Valid EER: {valid_eer:.3f}, Best EER: {lowest_eer:}')


# valid
def valid(model:torch.nn.Module, valid_loader:DataLoader):
    model.eval()

    embeddings:dict = {}
    all_scores = list()
    all_labels:list[int] = list()
    with tqdm(iter(valid_loader)) as pbar:
        for input_data, data_path in pbar:
            with torch.no_grad():
                embeddings[data_path[0]] = model.forward(input_data.cuda())
                

    cosine_similarity = nn.CosineSimilarity(dim=1,eps=1e-6)
    with open('/data/VoxCeleb1/trials.txt') as f:
        lines_of_test_dataset = f.readlines()
    for index, line in enumerate(lines_of_test_dataset):
        data = line.split()
        # Append random label if missing
        if len(data) == 2:
            data = [random.randint(0, 1)] + data

        ref_feat = embeddings[os.path.join('/data/VoxCeleb1/test', data[1])].cuda()
        com_feat = embeddings[os.path.join('/data/VoxCeleb1/test', data[2])].cuda()
        print(data[1],1,data[2],2)
        score = cosine_similarity(ref_feat,com_feat).detach().cpu().numpy()
        #print(score)
        all_scores.append(score)
        all_labels.append(int(data[0]))
        #print(len(all_scores))
    return computeEER(all_scores, all_labels)