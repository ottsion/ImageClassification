from torchvision import models
import torch.nn as nn
import torch as torch
from torch import optim
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from recognition import ImageUtils

model_base_path=".data/model/model_param_{}.plk"
os.environ['CUDA_VISIBLE_DEVICES']='0,1'

class VGG16():

    def __init__(self, n_inputs, n_classes, pre_model_path):
        self.model = self.vgg16_model(n_inputs, n_classes, pre_model_path)

    def printParameterCount(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} 参数总数.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} 可训练参数总数.')

    def vgg16_model(self, n_inputs, n_classes, model_path):
        # model = models.vgg16(pretrained=True)
        model = torch.load(model_path)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, n_classes),
            nn.LogSoftmax(dim=1))
        return model.cuda()
    def trainning(self, n_epochs):
        # Loss and optimizer
        weights = torch.from_numpy(np.array([0.035,0.193,0.193,0.193,0.193,0.193])).float().cuda()
        self.criteration = nn.NLLLoss(weight=weights, reduce=True, size_average=True)
        self.optimizer = optim.Adam(self.model.parameters())

        dataloader, testloader = ImageUtils.getDataSet()
        print("cuda是否有效：", torch.cuda.is_available())
        plot_data = []
        for epoch in tqdm(range(n_epochs)):
            epoch_loss = 0.0
            epoch_correct = 0.0
            times = len(dataloader['train'])
            for data, targets in tqdm(dataloader['train']):
                data = data.to("cuda", non_blocking=True)
                targets = targets.to("cuda", non_blocking=True)
                self.optimizer.zero_grad()
                # Generate predictions
                out = self.model(data)
                # Calculate loss
                loss = self.criteration(out, targets)
                # Backpropagation
                loss.backward()
                # Update model parameters
                self.optimizer.step()

                _, predicted = torch.max(out, 1)
                running_corrects = torch.sum(predicted.data == targets.data)

                epoch_loss += float(loss)
                epoch_correct += float(running_corrects)
            plot_data.append(100 * epoch_correct / (times * 128))
            print("Train Loss:{:.4f}, Train ACC:{:.4F}%".format(epoch_loss,100 * epoch_correct / (times * 128)))
        plt.plot(range(0,len(plot_data)), plot_data)
        plt.show()


def main():
    pre_model_path = model_base_path.format(60)
    n_inputs = 4096

    n_classes = 6
    vgg16 = VGG16(n_inputs, n_classes, pre_model_path)
    vgg16.printParameterCount()


    for i in range(80,161,20):
        print("start round...", i)
        vgg16 = VGG16(n_inputs, n_classes, pre_model_path)
        vgg16.trainning(i)
        torch.save(vgg16.model, model_base_path.format(i))
        pre_model_path = model_base_path.format(i)
        dataloader, testloader = ImageUtils.getDataSet()
        count = 0
        current = 0
        print('*'*300)
        for data, targets in tqdm(testloader):
            data = data.cuda()
            out = vgg16.model(data)
            # Find predictions and correct
            _, predicted = torch.max(out, 1)
            equals = torch.sum(predicted.data == targets.cuda().data)
            # Calculate accuracy
            if equals.data==1:
                current += 1
            count += 1
        print("第"+str(i)+"次遍历后：",current, count, current/count)


if __name__ == "__main__":
    main()