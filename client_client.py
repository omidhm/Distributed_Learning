##############################################################
# libraries ##################################################
import requests
import os.path
from os import path
import time as ti
from datetime import datetime, time
import io
import PIL.Image as Image
import base64
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import socket
from socket import *
import logging

# hyper params ###############################################
ones = 1
num_epochs = 10
num_classes = 10
batch_size = 1000
set_size = 20000; # the number of trained data sets
testSize = 1000 # the number of test data sets    
learning_rate = 0.05

# data #######################################################
from torchvision.datasets import MNIST
trainset = MNIST(root = './', train = True, download = True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
set_size = len(trainset)//3
set1, set2, set3 = torch.utils.data.random_split(trainset, [set_size,set_size,(len(trainset)-set_size*2)])

train_loader = torch.utils.data.DataLoader(set2, batch_size=batch_size,shuffle=True)
testset = MNIST(root = './', train = False, download = True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))

# Set logger ################################################## 
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(filename='simpleNet.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

# Server ##################################################### 
url_file = 'http://192.168.0.9:5002/file'
untrained_model_name = 'untrained_model.pkl'
trained_model_name = 'trained_model_2.pkl'
test = 'test.pkl'

# model ######################################################
class model(nn.Module):
    def __init__(self):
        hidden_1 = 32
        hidden_2 = 32
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(1,  16,  kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32*10*10, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 10)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv2(x), 2))
        x = x.view(-1,32*10*10 )
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = model()
torch.save(net.state_dict(), test)
model_size = os.path.getsize(test)
os.remove(test)

# functions ##################################################
def ShouldISleep(t1, t2):
    passed = (t1-t2).seconds
    if passed < 10:
        time.sleep(10 - passed)

def train_model(model, untrained_model_name, learning_rate, train_loader):
    import torch
    cpu_device = torch.device('cpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    while True:
        if os.path.getsize(untrained_model_name) == model_size:
            print("check size")
        if os.path.getsize(untrained_model_name) >= model_size - 500:
            break
    net = model()
    net.load_state_dict(torch.load(untrained_model_name, map_location=cpu_device))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate)
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) 
        optimizer.zero_grad()
        # forward + backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        print(loss)
        logging.info(loss) 
        loss.backward() 
        optimizer.step()
        
#        print((time.time() - t_mb)/60)
    torch.save(net.state_dict(), trained_model_name) 

# main #######################################################
while True:
    if path.exists(untrained_model_name):
        start = datetime.now()
        try:
            
            if ones == 1:
                train_model(model, untrained_model_name, learning_rate, train_loader)
                ones = 0
            print("start sending to master...")
            formdata = {'document': open(trained_model_name, 'rb')}
            response = requests.post(url_file, files = formdata) 
            print("response")
            if response.text == "received":
                os.remove(untrained_model_name)
                os.remove(trained_model_name)
                ones = 1
        except requests.exceptions.ConnectionError as e:
            end = datetime.combine(start.date(), time(0))
            ShouldISleep(start, end)
            posting = False
            continue
        except requests.exceptions.ReadTimeout as e:
            end = datetime.combine(start.date(), time(0))
            ShouldISleep(start, end)
            posting = False
            continue
    else:
        pass
##############################################################
