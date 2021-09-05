# omid
# librraries ##########################################################
from flask import Flask, request, make_response, jsonify
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms
from datetime import datetime, time
import requests
import os.path
from os import path
import time as ti
import codecs
import json 
import base64
import torch
import torch.nn as nn
import torchvision
import numpy as np
import socket
from socket import *
import copy

# data ################################################################
test_data  = torchvision.datasets.MNIST(root = './', train = False, download = True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
testloader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)
# hyper parameters ####################################################
Err = []
epoch = 0
tot_epoch = 2

# client address ######################################################
url_file_1 = 'http://192.168.0.7:5001/file'
url_file_2 = 'http://192.168.0.14:5001/file'
url_file_3 = 'http://192.168.0.20:5001/file'
sett_fix = [url_file_2,url_file_1, url_file_3]
sett_fix = [url_file_2]
sett = copy.deepcopy(sett_fix)
untrained_model_name = 'untrained_model.pkl'
test = 'test.pkl'
#################################################################
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
s1 = ti.time()
model_size = os.path.getsize(test) 
os.remove(test)
print((ti.time() - s1)/60, "getting size needed time")
# functions ###########################################################
def ShouldISleep(t1, t2):
    passed = (t1 - t2).seconds
    if passed < 10:
        time.sleep(10 - passed)
        
def Learning():
    pass
    
def Sending(formdata, sett, epoch):
    import requests
    response = requests.post(sett[0], files = formdata) 
    print("sending untrain_model to", sett[0])
    if response.text == "received":
        sett.remove(sett[0])
    if sett == []:
        epoch += 1 
    return sett, epoch
# main ################################################################
# one time sleep. untrain_model_name is saved...

while True:
    
    if path.exists(untrained_model_name):
        start = datetime.now()
        try:            
            if epoch == tot_epoch:
                break
            
            
            while True:
                if os.path.getsize(untrained_model_name) >= model_size:
                    break
                             
            formdata = {'document': open(untrained_model_name, 'rb')}
            
            while len(sett) > 0:
                Learning()
                sett, epoch = Sending(formdata, sett, epoch)
           
            os.remove(untrained_model_name)        
            sett = copy.copy(sett_fix)
            
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

