# libraries #####################################################
from flask import Flask, request, make_response, jsonify
from werkzeug.utils import secure_filename
import codecs
import json 
import base64
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import socket
from socket import *
import time
import copy

# hyper parameters ##############################################
n_workers = 1
untrained_model_name = 'untrained_model.pkl'
trained_model_name = 'trained_model.pkl'
net_named_parameters = []
Weights_of_all_workers = []
Err = []
ones = 1
tot_epoch = 2
epoch = 0
# data ################################################################
test_data  = torchvision.datasets.MNIST(root = './', train = False, download = True, transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))
testloader = torch.utils.data.DataLoader(test_data, batch_size=1000, shuffle=False, num_workers=0)
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
dict_params = dict(net.named_parameters())
Fix_net_state_dict = copy.deepcopy(net.state_dict())
torch.save(net.state_dict(), untrained_model_name)
# Functions #####################################################
def Get_models(request, net, net_named_parameters):
    cpu_device = torch.device('cpu')
    f = request.files['document'] 
    filename = secure_filename(f.filename)
    file_path = filename
    f.save(file_path) 
    tempModel = net
    tempModel.load_state_dict(torch.load(file_path, map_location = cpu_device))
    my_parameters = list(iter(tempModel.named_parameters()))
    my_parameters2 = copy.deepcopy(my_parameters)
    net_named_parameters.append(my_parameters2)
    params = Corruption_in_advance(net_named_parameters[-1])
    Weights_of_all_workers.append(params[None,:])
    return net_named_parameters, Weights_of_all_workers
    
def Corruption_in_advance(net_g):
    import torch
    import random
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = []
    for name, param in net_g:
        params.append(param.data.view(-1).to(device))
    params = torch.cat(params)   
    params = params.unsqueeze(0)
    return params
        
def Detect_adv(Weights_of_all_workers):
    clusters = np.ones(len(Weights_of_all_workers))
    return clusters

def CalcErr(Fix_net_state_dict, net, testloader, Err):
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    correct = 0
    total = 0
    net.load_state_dict(Fix_net_state_dict)
    with torch.no_grad():
        for data in testloader:
            images_test, labels_test = data[0].to(device),  data[1].to(device)
            outputs = net(images_test)
            _, predicted = torch.max(outputs.data, 1)
            total += labels_test.size(0)
            correct += (predicted == labels_test).sum().item()
            break
        Err.append(1 - (correct / total))
    return Err


def Merge_models(net_named_parameters, Weights_of_all_workers, n_workers,dict_params, Fix_net_state_dict, testloader, Err, net, epoch):
    if len(net_named_parameters) == n_workers:
        epoch += 1
        clusters = Detect_adv(Weights_of_all_workers)
        ones = 0
        Number_of_admitted_workers = len(np.nonzero(clusters))
        for i, j in enumerate(net_named_parameters, 0):
            if clusters[i] == 1:
                ones += 1
                for name, param in j:
                    if ones == 1:
                        Fix_net_state_dict[name].data = dict_params[name].data * 0
                    part1 = param.data/Number_of_admitted_workers
                    part2 = Fix_net_state_dict[name].data
                    Fix_net_state_dict[name].data = part1  + part2
        Err = CalcErr(Fix_net_state_dict, net, testloader, Err)             
        torch.save(Fix_net_state_dict, untrained_model_name)        
        net_named_parameters = []
        Weights_of_all_workers = []
    else:
        pass  
    return net_named_parameters, Weights_of_all_workers, Err, epoch

def Figure(Err):
    import matplotlib.pyplot as plt
    plt.plot(Err, '-o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy on 1000 Testdata')
    plt.title('Err_Epoch_Jetson')
    plt.show()
    
# Flask part #####################################################
app = Flask(__name__)
@app.route('/file', methods = ['GET', 'POST'])
def textFile():
    global net_named_parameters
    global Weights_of_all_workers
    global ones
    global Err
    global epoch
    if ones == 1:
        Err = CalcErr(Fix_net_state_dict, net, testloader, Err)
        ones = 0
    if request.method == 'POST':         
        net_named_parameters, Weights_of_all_workers = Get_models(request, net, net_named_parameters)
        net_named_parameters, Weights_of_all_workers, Err, epoch = Merge_models(net_named_parameters, Weights_of_all_workers, n_workers, dict_params, Fix_net_state_dict, testloader, Err, net, epoch)
        
        if epoch == tot_epoch:
            print("Fig of errors ...")
            epoch = 0
            Figure(Err)
            
            

        return "received"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, threaded=True) 
    
    
    
    
    
    
    
