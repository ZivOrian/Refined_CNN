import torch
from torch import nn
from torchvision import transforms

def ret_shape(training_set)->str:
    ret_string = ""
    for idx, data in enumerate(training_set):
            datas = data[0]
            labels = data[1]
            ret_string+="{}\n".format(datas.shape)
            ret_string+="Labels:{}\n".format(labels)
            ret_string+="Labels shape:{}\n".format(len(labels))
            ret_string+="Labels[0] shape:{}\n".format(labels[0].shape)
            break
    return ret_string


def max_index(max_tnsr):
    max_tnsr = max_tnsr.tolist()
    max_num = 0
    for num in max_tnsr:
        if max_num < num:
            max_num = num
    return max_tnsr.index(max_num)

def val_func(convNet, cnn_num,valDS, dev): # Selects the best network
                                                #   out of n networks
    cnns_loss = [0]*cnn_num # represents the 10 cnns
    print("\n| Starting validation set run |\n")
    for val_sample in valDS:
        for j in range(cnn_num):
            prediction = convNet[j].forward(torch.squeeze(val_sample[0].to(dev)))
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(prediction, val_sample[1].to(dev))
            cnns_loss[j]+=loss # Adds the loss of the
    cnns_loss = [j*-1 for j in cnns_loss]
    return convNet[max_index(cnns_loss)]# picks the cnn with the lowest loss

class gcn(object): # A transformation function of 
                   #Global Contrast Normalization, to pre-process an image to have 0 contrast variance
    def __init__(self,output_size,s ,epsilon):
        assert isinstance(output_size, (int,tuple))
        self.output_size = output_size
        self.s = s
        self.epsilon = epsilon
    def __call__(self, sample):
        X = sample[0]
        X_mean = torch.mean(X)
        X_sum = torch.sum()
        return self.s*(sample)
