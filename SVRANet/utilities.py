import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import collections
import torch.optim as optim

import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


def utility(batch_data, allocation, pay):
    """ Given input valuation , payment  and allocation , computes utility
            Input params:
                valuation : [num_batches, num_agents, num_items]
                allocation: [num_batches, num_agents, num_items]
                pay       : [num_batches, num_agents]
            Output params:
                utility: [num_batches, num_agents]
    """
    return (pay-batch_data[0] * allocation)#(bs,n)





def misreportUtility(mechanism, batch_data, batchMisreports):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining all the misreported utilities

        #batchMisreports.shape torch.Size([500, 1, 2, 6])
        batchMisreports.shape torch.Size([500, 100, 2, 6])
    """

    batchTrue_bid= batch_data[0]
    batchTrue_value = batch_data[1]
    batchTrue_bw = batch_data[2]
    batchTrue_bwr = batch_data[3]
    batchTrue_thp = batch_data[4]

    nAgent = batchTrue_bwr.shape[-2]
    nObjects = batchTrue_bwr.shape[-1]
    batchSize = batchTrue_bwr.shape[0]
    nbrInitializations = batchMisreports.shape[1]


    V = batchTrue_bid.unsqueeze(1)  # (bs, 1, n_bidder)
    V = V.repeat(1, nbrInitializations, 1)  # (bs, n_init, n_bidder)
    V = V.unsqueeze(0)  # (1, bs, n_init, n_bidder)
    V = V.repeat(nAgent, 1, 1, 1)  # (n_bidder, bs, n_init, n_bidder)

    W = batchTrue_value.unsqueeze(1)# (bs, 1, n_bidder)
    W = W.repeat(1, nbrInitializations, 1) # (bs, n_init, n_bidder)
    W = W.unsqueeze(0)# (1, bs, n_init, n_bidder)
    W = W.repeat(nAgent, 1, 1, 1)# (n_bidder, bs, n_init, n_bidder)

    bw = batchTrue_bw.unsqueeze(1)# (bs, 1, n_bidder)
    bw = bw.repeat(1, nbrInitializations, 1)# (bs, n_init, n_bidder)
    bw = bw.unsqueeze(0)# (1, bs, n_init, n_bidder)
    bw = bw.repeat(nAgent, 1, 1, 1)# (n_bidder, bs, n_init, n_bidder)

    bwr = batchTrue_bwr.unsqueeze(1)# (bs, 1, n_bidder,n_item)
    bwr = bwr.repeat(1, nbrInitializations, 1,1)# (bs, n_init, n_bidder,n_item)
    bwr = bwr.unsqueeze(0)# (1, bs, n_init, n_bidder,n_item)
    bwr = bwr.repeat(nAgent, 1, 1, 1,1)# (n_bidder, bs, n_init, n_bidder,n_item)

    thp = batchTrue_thp.unsqueeze(1)# (bs, 1, n_item)
    thp = thp.repeat(1, nbrInitializations, 1)# (bs, n_init, n_item)
    thp = thp.unsqueeze(0)# (1, bs, n_init, n_item)
    thp = thp.repeat(nAgent, 1, 1, 1)# (n_bidder, bs, n_init, n_item)

    M = batchMisreports.unsqueeze(0)  #(1,bs, n_init, n_bidder)
    M = M.repeat(nAgent, 1, 1, 1)  # (n_bidder, bs, n_init, n_bidder)

    mask1 = np.zeros((nAgent, nAgent))
    mask1[np.arange(nAgent), np.arange(nAgent)] = 1.0
    mask2 = np.ones((nAgent, nAgent))
    mask2 = mask2 - mask1

    mask1 = (torch.tensor(mask1).float()).to(device)
    mask2 = (torch.tensor(mask2).float()).to(device)

    V   = V.permute(1, 2, 0, 3)  # (bs, n_init, n_bidder, n_bidder)
    W   = W.permute(1, 2, 0, 3)
    bw  = bw.permute(1, 2, 0, 3) # (bs, n_init, n_bidder, n_bidder)
    bwr = bwr.permute(1, 2, 0, 3,4)# (bs, n_init, n_bidder, n_bidder ,n_item)
    thp = thp.permute(1, 2, 0, 3)# (bs, n_init, n_bidder, n_item)
    M = M.permute(1, 2, 0, 3)  # (bs, n_init, n_bidder, n_bidder)



    tensor = M * mask1 + V * mask2

    tensor = tensor.permute(2, 0, 1, 3)  # (n_bidder, bs, n_init, n_bidder)
    W   = W.permute(2, 0, 1, 3)# (n_bidder, bs, n_init, n_bidder)
    bw  = bw.permute(2, 0, 1, 3)# (n_bidder, bs, n_init, n_bidder)
    bwr = bwr.permute(2, 0, 1, 3,4)# (n_bidder, bs, n_init, n_bidder, n_item)
    thp = thp.permute(2, 0, 1, 3)# (n_bidder, bs, n_init, n_item)


    tensor = View(-1, nAgent)(tensor)  # (n_bidder * bs * n_init, n_bidder)
    W   = View(-1, nAgent)(W) # (n_bidder * bs * n_init, n_bidder)
    bw  = View(-1, nAgent)(bw)# (n_bidder * bs * n_init, n_bidder)
    bwr = View(-1, nAgent, nObjects)(bwr)# (n_bidder * bs * n_init, n_bidder,n_item)
    thp = View(-1, nObjects)(thp)# (n_bidder * bs * n_init, item)
    tensor = tensor.float()

    allocation, payment ,flow= mechanism(tensor, W,bw,bwr,thp)  # (n_bidder * bs * n_init, n_bidder)\(n_bidder * bs * n_init, n_bidder,n_item)

    V = V.permute(2, 0, 1, 3)  # (n_bidder, bs, n_init, n_bidder)
    # M = M.permute(2, 0, 1, 3)  # (n_bidder, bs, n_init, n_bidder)


    allocation = View(nAgent, batchSize, nbrInitializations, nAgent)(allocation)
    payment = View(nAgent, batchSize, nbrInitializations, nAgent)(payment)
    flow = View(nAgent, batchSize, nbrInitializations, nAgent, nObjects)(flow)

    advUtilities = payment- allocation * V

    advUtility = advUtilities[np.arange(nAgent), :, :, np.arange(nAgent)]

    return (advUtility.permute(1, 2, 0))




def misreportOptimization(mechanism, batch, data, misreports, R, gamma):

    localMisreports = misreports[:]

    batchMisreports = torch.tensor(misreports[batch]).to(device)

    batchTrue_bid = torch.tensor(data[0][batch]).to(device)
    batchTrue_value = torch.tensor(data[1][batch]).to(device)
    batchTrue_bw = torch.tensor(data[2][batch]).to(device)
    batchTrue_bwr = torch.tensor(data[3][batch]).to(device)
    batchTrue_thp = torch.tensor(data[4][batch]).to(device)

    batch_data = (batchTrue_bid, batchTrue_value,batchTrue_bw,batchTrue_bwr,batchTrue_thp)

    batchMisreports.requires_grad = True

    opt = optim.Adam([batchMisreports], lr=gamma)

    for k in range(R):
        advU = misreportUtility(mechanism, batch_data, batchMisreports)
        loss = -1 * torch.sum(advU).to(device)


        opt.zero_grad()
        loss.backward()
        opt.step()

    mechanism.zero_grad()
    #    (bs,init,n)                                   (bs,init,n)
    localMisreports[batch, :, :] = batchMisreports.cpu().detach().numpy()
    return (localMisreports)


def trueUtility(batch_data, allocation=None, payment=None):
    """ This function takes the valuation batches
        and returns a tensor constaining the utilities
    """
    return utility(batch_data, allocation, payment)


def regret(mechanism, batch_data, batchMisreports, allocation, payment):
    """ This function takes the valuation and misreport batches
        and returns a tensor constaining the regrets for each bidder and each batch


    """
    missReportUtilityAll = misreportUtility(mechanism, batch_data, batchMisreports)
    misReportUtilityMax = torch.max(missReportUtilityAll, dim=1)[0]#(bs,n)
    return (misReportUtilityMax - trueUtility(batch_data, allocation, payment))



def loss(mechanism, lamb_rgt, rho_rgt, lamb_ir, rho_ir, lamb_bf, rho_bf, lamb_bw, rho_bw, lamb_thp, rho_thp, batch,
     data, misreports, budget):
    """
    This function tackes a batch which is a numpy array of indices and computes
    the loss function                                                             : loss
    the average regret per agent which is a tensor of size [nAgent]               : rMean
    the maximum regret among all batches and agenrs which is a tensor of size [1] : rMax
    the average payments which is a tensor of size [1]                            : -paymentLoss

    """
    batchMisreports = torch.tensor(misreports[batch]).to(device)

    batchTrue_bid   = torch.tensor(data[0][batch]).to(device)
    batchTrue_value = torch.tensor(data[1][batch]).to(device)
    batchTrue_bw    = torch.tensor(data[2][batch]).to(device)
    batchTrue_bwr   = torch.tensor(data[3][batch]).to(device)
    batchTrue_thp   = torch.tensor(data[4][batch]).to(device)

    allocation, payment ,flow= mechanism(batchTrue_bid,batchTrue_value,batchTrue_bw,batchTrue_bwr,batchTrue_thp)
    paymentLoss = (-torch.sum(allocation*batchTrue_value)+torch.sum(payment)) / batch.shape[0]#

    batch_data = (batchTrue_bid, batchTrue_value,batchTrue_bw,batchTrue_bwr,batchTrue_thp)
    r = F.relu(regret(mechanism, batch_data, batchMisreports, allocation, payment))#(bs,n)

    # rgt
    rMean = torch.mean(r, dim=0).to(device)
    rMax = torch.max(r).to(device)
    lagrangianLoss_r = torch.sum(rMean * lamb_rgt)
    lagLoss_r = (rho_rgt / 2) * torch.sum(torch.pow(rMean, 2))

    # Ir
    I=F.relu(allocation*batchTrue_bid-payment)# (bs,n)
    I_Mean=torch.mean(I, dim=0).to(device)#(n,)
    I_Max=torch.max(I).to(device) #(1,)
    lagrangianLoss_I= torch.sum(I_Mean*lamb_ir)
    lagLoss_I = (rho_ir / 2) * torch.sum(torch.pow(I_Mean, 2))

    # Bf               (bs,n)
    B=F.relu(torch.sum(payment,dim=1)-budget)#(bs,)
    B_Mean = torch.mean(B).to(device)
    B_Max=torch.max(B).to(device)
    lagrangianLoss_B= torch.sum(B_Mean*lamb_bf).to(device)
    lagLoss_B= (rho_bf / 2) * torch.sum(torch.pow(B_Mean, 2)).to(device)

    #bw       (bs,n,m)
    # print(batchTrue_bid.shape,batchTrue_bid.shape,batchTrue_bw.shape,batchTrue_bwr.shape,batchTrue_thp.shape)
    # print(flow.shape)
    BW=F.relu(flow-batchTrue_bwr)
    BW_Mean=torch.mean(BW,dim=0).to(device)#(n,m)
    BW_Max=torch.max(BW_Mean).to(device)
    # print(BW_Mean.shape,lamb_bw.shape)
    lagrangianLoss_BW=torch.sum(BW_Mean*lamb_bw)
    lagLoss_BW=(rho_bw / 2) * torch.sum(torch.pow(BW_Mean, 2))

    #thp                (bs,n,m)
    THP=F.relu(torch.sum(flow,dim=1)-batchTrue_thp)#(bs,m)
    THP_Mean=torch.mean(THP,dim=0).to(device)#(m,)
    THP_Max=torch.max(THP).to(device)
    lagrangianLoss_THP = torch.sum(THP_Mean * lamb_thp)
    lagLoss_THP = (rho_thp / 2) * torch.sum(torch.pow(THP_Mean, 2))

    loss = paymentLoss + lagrangianLoss_r + lagLoss_r + lagrangianLoss_I + lagLoss_I + lagrangianLoss_B + lagLoss_B +lagrangianLoss_BW+lagLoss_BW+lagrangianLoss_THP+lagLoss_THP
    return (loss, rMean, rMax,I_Mean,I_Max,B_Mean,B_Max,BW_Mean,BW_Max,THP_Mean,THP_Max, -paymentLoss,torch.sum(payment)/batch.shape[0])





