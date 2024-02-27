import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Transformer2DNet(nn.Module):#interaction layer
    def __init__(self, d_input, d_output, n_layer, n_head):
        super(Transformer2DNet, self).__init__()
        self.d_input = d_input#d_hid
        self.d_output = d_output#2
        self.n_layer = n_layer

        d_in = d_input#d_in=d_hid
        d_hidden=4*d_in

        self.row_transformer = nn.ModuleList()
        self.col_transformer = nn.ModuleList()
        self.fc = nn.ModuleList()
        for i in range(n_layer):
            d_out = d_in if i != n_layer - 1 else d_output
            #torch.nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048)#d_hidden(=64)
            self.row_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))
            self.col_transformer.append(nn.TransformerEncoderLayer(d_in, n_head, d_hidden, batch_first=True, dropout=0))    #batch_first (bool) – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
            self.fc.append(nn.Sequential(
                nn.Linear(2 * d_in, d_in),
                nn.ReLU(),
                nn.Linear(d_in, d_out)
            ))


    def forward(self, input):
        bs, n_bidder, n_item, d = input.shape
        x = input
        for i in range(self.n_layer):
            row_x = x.view(-1, n_item, d)           #(batch_size*n_bidder,n_item, d)
            row = self.row_transformer[i](row_x)
            row = row.view(bs, n_bidder, n_item, -1)

            col_x = x.permute(0, 2, 1, 3).reshape(-1, n_bidder, d)#(batch_size*n_item,n_bidder, d)
            col = self.col_transformer[i](col_x)
            col = col.view(bs, n_item, n_bidder, -1).permute(0, 2, 1, 3)



            x = torch.cat([row, col], dim=-1)

            x = self.fc[i](x)
        return x

class TransformerMechanism(nn.Module):
    def __init__(self,  n_layer, n_head, d_hidden):
        super(TransformerMechanism, self).__init__()

        self.pre_net= nn.Sequential(
            nn.Linear(5, d_hidden),#d_hid=64
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden-5)#
        )


        d_input = d_hidden
        self.n_layer, self.n_head  =  n_layer, n_head
        self.mechanism = Transformer2DNet(d_input, 3, self.n_layer, self.n_head)


    def forward(self, batch_bid,batch_value,batch_bw,batch_BW,batch_THP):
        bid,value,bw,BW,THP=batch_bid,batch_value,batch_bw,batch_BW,batch_THP
        n,m= batch_BW.shape[1],batch_BW.shape[2]

        x1 = bid.unsqueeze(-1).repeat(1,1,m)
        x2 = value.unsqueeze(-1).repeat(1,1,m)
        x3 = bw.unsqueeze(-1).repeat(1,1,m)
        x4 = BW
        x5 = THP.unsqueeze(1).repeat(1,n,1)

        y1=torch.cat([x1.unsqueeze(-1),x2.unsqueeze(-1),x3.unsqueeze(-1),x4.unsqueeze(-1),x5.unsqueeze(-1)],dim=-1)
        y=self.pre_net(y1)

        x = torch.cat([x1.unsqueeze(-1),x2.unsqueeze(-1),x3.unsqueeze(-1),x4.unsqueeze(-1),x5.unsqueeze(-1),y],dim=-1)

        mechanism = self.mechanism(x)

        allocation, payment , flow= \
            mechanism[:, :, :, 0], mechanism[:, :, :, 1],mechanism[:, :, :, 2]

        allocation=torch.sigmoid(torch.mean(allocation,dim=-1))

        payment = torch.mean(payment,dim=-1)

        flow=torch.softmax(flow,dim=-1)
        allocation_bw=allocation*bw
        flow=allocation_bw.unsqueeze(-1).repeat(1,1,m)*flow

        return allocation, payment ,flow


mechanism=TransformerMechanism(3,8,32)

# print((mechanism))
# batch_bid,batch_value,batch_bw,batch_BW,batch_THP=torch.rand(10,3),torch.rand(10,3),torch.rand(10,3),torch.rand(10,3,2),torch.rand(10,2)
#
# print(batch_bw)
# print("batch_bw.shape:",batch_bw.shape)
# allocation,payment,flow=mechanism(batch_bid,batch_value,batch_bw,batch_BW,batch_THP)
# print(allocation)
# print("allocation.shape:",allocation.shape)
#
# print(payment)
# print("payment.shape:",payment.shape)
#
# print(flow)
# print("flow.shape:",flow.shape)