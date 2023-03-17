# Taken straight from https://github.com/jwzhanggy/GResNet
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class MethodGCN(nn.Module):
    def __init__(self, input_features, hidden_dim, output_features, dropout):
        nn.Module.__init__(self)

        self.gc1 = GraphConvolution(input_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_features)
        self.dropout = dropout
    def forward(self, raw_x, adj, eigen_adj=None):
        x = F.relu(self.gc1(raw_x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        pred_y = F.log_softmax(x, dim=1)
        return pred_y

class MethodDeepGCNResNet(nn.Module):
    def __init__(self, input_features, hidden_dim, output_features, dropout, depth):
        nn.Module.__init__(self)
        self.depth = depth
        self.gc_list = [None] * self.depth
        self.residual_weight_list = [None] * self.depth
        if self.depth == 1:
            self.gc_list[self.depth-1] = GraphConvolution(input_features, output_features)
            self.residual_weight_list[self.depth-1] = Parameter(torch.FloatTensor(input_features, output_features))
        else:
            for i in range(self.depth):
                if i == 0:
                    self.gc_list[i] = GraphConvolution(input_features, hidden_dim)
                    self.residual_weight_list[i] = Parameter(torch.FloatTensor(input_features, hidden_dim))
                elif i == self.depth - 1:
                    self.gc_list[i] = GraphConvolution(hidden_dim, output_features)
                    self.residual_weight_list[i] = Parameter(torch.FloatTensor(hidden_dim, output_features))
                else:
                    self.gc_list[i] = GraphConvolution(hidden_dim, hidden_dim)
                    self.residual_weight_list[i] = Parameter(torch.FloatTensor(hidden_dim, hidden_dim))
        for i in range(self.depth):
            stdv = 1. / math.sqrt(self.residual_weight_list[i].size(1))
            self.residual_weight_list[i].data.uniform_(-stdv, stdv)
        self.dropout = dropout

    def myparameters(self):
        parameter_list = list(self.parameters())
        for i in range(self.depth):
            parameter_list += self.gc_list[i].parameters()
        parameter_list += self.residual_weight_list
        return parameter_list

    # ---- non residual ----
    def forward(self, raw_x, adj, eigen_adj=None):
        x = raw_x
        for i in range(self.depth - 1):
            x = F.relu(self.gc_list[i](x, adj) + torch.mm(raw_x, self.residual_weight_list[0]))
            x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            y = self.gc_list[self.depth - 1](x, adj) + torch.mm(raw_x, self.residual_weight_list[0])
        else:
            y = self.gc_list[self.depth - 1](x, adj) + torch.mm(torch.mm(raw_x, self.residual_weight_list[0]),
                                                                self.residual_weight_list[self.depth - 1])
        pred_y = F.log_softmax(y, dim=1)
        return pred_y

class MethodDeepGATResNet(nn.Module):

    def __init__(self, input_features, hidden_dim, output_features, dropout, alpha, nheads, depth):
        nn.Module.__init__(self)

        self.dropout = dropout
        self.depth = depth
        self.gat_list = [None] * self.depth
        self.residual_weight_list = [None] * self.depth

        if self.depth == 1:
            self.gat_list = []
            self.out_att = GraphAttentionLayer(input_features, output_features, dropout=dropout, alpha=alpha, concat=False)
            self.residual_weight_list[self.depth-1] = Parameter(torch.FloatTensor(input_features, output_features))
        else:
            for depth_index in range(self.depth - 1):
                if depth_index == 0:
                    self.gat_list[depth_index] = [GraphAttentionLayer(input_features, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.residual_weight_list[depth_index] = Parameter(torch.FloatTensor(input_features, hidden_dim * nheads))
                else:
                    self.gat_list[depth_index] = [GraphAttentionLayer(hidden_dim * nheads, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
                    self.residual_weight_list[depth_index] = Parameter(torch.FloatTensor(hidden_dim * nheads, hidden_dim * nheads))
                for i, attention in enumerate(self.gat_list[depth_index]):
                    self.add_module('attention_{}_{}'.format(depth_index, i), attention)
            self.out_att = GraphAttentionLayer(hidden_dim * nheads, output_features, dropout=dropout, alpha=alpha, concat=False)
            self.residual_weight_list[self.depth-1] = Parameter(torch.FloatTensor(hidden_dim * nheads, output_features))
        for i in range(self.depth):
            stdv = 1. / math.sqrt(self.residual_weight_list[i].size(1))
            self.residual_weight_list[i].data.uniform_(-stdv, stdv)

    def myparameters(self):
        parameter_list = list(self.parameters())
        for i in range(self.depth - 1):
            for gat in self.gat_list[i]:
                parameter_list += gat.parameters()
        parameter_list += self.out_att.parameters()
        parameter_list += self.residual_weight_list
        return parameter_list
    def forward(self, raw_x, adj):
        x = raw_x
        for i in range(self.depth - 1):
            x = F.dropout(x, self.dropout, training=self.training)
            x = torch.cat([att(x, adj) for att in self.gat_list[i]], dim=1) + torch.mm(raw_x,
                                                                                       self.residual_weight_list[0])
        x = F.dropout(x, self.dropout, training=self.training)
        if self.depth == 1:
            x = F.elu(self.out_att(x, adj)) + torch.mm(raw_x, self.residual_weight_list[self.depth - 1])
        else:
            x = F.elu(self.out_att(x, adj)) + torch.mm(torch.mm(raw_x, self.residual_weight_list[0]),
                                                       self.residual_weight_list[self.depth - 1])
        return F.log_softmax(x, dim=1)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime