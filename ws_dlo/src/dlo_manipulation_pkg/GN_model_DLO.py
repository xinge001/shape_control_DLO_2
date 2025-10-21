import copy
import numpy as np
import torch
import tqdm
from sklearn.model_selection import train_test_split

import torch.nn as nn
# import cv2
# import matplotlib.pyplot as plt
import torch.optim as optim
import networkx as nx
import torch_geometric as pyg

import my_functions as mf
import torch_scatter

from torch.nn.functional import mse_loss



loss_fn = mse_loss


DEVICE = torch.device(f'cuda:{0}')

class MySubEdge(pyg.nn.MessagePassing):
    def __init__(self, layer_type, input_size, hidden_size, sudo_input_size, output_size, args, edge_layer = None, a=0):
        super().__init__()
        # self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        # self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

        # always pass through a relu layer
        # self.lin_edge = MLP_encoder(350, hidden_size, hidden_size, layers)
        # self.lin_node = MLP_encoder(200, hidden_size, hidden_size, layers)

        self.type = layer_type #'ee' or 'pe' encoder or propagation

        if edge_layer:
            self.edge_layer = edge_layer
        else:
            self.edge_layer = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.sudo_node_layer = torch.nn.Linear(sudo_input_size, output_size, bias=False)
        self.leaky_relu = torch.nn.functional.leaky_relu
        self.aggregator = pyg.nn.aggr.SumAggregation()
        self.a = a

        # assume train all layers
        self.alpha0=1
        self.alpha1=1
        self.alpha2=1


        self.lr = args.lr

    def forward(self, particle_effect, node_feature, edge_index, edge_feature):
        edge_all, aggr = self.propagate(edge_index, x=(particle_effect, particle_effect), edge_feature=edge_feature)
        edge_in, edge_out, edge_out_dot = edge_all
        # edge_out = self.propagate(edge_index, x=(particle_effect, particle_effect), edge_feature=edge_feature)
        # print("edge output shape", edge_out.shape)
        # print("aggr edge shape", aggr.shape)
        # print("aggr edge shape", aggr[0,:])
        if self.type == "ee":
            node_out = self.sudo_node_layer(aggr)
        elif self.type == 'pe':
            node_out = self.sudo_node_layer(torch.cat((node_feature, aggr), dim=-1))


        #delete memory for Adam memory problem
        del edge_in
        torch.cuda.empty_cache()
        del edge_out_dot
        torch.cuda.empty_cache()

        if self.type=='ee':
            return node_out, edge_out
        elif self.type=='pe':
            return node_out, edge_out, torch.cat((node_feature, aggr), dim=-1)


    def message(self, x_i, x_j, edge_feature):
        # x = torch.cat((x_i, x_j, edge_feature), dim=-1)
        # propnet has a different order

        if self.type == 'ee':
            x = edge_feature
        elif self.type == 'pe':
            x = torch.cat((edge_feature, x_j, x_i), dim=-1)

        x1 = x.detach().clone()
        # x1 = x.detach()
        x = self.edge_layer(x)
        # print("node before relu", x)
        x_dot = mf.derivative_fun(self.leaky_relu)(x,self.a)
        # print("node derivative", x_dot)
        x = self.leaky_relu(x,self.a)
        # print("message output", x)
        return (x1, x, x_dot)

    def aggregate(self, inputs, index, dim_size=None):
        # print(index)
        edge_in, edge_out, edge_out_dot = inputs
        out = torch_scatter.scatter(edge_out, index, dim=self.node_dim, dim_size=dim_size, reduce="sum")
        # print("aggregate output", out)
        return (inputs, out)

    def train(self, particle_effect, node_feature, edge_index, edge_feature, y):
        lr = self.lr

        edge_all, aggr = self.propagate(edge_index, x=(particle_effect, particle_effect), edge_feature=edge_feature)
        edge_in, edge_out, edge_out_dot = edge_all
        # edge_out = self.propagate(edge_index, x=(particle_effect, particle_effect), edge_feature=edge_feature)
        # print("edge output shape", edge_out.shape)
        # print("aggr edge shape", aggr.shape)
        # print("aggr edge shape", aggr[0,:])
        if self.type == "ee":
            node_out = self.sudo_node_layer(aggr)
        elif self.type == 'pe':
            node_out = self.sudo_node_layer(torch.cat((node_feature, aggr), dim=-1))
        # print("node feature", node_out)
        # edge_out = edge_feature + edge_out
        # node_out = x + node_out
        phi = aggr.detach()
        x2 = node_feature.detach()
        x1 = edge_in.detach()
        edge_out_dot = edge_out_dot.detach()
        # print("node output", node_out)
        # return node_out, edge_out

        e = y-node_out

        w1 = self.edge_layer.weight.data
        w2 = self.sudo_node_layer.weight.data
        w21 = self.sudo_node_layer.weight.data[:,:w2.shape[1]-w1.shape[0]]
        w22 = self.sudo_node_layer.weight.data[:,w2.shape[1]-w1.shape[0]:]

        # check convergence
        def check_convergence(lr):
            e_row, e_col = e.shape
            e_dim = e_row * e_col

            condition_sum1_mx = torch.zeros((e_dim, e_dim), dtype=torch.float).to(DEVICE)
            for i in range(w1.shape[0]):
                CRX = self.aggregator(edge_out_dot[:, i].view(edge_index.shape[1], 1) * x1, edge_index[1])
                first_half = torch.cat(tuple(w22[:, i:i+1][:,None]*CRX),dim=0)
                condition_sum1_mx += first_half @ (first_half.t())

            # second part
            blocks1 = [phi @ phi.T] * w22.shape[0]  # Repeating a few times in a list
            blocks2 = [x2 @ x2.T] * w21.shape[0]  # Repeating a few times in a list
            # Use torch.block_diag to create the block diagonal matrix
            # condition_sum2_mx = torch.block_diag(*blocks)
            condition_sum2_mx = torch.block_diag(*blocks1) + torch.block_diag(*blocks2)

            # condition_sum2_mx = -(self.alpha0*lr * lr * condition_sum1_mx + self.alpha1*lr * lr * condition_sum2_mx)
            condition_sum2_mx = -(lr * lr * condition_sum1_mx + lr * lr * condition_sum2_mx)

            torch.diagonal(condition_sum2_mx).copy_((2) * lr + torch.diagonal(condition_sum2_mx))
            condition_sum1_mx = torch.empty(0)
            del condition_sum1_mx
            torch.cuda.empty_cache()

            L,info = torch.linalg.cholesky_ex(condition_sum2_mx)
            condition_sum2_mx = torch.empty(0)
            del condition_sum2_mx
            torch.cuda.empty_cache()

            if info == 0:
                return True
            else:
                return False

        pho = 1.25
        if_converge = check_convergence(lr)

        # while if_converge:
        #     print("increase", lr)
        #     lr = lr * pho
        #     if_converge = check_convergence(lr)

        while not if_converge:
            lr = lr / pho
            # print("decrease", lr)
            if_converge = check_convergence(lr)

        print("lr", lr)

        #update law

        for i in range(w1.shape[0]):
            delta_w1 = self.aggregator(edge_out_dot[:, i].view(edge_index.shape[1], 1) * x1, edge_index[1]).t() @ (
                        e @ (w22[:, i].unsqueeze(0).t()))
            w1[i, :] = w1[i, :] + self.alpha0*lr*delta_w1.t()

        w22 += self.alpha1*lr*(phi.t() @ e).t()
        if self.type=='pe':
            w21 += self.alpha1*lr*(x2.t() @ e).t()

        # print("loss", mse_loss(node_out, y))
        self.lr = lr

        # return node_out
        if self.type=='ee':
            return node_out, edge_out.detach()
        elif self.type=='pe':
            return node_out, edge_out, torch.cat((node_feature, aggr), dim=-1)


class MySubNode(nn.Module):
    def __init__(self, layer_type, input_size, hidden_size, sudo_input_size, output_size, args, node_layer = None, a=0):
        super().__init__()
        # self.lin_edge = MLP(hidden_size * 3, hidden_size, hidden_size, layers)
        # self.lin_node = MLP(hidden_size * 2, hidden_size, hidden_size, layers)

        # always pass through a relu layer
        # self.lin_edge = MLP_encoder(350, hidden_size, hidden_size, layers)
        # self.lin_node = MLP_encoder(200, hidden_size, hidden_size, layers)
        self.type = layer_type
        if node_layer:
            self.node_layer = node_layer
        else:
            self.node_layer = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.sudo_node_layer = torch.nn.Linear(sudo_input_size, output_size, bias=False)
        self.leaky_relu = torch.nn.functional.leaky_relu
        # self.aggregator = pyg.nn.aggr.SumAggregation()

        self.a = a

        self.alpha0=1
        self.alpha1=1
        self.alpha2=1

        self.lr = args.lr

    def forward(self, x):

        x = self.node_layer(x)
        phi = self.leaky_relu(x,self.a)
        out = self.sudo_node_layer(phi)

        return out, phi

    def train(self, x, y):
        lr = self.lr

        xw = self.node_layer(x).detach().clone()
        xw_dot = mf.derivative_fun(self.leaky_relu)(xw,self.a)
        phi = self.leaky_relu(xw,self.a).detach()
        node_out = self.sudo_node_layer(phi)

        e = y-node_out

        w1 = self.node_layer.weight.data
        w2 = self.sudo_node_layer.weight.data

        # check convergence
        def check_convergence(lr):
            e_row, e_col = e.shape
            e_dim = e_row * e_col

            condition_sum1_mx = torch.zeros((e_dim, e_dim), dtype=torch.float).to(DEVICE)
            for i in range(w1.shape[0]):
                CRX = xw_dot[:, i].view(-1, 1) * x #C=I, no need aggergator
                first_half = torch.cat(tuple(w2[:, i:i+1][:,None]*CRX),dim=0)
                condition_sum1_mx += first_half @ (first_half.t())

            # second part
            blocks1 = [phi @ phi.T] * w2.shape[0]  # Repeating a few times in a list
            # blocks2 = [x2 @ x2.T] * w21.shape[0]  # Repeating a few times in a list
            # Use torch.block_diag to create the block diagonal matrix
            # condition_sum2_mx = torch.block_diag(*blocks)
            condition_sum2_mx = torch.block_diag(*blocks1)

            # condition_sum2_mx = -(self.alpha0*lr * lr * condition_sum1_mx + self.alpha1*lr * lr * condition_sum2_mx)
            condition_sum2_mx = -(lr * lr * condition_sum1_mx + lr * lr * condition_sum2_mx)

            torch.diagonal(condition_sum2_mx).copy_((2) * lr + torch.diagonal(condition_sum2_mx))
            condition_sum1_mx = torch.empty(0)
            del condition_sum1_mx
            torch.cuda.empty_cache()

            L,info = torch.linalg.cholesky_ex(condition_sum2_mx)
            condition_sum2_mx = torch.empty(0)
            del condition_sum2_mx
            torch.cuda.empty_cache()

            if info == 0:
                return True
            else:
                return False

        pho = 1.25
        if_converge = check_convergence(lr)

        # while if_converge:
        #     print("increase", lr)
        #     lr = lr * pho
        #     if_converge = check_convergence(lr)

        while not if_converge:
            lr = lr / pho
            if_converge = check_convergence(lr)

        print("lr", lr)


        # update laws
        for i in range(w1.shape[0]):
            delta_w1 = (xw_dot[:, i].view(-1, 1) * x).t() @ (
                        e @ (w2[:, i].unsqueeze(0).t()))
            w1[i, :] = w1[i, :] + self.alpha0*lr*delta_w1.t()

        w2 += self.alpha1*lr*(phi.t() @ e).t()
        self.lr = lr
        # print("loss", mse_loss(node_out, y))

        return node_out, phi