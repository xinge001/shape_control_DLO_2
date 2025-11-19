import torch

import torch.nn as nn
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

    def train(self, particle_effect, node_feature, edge_index, edge_feature, y, check_conv=True):
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

        if check_conv:
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
            self.lr = lr

        #update law

        for i in range(w1.shape[0]):
            delta_w1 = self.aggregator(edge_out_dot[:, i].view(edge_index.shape[1], 1) * x1, edge_index[1]).t() @ (
                        e @ (w22[:, i].unsqueeze(0).t()))
            w1[i, :] = w1[i, :] + self.alpha0*lr*delta_w1.t()

        w22 += self.alpha1*lr*(phi.t() @ e).t()
        if self.type=='pe':
            w21 += self.alpha1*lr*(x2.t() @ e).t()

        # print("loss", mse_loss(node_out, y))

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

    def train(self, x, y, check_conv = True):
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

        if check_conv:
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
            self.lr = lr


        # update laws
        for i in range(w1.shape[0]):
            delta_w1 = (xw_dot[:, i].view(-1, 1) * x).t() @ (
                        e @ (w2[:, i].unsqueeze(0).t()))
            w1[i, :] = w1[i, :] + self.alpha0*lr*delta_w1.t()

        w2 += self.alpha1*lr*(phi.t() @ e).t()

        # print("loss", mse_loss(node_out, y))

        return node_out, phi


class MySimulator(torch.nn.Module):
    """Graph Network-based Simulators(GNS)"""
    def __init__(self,args):
        super().__init__()

        # self.edge_prop_layer = torch.nn.Linear(350, 100, bias=False)
        # self.node_prop_layer = torch.nn.Linear(200, 100, bias=False)

        self.args = args

        node_input_dim = args.node_input_dim
        edge_input_dim = args.edge_input_dim

        subnet0_ee = MySubEdge('ee', edge_input_dim,100,100,2,args)
        subnet1_ee = MySubEdge('ee', 100,100,100,2,args)
        subnet2_ee = MySubEdge('ee', 100,100,100,2,args)
        subnet3_en = MySubNode('en', node_input_dim,100,100,2,args)
        subnet4_en = MySubNode('en', 100,100,100,2,args)


        subnet5_pe = MySubEdge('pe', 300,100,200,2,args)
        subnet6_pn = MySubNode('pn', 200,100,100,2,args)

        subnet7_pe = MySubEdge('pe', 300,100,200,2,args)
        subnet8_pn = MySubNode('pn', 200,100,100,2,args)

        subnet9_pe = MySubEdge('pe',  300,100,200,2,args)
        subnet10_pn = MySubNode('pn', 200,100,100,2,args)

        subnet11_dn = MySubNode('dn', 100, 100, 100, 2,args)


        subnet5_pe_v = MySubEdge('pe', 300,100,200,2,args)
        subnet6_pn_v = MySubNode('pn', 200,100,100,2,args)

        subnet7_pe_v = MySubEdge('pe', 300,100,200,2,args)
        subnet8_pn_v = MySubNode('pn', 200,100,100,2,args)

        subnet9_pe_v = MySubEdge('pe',  300,100,200,2,args)
        subnet10_pn_v = MySubNode('pn', 200,100,100,2,args)

        subnet11_dn_v = MySubNode('dn', 100, 100, 100, 2,args)

        # self.layers = torch.nn.ModuleList([MySubEdge('pe', 350,100, 200, 2), MySubNode(200, 100, 2), MySubNode(100, 100, 2)])
        # self.subnets= torch.nn.ModuleList([subnet0_ee, subnet1_ee, subnet2_ee, subnet3_en, subnet4_en, subnet5_pe, subnet6_pn, subnet5_pe, subnet6_pn, subnet5_pe, subnet6_pn, subnet11_dn])
        # self.subnets= torch.nn.ModuleList([subnet0_ee, subnet3_en, subnet5_pe, subnet6_pn])
        self.subnets= torch.nn.ModuleList([subnet0_ee, subnet1_ee, subnet2_ee, subnet3_en, subnet4_en, subnet5_pe, subnet6_pn, subnet7_pe, subnet8_pn,  subnet9_pe, subnet10_pn, subnet11_dn])
        self.subnets_v= torch.nn.ModuleList([subnet0_ee, subnet1_ee, subnet2_ee, subnet3_en, subnet4_en, subnet5_pe_v, subnet6_pn_v, subnet7_pe_v, subnet8_pn_v,  subnet9_pe_v, subnet10_pn_v, subnet11_dn_v])


        # ensure the initial weights are same for main nets, and virtual nets
        for i, subnet in enumerate(self.subnets_v):
            if subnet.type == 'pe':
                subnet.edge_layer.weight.data = self.subnets[i].edge_layer.weight.data.clone()
                subnet.sudo_node_layer.weight.data = self.subnets[i].sudo_node_layer.weight.data.clone()
            elif subnet.type == 'pn' or subnet.type == 'dn':
                subnet.node_layer.weight.data = self.subnets[i].node_layer.weight.data.clone()
                subnet.sudo_node_layer.weight.data = self.subnets[i].sudo_node_layer.weight.data.clone()

    def my_load_weights(self, checkpoint_path, layers_to_load=[]):
        # Load the state dictionary from a saved file
        state_dict = torch.load(checkpoint_path, map_location='cuda:0')

        # Create a new dictionary to store the filtered layers that we want to load
        filtered_state_dict = {}
        # Specify the layers we want to load (for example, we only want to load layer1 and layer2)
        # Iterate over the state dictionary and filter out the layers we want
        if layers_to_load:
            for layer_name in state_dict:
                if layer_name in layers_to_load:
                    # print("load", layer_name)
                    filtered_state_dict[layer_name] = state_dict[layer_name]
        else:  # load all
            for layer_name in state_dict:
                # print("load", layer_name)
                filtered_state_dict[layer_name] = state_dict[layer_name]

        print("Param names trying to load")
        for name in filtered_state_dict:
            print(name)

        # Load the filtered state dictionary into the model
        self.load_state_dict(filtered_state_dict, strict=False)

        # Freeze the layers with preloaded weights
        # if layers_to_load:
        #     for name, param in self.named_parameters():
        #         if name in layers_to_load:
        #             param.requires_grad = False
        # else:
        #     for name, param in self.named_parameters():
        #         param.requires_grad = False

        # 'strict=False' allows partial loading, meaning if some layers in the model are missing
        # from the filtered_state_dict, it will not raise an error.

    def load_weights_main_virtual(self):
        # make sure main and virtual systems have the same initial/loaded weights before training
        for i, subnet in enumerate(self.subnets_v):
            if subnet.type == 'pe':
                subnet.edge_layer.weight.data = self.subnets[i].edge_layer.weight.data.clone()
                subnet.sudo_node_layer.weight.data = self.subnets[i].sudo_node_layer.weight.data.clone()
            elif subnet.type == 'pn' or subnet.type == 'dn':
                subnet.node_layer.weight.data = self.subnets[i].node_layer.weight.data.clone()
                subnet.sudo_node_layer.weight.data = self.subnets[i].sudo_node_layer.weight.data.clone()

    def exchange_weights(self):
        for i, subnet in enumerate(self.subnets):
            if subnet.type == 'pe':
                subnet.sudo_node_layer.weight.data = self.subnets_v[i].sudo_node_layer.weight.data.clone()
            elif subnet.type == 'pn' or subnet.type == 'dn':
                subnet.sudo_node_layer.weight.data = self.subnets_v[i].sudo_node_layer.weight.data.clone()

        for i, subnet in enumerate(self.subnets_v):
            if subnet.type == 'pe':
                subnet.edge_layer.weight.data = self.subnets[i].edge_layer.weight.data.clone()
            elif subnet.type == 'pn' or subnet.type == 'dn':
                subnet.node_layer.weight.data = self.subnets[i].node_layer.weight.data.clone()

    def exchange_lr(self):
        for i, subnet in enumerate(self.subnets_v):
            if subnet.type == 'pe':
                subnet.lr = self.subnets[i].lr
            elif subnet.type == 'pn' or subnet.type == 'dn':
                subnet.lr = self.subnets[i].lr

    def forward(self, data):
        node_feature = data.x
        edge_feature = data.edge_attr
        # particle_effect = torch.zeros((node_feature.size(0), 100)).to(DEVICE)
        for subnet in self.subnets:
            if subnet.type == 'ee':
                node_out, edge_feature = subnet.forward(node_feature, node_feature, data.edge_index, edge_feature)
            elif subnet.type == 'en':
                node_out, node_feature = subnet.forward(node_feature)
            elif subnet.type == 'pe':
                node_out, edge_feature, input_to_node = subnet.forward(node_feature, node_feature, data.edge_index, edge_feature)
            elif subnet.type == 'pn':
                node_out, node_feature = subnet.forward(input_to_node)
            elif subnet.type == 'dn':
                node_out, _ = subnet.forward(node_feature)
        return node_out

    def forward_to_subnet(self, data, last_net):
        node_feature = data.x
        edge_feature = data.edge_attr
        # particle_effect = torch.zeros((node_feature.size(0), 100)).to(DEVICE)
        for i, subnet in enumerate(self.subnets):
            if i<=last_net:
                if subnet.type == 'ee':
                    node_out, edge_feature = subnet.forward(node_feature, node_feature, data.edge_index, edge_feature)
                elif subnet.type == 'en':
                    node_out, node_feature = subnet.forward(node_feature)
                elif subnet.type == 'pe':
                    node_out, edge_feature, input_to_node = subnet.forward(node_feature, node_feature, data.edge_index,
                                                                           edge_feature)
                elif subnet.type == 'pn':
                    node_out, node_feature = subnet.forward(input_to_node)
                elif subnet.type == 'dn':
                    node_out, _ = subnet.forward(node_feature)
            else:
                continue
        return node_out

    def train_sub_node(self, layer, x, y, lr=0.001):
        node_out = self.layers[layer].train(x, y, lr=lr)
        return node_out

    def train_all_subnets(self, data, check_conv=True):
        i_range = self.args.train_nets
        node_feature = data.x
        edge_feature = data.edge_attr
        particle_effect = torch.zeros((node_feature.size(0), 100)).to(DEVICE)
        for i, subnet in enumerate(self.subnets):
            if subnet.type == 'ee':
                if i in i_range:
                    node_out, edge_feature = subnet.train(node_feature, node_feature, data.edge_index, edge_feature, data.y, check_conv)
                else:
                    _, edge_feature = subnet(node_feature, node_feature, data.edge_index, edge_feature)
            elif subnet.type == 'en':
                if i in i_range:
                    node_out, node_feature = subnet.train(node_feature, data.y, check_conv)
                else:
                    _, node_feature = subnet(node_feature)
            elif subnet.type == 'pe':
                if i in i_range:
                    node_out, edge_feature, input_to_node = subnet.train(node_feature, node_feature, data.edge_index,
                                                                           edge_feature, data.y, check_conv)
                else:
                    _, edge_feature, input_to_node = subnet(node_feature, node_feature, data.edge_index,
                                                                           edge_feature)
            elif subnet.type == 'pn':
                if i in i_range:
                    node_out, node_feature = subnet.train(input_to_node, data.y, check_conv)
                else:
                    _, node_feature = subnet(input_to_node)
            elif subnet.type == 'dn':
                if i in i_range:
                    node_out, _ = subnet.train(node_feature, data.y, check_conv)
                else:
                    _,_ = subnet(node_feature)

        loss = loss_fn(node_out, data.y)
        return loss


    def train_all_subnets_v(self, data, check_conv=True):
        i_range = self.args.train_nets
        node_feature = data.x
        edge_feature = data.edge_attr
        particle_effect = torch.zeros((node_feature.size(0), 100)).to(DEVICE)
        for i, subnet in enumerate(self.subnets_v):
            if subnet.type == 'ee':
                if i in i_range:
                    node_out, edge_feature = subnet.train(node_feature, node_feature, data.edge_index, edge_feature, data.y, check_conv)
                else:
                    _, edge_feature = subnet(node_feature, node_feature, data.edge_index, edge_feature)
            elif subnet.type == 'en':
                if i in i_range:
                    node_out, node_feature = subnet.train(node_feature, data.y, check_conv)
                else:
                    _, node_feature = subnet(node_feature)
            elif subnet.type == 'pe':
                if i in i_range:
                    node_out, edge_feature, input_to_node = subnet.train(node_feature, node_feature, data.edge_index,
                                                                           edge_feature, data.y, check_conv)
                else:
                    _, edge_feature, input_to_node = subnet(node_feature, node_feature, data.edge_index,
                                                                           edge_feature)
            elif subnet.type == 'pn':
                if i in i_range:
                    node_out, node_feature = subnet.train(input_to_node, data.y, check_conv)
                else:
                    _, node_feature = subnet(input_to_node)
            elif subnet.type == 'dn':
                if i in i_range:
                    node_out, _ = subnet.train(node_feature, data.y, check_conv)
                else:
                    _,_ = subnet(node_feature)

        loss = loss_fn(node_out, data.y)
        return loss

