from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_cluster import knn_graph
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, PairTensor


class EdgeConv_lrp(MessagePassing):
    """
    Copied from pytorch_geometric source code, with the following edits
    1. torch.cat([x_i, x_j - x_i], dim=-1)) -> torch.cat([x_i, x_j], dim=-1))
    2. retrieve edge_activations
    """

    def __init__(self, nn: Callable, aggr: str = "max", **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return (
            self.propagate(edge_index, x=x, size=None),
            self.edge_activations,
        )

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        # self.edge_activations = self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        # return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))
        self.edge_activations = self.nn(torch.cat([x_i, x_j], dim=-1))
        return self.nn(torch.cat([x_i, x_j], dim=-1))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


class EdgeConvBlock(nn.Module):
    def __init__(self, in_size, layer_size, depth):
        super().__init__()

        layers = []

        layers.append(nn.Linear(in_size * 2, layer_size))
        layers.append(nn.BatchNorm1d(layer_size))
        layers.append(nn.ReLU())

        for _i in range(depth):
            layers.append(nn.Linear(layer_size, layer_size))
            layers.append(nn.BatchNorm1d(layer_size))
            layers.append(nn.ReLU())

        self.edge_conv = EdgeConv_lrp(nn.Sequential(*layers), aggr="mean")

    def forward(self, x, edge_index):
        return self.edge_conv(x, edge_index)


class ParticleNet(nn.Module):
    def __init__(
        self,
        for_LRP,
        node_feat_size,
        num_classes=1,
        k=16,
        depth=2,
        dropout=False,
    ):
        super().__init__()
        self.for_LRP = for_LRP

        self.node_feat_size = node_feat_size
        self.num_classes = num_classes

        self.k = k
        self.num_edge_conv_blocks = 3

        self.kernel_sizes = [self.node_feat_size, 64, 128, 256]
        self.input_sizes = np.cumsum(self.kernel_sizes)  # [4, 4+64, 4+64+128, 4+64+128+256]

        self.fc_size = 256

        if dropout:
            self.dropout = 0.1
            self.dropout_layer = nn.Dropout(p=self.dropout)
        else:
            self.dropout = None

        # define the edgeconvblocks
        self.edge_conv_blocks = nn.ModuleList()
        for i in range(0, self.num_edge_conv_blocks):
            self.edge_conv_blocks.append(EdgeConvBlock(self.input_sizes[i], self.kernel_sizes[i + 1], depth=depth))

        # define the fully connected networks (post-edgeconvs)
        self.fc1 = nn.Linear(self.input_sizes[-1], self.fc_size)
        self.fc2 = nn.Linear(self.fc_size, self.num_classes)

        self.sig = nn.Sigmoid()

    def forward(self, batch):
        x = batch.x
        y = batch.y
        batch = batch.batch

        # input transformations
        x[:, 2] = (x[:, 2] - 1.7) * 0.7  # part_pt_log
        x[:, 3] = (x[:, 3] - 2.0) * 0.7  # part_e_log
        x[:, 4] = (x[:, 4] + 4.7) * 0.7  # part_logptrel
        x[:, 5] = (x[:, 5] + 4.7) * 0.7  # part_logerel
        x[:, 6] = (x[:, 6] - 0.2) * 4.7  # part_deltaR

        # useful placeholders for LRP studies
        edge_activations = {}
        edge_block_activations = {}
        edge_index = {}

        for i in range(self.num_edge_conv_blocks):
            # using only angular coords for knn in first edgeconv block
            edge_index[f"edge_conv_{i}"] = knn_graph(x[:, :2], self.k, batch) if i == 0 else knn_graph(x, self.k, batch)

            out, edge_activations[f"edge_conv_{i}"] = self.edge_conv_blocks[i](x, edge_index[f"edge_conv_{i}"])

            x = torch.cat((out, x), dim=1)  # concatenating with latent features i.e. skip connections per EdgeConvBlock

            edge_block_activations[f"edge_conv_{i}"] = x

        x = global_mean_pool(x, batch)

        x = F.relu(self.fc1(x))
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.fc2(x)
        x = self.sig(x)

        # save different objects if you are running lrp studies
        if self.for_LRP:
            return x, edge_activations, edge_block_activations, edge_index
        else:
            return x, y
