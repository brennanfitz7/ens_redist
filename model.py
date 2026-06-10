import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv

class my_gat(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in, dim_out=20, edge_dim=3, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, 20, edge_dim=edge_dim,heads=heads)
    self.gat2 = GATv2Conv(20*heads, 40, edge_dim=edge_dim,heads=1)
    self.linear1 = nn.Linear(40, 10)


  def forward(self, x, edge_index, edge_attr):
    h = F.dropout(x, p=0.3, training=self.training)
    h = self.gat1(h, edge_index, edge_attr=edge_attr)
    h = F.elu(h)
    h = F.dropout(h, p=0.3, training=self.training)
    h = self.gat2(h, edge_index, edge_attr=edge_attr)
    h = F.elu(h)
    h= self.linear1(h)
      
    return h


class both_structs(torch.nn.Module):
    def __init__(self, dim_in, edge_dim=3, heads=8):
        super().__init__()
        self.single_gat= my_gat(dim_in=dim_in)
        self.linear1 = nn.Linear(40, 20)
        self.linear2 = nn.Linear(20, 30)
        self.linear3 = nn.Linear(30, 1)

        self.optimizer = torch.optim.Adam(self.parameters(),
                                    lr=0.005,
                                    weight_decay=5e-4)


    def forward(self, data1, data2):

        #run the single gats on both the structures
        h1=self.single_gat(data1.x, data1.edge_index, edge_attr=data1.edge_attr)
        h2=self.single_gat(data2.x, data2.edge_index, edge_attr=data2.edge_attr)

        #concatenate the structures with the residue information
        h=torch.cat((h1,h2,data1.resid),1)

        #run through an MLP
        h=self.linear1(h)
        h=F.relu(h)
        h=self.linear2(h)
        h=F.relu(h)
        h=self.linear3(h)

        return h
 

