import torch.nn.functional as F
from torch import nn
import torch
from torch.nn import Parameter
from model.utils.utils import degree_hyedge, degree_node
from model.utils.utils import neighbor_distance, get_full_H, weight_init, create_attr_H
from einops import rearrange, repeat
from torch import einsum

class HyConv(nn.Module):
    def __init__(self, in_ch, out_ch,drop_out=0.3, bias=True) -> None:
        super().__init__()
        self.theta = Parameter(torch.Tensor(in_ch, out_ch))
        self.drop_out_ratio = drop_out

        if bias:
            self.bias = Parameter(torch.Tensor(out_ch))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self.relu = nn.LeakyReLU(inplace=True)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, H: torch.Tensor,mask=None, hyedge_weight=None):
        assert len(x.shape) == 2, 'the input of HyperConv should be N * C'
        # feature transform
        y = einsum('nc,co->no',x,self.theta)
        y = y + self.bias.unsqueeze(0)

        if hyedge_weight is not None:
            Dv = torch.diag_embed(1.0/(H*hyedge_weight).sum(1))
        else:
            Dv = torch.diag_embed(1.0/H.sum(1))
        De = torch.diag_embed(1.0/H.sum(0))
        if mask is not None:
            H = H * mask

        HDv = einsum('kv,ve->ke',Dv,H)
        HDe = einsum('ve,ek->vk',H,De)
        if hyedge_weight is not None:
            HDe *= hyedge_weight
        y = einsum('vc,ve->ec',y,HDe)

        y = einsum('ec,ve->vc',y,HDv)

        return y


class interModel(nn.Module):
    def __init__(self, in_channels, n_target, hiddens,label_in,label_hiddens, scale, k_threshold=None,k_nearest=None,dropout=0.3):
        super().__init__()
        self.drop_out = nn.Dropout(dropout)
        _in = in_channels
        self.hyconvs = []
        for _h in hiddens:
            _out = _h
            self.hyconvs.append(HyConv(_in, _out))
            _in = _out
        
        _in = label_in
        self.label_hyconvs = []
        for _h in label_hiddens:
            _out = _h
            self.label_hyconvs.append(HyConv(_in, _out))
            _in = _out

        self.label_fc = nn.Linear(label_hiddens[-1], 1)
        self.label_hyconvs = nn.ModuleList(self.label_hyconvs)
        self.hyconvs = nn.ModuleList(self.hyconvs)
        self.mlp_l1 = nn.Linear(hiddens[-1], hiddens[-1]) #+label_ft_dim
        self.mlp_l2 = nn.Linear(hiddens[-1], hiddens[-1])
        self.mlp_l3 = nn.Linear(hiddens[-1], n_target)

        self.k_nearest = k_nearest
        self.k_threshold = k_threshold
        self.reset_parameters()
    def reset_parameters(self):
        self.label_fc.apply(weight_init)
        self.mlp_l1.apply(weight_init)
        self.mlp_l2.apply(weight_init)
        self.mlp_l3.apply(weight_init)

    def forward(self, x, risk, train_fts=None, train_risk=None, attr=None,train_attr=None):
        label = risk 
        if train_fts is not None:
            H, edge_weight = self.get_ft_H(torch.concat((x,train_fts),dim=0),full=True)
            if attr is not None:
                H_attr = self.get_attr_H(torch.concat((attr,train_attr),dim=0))
                H = torch.concat((H,H_attr),dim=-1)
                x = torch.concat((x,attr),dim=-1)
                train_fts = torch.concat((train_fts,train_attr),dim=-1)
            label = torch.concat((label,train_risk),dim=0)
            x = torch.concat((x,train_fts),dim=0)
        else:
            H, edge_weight = self.get_ft_H(x, full=True)
            if attr is not None:
                H_attr = self.get_attr_H(attr)
                H = torch.concat((H,H_attr),dim=-1)
                x = torch.concat((x,attr),dim=-1)

        for hyconv in self.label_hyconvs:
            label = hyconv(label, H)
            label = self.drop_out(label)
        label = label[:risk.shape[0]]
        label = self.label_fc(label)

        for hyconv in self.hyconvs:
            x = hyconv(x, H)
            x = self.drop_out(x)
        x = x[:risk.shape[0]]
        x = x * label

        x = self.mlp_l1(x)
        x = self.mlp_l2(x)

        fts = x
        x = self.drop_out(x)
        x = self.mlp_l3(x) 


        return x, x+risk, fts

    def get_ft_H(self, fts, full=False):
        if full:
            H, edge_weight = get_full_H(fts,k_threshold=self.k_threshold,k_nearest=self.k_nearest)
            return H, edge_weight
        else:
            return neighbor_distance(fts, self.k_nearest)
    
    def get_attr_H(self, attr):
        return create_attr_H(attr)

