import torch
from einops import rearrange, repeat
from torch import einsum
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn as nn
from torch.nn import ReLU, SELU, PReLU, GELU, ELU, LeakyReLU

def degree_node(H):
    node_idx, edge_idx = H
    node_num = H[1].max().item() + 1
    src = torch.ones_like(node_idx).float().to(H.device)
    out = torch.zeros(node_idx.size(0),node_num).to(H.device)
    return out.scatter_add(1, node_idx, src).long()
    # return torch.zeros(node_num).scatter_add(0, node_idx, torch.ones_like(node_idx).float()).long()


def degree_hyedge(H: torch.Tensor):
    node_idx, hyedge_idx = H
    edge_num = H[0].max().item() + 1
    src = torch.ones_like(hyedge_idx).float().to(H.device)
    out = torch.zeros(hyedge_idx.size(0),edge_num).to(H.device)
    return out.scatter_add(1, hyedge_idx, src).long()

def pairwise_euclidean_distance_3d(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 3
    x = x.float()
    
    x_transpose = rearrange(x, 'n v c -> n c v').contiguous() #torch.transpose(x, dim0=0, dim1=1)
    x_inner = einsum('nvc,nck -> nvk', x, x_transpose) #torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=-1, keepdim=True)
    x_square_transpose = rearrange(x_square, 'n v c -> n c v').contiguous()
    dis = x_square + x_inner + x_square_transpose
    return dis
def pairwise_euclidean_distance_2d(x: torch.Tensor):
    """
    generate N x N node distance matrix
    :param x: a tensor of size N x C (N nodes with C feature dimension)
    :return: a tensor of N x N (distance matrix for each node pair)
    """
    assert isinstance(x, torch.Tensor)
    assert len(x.shape) == 2
    x = x.float()

    x_transpose = torch.transpose(x, dim0=0, dim1=1)
    x_inner = torch.matmul(x, x_transpose)
    x_inner = -2 * x_inner
    x_square = torch.sum(x ** 2, dim=1, keepdim=True)
    x_square_transpose = torch.transpose(x_square, dim0=0, dim1=1)
    dis = x_square + x_inner + x_square_transpose
    return dis

def neighbor_distance(x: torch.Tensor, k_nearest, dis_metric=pairwise_euclidean_distance_3d):
    """
    construct hyperedge for each node in x matrix. Each hyperedge contains a node and its k-1 nearest neighbors.
    :param x: N x C matrix. N denotes node number, and C is the feature dimension.
    :param k_nearest:
    :return:
    """

    assert len(x.shape) == 3, 'should be a tensor with dimension (N * V * C)'

    # N x C
    node_num = x.size(1)
    dis_matrix = dis_metric(x)
    _, nn_idx = torch.topk(dis_matrix, k_nearest, dim=-1, largest=False)
    hyedge_idx = torch.arange(node_num).to(x.device).unsqueeze(0).repeat(k_nearest, 1).transpose(1, 0).reshape(-1).unsqueeze(0).repeat(x.size(0), 1)
    H = torch.stack([nn_idx.reshape(x.size(0),-1), hyedge_idx])
    return H

def get_full_select_H(x: torch.Tensor, top_k, k_threshold, dis_metric=pairwise_euclidean_distance_3d):
    ## top_k means the number of selected hyperedges
    assert len(x.shape) == 3, 'should be a tensor with dimension (N * V * C)'

    # N x V x C
    N, V, C = x.size()
    dis_matrix = dis_metric(x)
    max_dis_matrix, _ = dis_matrix.max(-1,keepdim=True)
    norm_dis_matrix = dis_matrix / max_dis_matrix
    ones = torch.ones_like(dis_matrix)
    zeros = torch.zeros_like(dis_matrix)
    H = torch.where(norm_dis_matrix<k_threshold,ones,zeros)
    _, indices = H.sum(-1).topk(k=top_k,dim=-1)
    H_select = torch.stack([H[i,:,indices[i,:]] for i in range(N)])
    return H_select

def get_full_H_3d(x: torch.Tensor, k_nearest=None,k_threshold=None, largest=False, dis_metric=pairwise_euclidean_distance_3d):

    assert len(x.shape) == 3, 'should be a tensor with dimension (N * V * C)'

    # N x V x C
    N, V, C = x.size()
    H_nearest = None
    H_threshold = None
    dis_matrix = dis_metric(x)
    max_dis_matrix, _ = dis_matrix.max(1,keepdim=True)
    norm_dis_matrix = dis_matrix / max_dis_matrix
    if k_nearest is not None:
        min_topk, _ = torch.topk(dis_matrix, k_nearest, dim=1, largest=largest)
        min_threshold = min_topk[:,-1,:].unsqueeze(1).repeat(1,V,1)
        ones = torch.ones_like(dis_matrix)
        zeros = torch.zeros_like(dis_matrix)
        H_nearest = torch.where(dis_matrix<=min_threshold,ones,zeros) # dis_matrix
        H_nearest_weight = 1-(((H_nearest * norm_dis_matrix).sum(dim=-2)) / H_nearest.sum(dim=-2)) #F.sigmoid

    if k_threshold is not None:
        ones = torch.ones_like(dis_matrix)
        zeros = torch.zeros_like(dis_matrix)
        H_threshold = torch.where(norm_dis_matrix<k_threshold,ones,zeros)
        H_threshold_weight = 1-(((H_threshold * norm_dis_matrix).sum(dim=-2)) / H_threshold.sum(dim=-2)) #F.sigmoid
        # vertex_count = torch.sum(H_threshold, dim=1)
        # single_vertex_hyperedges = torch.where(vertex_count == 1)
        # H_threshold[single_vertex_hyperedges] = 0
    if H_threshold is not None and H_nearest is not None:
        return torch.concat((H_nearest,H_threshold),dim=-1), torch.concat((H_nearest_weight,H_threshold_weight),dim=-1)
    elif H_threshold is not None:
        return H_threshold, H_threshold_weight
    else:
        return H_nearest, H_nearest_weight

def get_full_H_2d(x: torch.Tensor, k_nearest=None,k_threshold=None, largest=False, dis_metric=pairwise_euclidean_distance_2d):

    assert len(x.shape) == 2, 'should be a tensor with dimension (N * C)'

    # N x V x C
    N, C = x.size()
    H_nearest = None
    H_threshold = None
    dis_matrix = dis_metric(x)
    max_dis_matrix, _ = dis_matrix.max(0,keepdim=True)
    norm_dis_matrix = dis_matrix / max_dis_matrix
    if k_nearest is not None:
        min_topk, _ = torch.topk(dis_matrix, k_nearest, dim=0, largest=largest)
        min_threshold = min_topk[-1,:].unsqueeze(0)
        ones = torch.ones_like(dis_matrix)
        zeros = torch.zeros_like(dis_matrix)
        H_nearest = torch.where(dis_matrix<=min_threshold,ones,zeros)
        H_nearest_weight = 1-(((H_nearest * norm_dis_matrix).sum(dim=0,keepdim=True)) / H_nearest.sum(dim=0,keepdim=True)) #F.sigmoid

    if k_threshold is not None:
        ones = torch.ones_like(dis_matrix)
        zeros = torch.zeros_like(dis_matrix)
        H_threshold = torch.where(norm_dis_matrix<k_threshold,ones,zeros)
        H_threshold_weight = 1-(((H_threshold * norm_dis_matrix).sum(dim=0,keepdim=True)) / H_threshold.sum(dim=0,keepdim=True)) #F.sigmoid

    if H_threshold is not None and H_nearest is not None:
        return torch.concat((H_nearest,H_threshold),dim=-1), torch.concat((H_nearest_weight,H_threshold_weight),dim=-1)
    elif H_threshold is not None:
        return H_threshold, H_threshold_weight
    else:
        return H_nearest, H_nearest_weight

def get_full_H(x: torch.Tensor, k_nearest=None,k_threshold=None, largest=False):
    if len(x.shape) == 3:
        H, edge_weight = get_full_H_3d(x,k_nearest,k_threshold,largest)
        return H, edge_weight
    elif len(x.shape) == 2:
        H, edge_weight = get_full_H_2d(x,k_nearest,k_threshold,largest)
        return H, edge_weight
    
def get_fusion_H_3d(x: torch.Tensor, k_nearest=None,k_threshold=None, largest=False, dis_metric=pairwise_euclidean_distance_3d):

    assert len(x.shape) == 3, 'should be a tensor with dimension (N * V * C)'

    # N x V x C
    N, V, C = x.size()
    H_nearest = None
    H_threshold = None
    dis_matrix = dis_metric(x)
    ones = torch.ones_like(dis_matrix)
    zeros = torch.zeros_like(dis_matrix)
    max_dis_matrix, _ = dis_matrix.max(1,keepdim=True)
    norm_dis_matrix = dis_matrix / max_dis_matrix
    if k_nearest is not None:
        min_topk, _ = torch.topk(dis_matrix, k_nearest, dim=1, largest=largest)
        min_threshold = min_topk[:,-1,:].unsqueeze(1).repeat(1,V,1)
        H_nearest = torch.where(dis_matrix<=min_threshold,ones,zeros)
        H_nearest_weight = 1-(((H_nearest * norm_dis_matrix).sum(dim=-1)) / H_nearest.sum(dim=-1)) #F.sigmoid

    if k_threshold is not None:
        max_dis_matrix, _ = dis_matrix.max(1,keepdim=True)
        norm_dis_matrix = dis_matrix / max_dis_matrix
        H_threshold = torch.where(norm_dis_matrix<k_threshold,ones,zeros)
        H_threshold_weight = 1-(((H_threshold * norm_dis_matrix).sum(dim=-1)) / H_threshold.sum(dim=-1)) #F.sigmoid
        
    if H_threshold is not None and H_nearest is not None:
        H = H_nearest + H_threshold
        H = torch.where(H>0,ones,zeros)
        H_weight = 1-(((H * norm_dis_matrix).sum(dim=-1)) / H.sum(dim=-1))
        return H, H_weight
    elif H_threshold is not None:
        return H_threshold, H_threshold_weight
    else:
        return H_nearest, H_nearest_weight

def get_fusion_H_2d(x: torch.Tensor, k_nearest=None,k_threshold=None, largest=False, dis_metric=pairwise_euclidean_distance_2d):

    assert len(x.shape) == 2, 'should be a tensor with dimension (N * C)'

    # N x V x C
    N, C = x.size()
    H_nearest = None
    H_threshold = None
    dis_matrix = dis_metric(x)
    ones = torch.ones_like(dis_matrix)
    zeros = torch.zeros_like(dis_matrix)
    max_dis_matrix, _ = dis_matrix.max(0,keepdim=True)
    norm_dis_matrix = dis_matrix / max_dis_matrix
    if k_nearest is not None:
        min_topk, _ = torch.topk(dis_matrix, k_nearest, dim=0, largest=largest)
        min_threshold = min_topk[-1,:].unsqueeze(0)
        H_nearest = torch.where(dis_matrix<=min_threshold,ones,zeros)
        H_nearest_weight = 1-(((H_nearest * norm_dis_matrix).sum(dim=-1)) / H_nearest.sum(dim=-1)) #F.sigmoid

    if k_threshold is not None:
        H_threshold = torch.where(norm_dis_matrix<k_threshold,ones,zeros)
        H_threshold_weight = 1-(((H_threshold * norm_dis_matrix).sum(dim=-1)) / H_threshold.sum(dim=-1)) #F.sigmoid

    if H_threshold is not None and H_nearest is not None:
        H = H_nearest + H_threshold
        H = torch.where(H>0,ones,zeros)
        H_weight = 1-(((H * norm_dis_matrix).sum(dim=-1)) / H.sum(dim=-1))
        return H, H_weight
    elif H_threshold is not None:
        return H_threshold, H_threshold_weight
    else:
        return H_nearest, H_nearest_weight
    
def get_fusion_H(x: torch.Tensor, k_nearest=None,k_threshold=None, largest=False):
    if len(x.shape) == 3:
        H, edge_weight = get_fusion_H_3d(x,k_nearest,k_threshold,largest)
        return H, edge_weight
    elif len(x.shape) == 2:
        H, edge_weight = get_fusion_H_2d(x,k_nearest,k_threshold,largest)
        return H, edge_weight


def create_attr_H(attr):
    H = None
    N, C = attr.size()
    for c in range(C):
        single_attr = attr[:,c]
        unique_attr = torch.unique(single_attr)
        for elem in unique_attr:
            zeros = torch.zeros_like(single_attr)
            zeros[single_attr==elem] = 1.0
            if H is None:
                H = zeros.unsqueeze(-1)
            else:
                H = torch.concat((H,zeros.unsqueeze(-1)),dim=-1)
    return H

def Graph_from_hypergraph(H, weights=True):
    G = einsum('n v e, n k e->n v k', H, H)
    if not weights:
        G[G!=0] = 1.0
    G_degree = G.sum(-1).unsqueeze(-1)
    G = G * (1 / G_degree)
    return G

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

def decide_loss_type(loss_type, dim):

    if loss_type == "RELU":
        loss_fun = ReLU()
    if loss_type == "Leaky":
        loss_fun = LeakyReLU(negative_slope=0.2)
    elif loss_type == "PRELU":
        loss_fun = PReLU(init=0.2, num_parameters=dim)
    else:
        loss_fun = PReLU(init=0.2, num_parameters=dim)

    return loss_fun