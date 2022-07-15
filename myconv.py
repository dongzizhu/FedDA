"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

from dgl import function as fn
from dgl.nn.pytorch import edge_softmax
from dgl._ffi.base import DGLError
from dgl.nn.pytorch.utils import Identity
from dgl.utils import expand_as_pair

# pylint: enable=W0235
class myGATConv(nn.Module):
    """
    Adapted from
    https://docs.dgl.ai/_modules/dgl/nn/pytorch/conv/gatconv.html#GATConv
    """
    def __init__(self,
                 edge_feats,
                 num_etypes,
                 num_ntypes,
                 in_dims, 
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None,
                 allow_zero_in_degree=False,
                 bias=False,
                 alpha=0.):
        super(myGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_ntypes = num_ntypes
        self._num_etypes = num_etypes
        # self.edge_emb = nn.Embedding(num_etypes, edge_feats)
        self.edge_emb = nn.ParameterList([nn.Parameter(th.FloatTensor(size=(1, edge_feats))) for _ in range(num_etypes)])
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            # self.fc_list = nn.ModuleList([nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False) for in_dim in in_dims])
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_e = nn.Linear(edge_feats, edge_feats*num_heads, bias=False)
        # self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        # self.attn_e = nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats)))
        # self.attn_l = nn.Embedding(num_ntypes, 1 * num_heads * out_feats)
        # self.attn_r = nn.Embedding(num_ntypes, 1 * num_heads * out_feats)
        # self.attn_e = nn.Embedding(num_etypes, 1 * num_heads * edge_feats)
        
        self.attn_l = nn.ParameterList([nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats))) for _ in range(num_ntypes)])
        self.attn_r = nn.ParameterList([nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats))) for _ in range(num_ntypes)])
        self.attn_e = nn.ParameterList([nn.Parameter(th.FloatTensor(size=(1, num_heads, edge_feats))) for _ in range(num_etypes)])
        
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.bias = bias
        if bias:
            self.bias_param = nn.Parameter(th.zeros((1, num_heads, out_feats)))
        self.alpha = alpha

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        for i in range(self._num_ntypes):
            nn.init.xavier_normal_(self.attn_l[i], gain=gain)
            nn.init.xavier_normal_(self.attn_r[i], gain=gain)
        for i in range(self._num_etypes):
            nn.init.xavier_normal_(self.attn_e[i], gain=gain)
            nn.init.xavier_normal_(self.edge_emb[i], gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_e.weight, gain=gain)
        
        # not sure
        # self.edge_emb.reset_parameters()
        # self.attn_l.reset_parameters()
        # self.attn_r.reset_parameters()
        # self.attn_e.reset_parameters()

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, e_feat, e_count, n_count, res_attn=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            
            device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
            e_feat.sort(key=lambda x: x[0])
            all_index = [x[1] for x in e_feat]
            e_feat = [x[0] for x in e_feat]
            e_feat = th.tensor(e_feat, dtype=th.long).to(device)
            all_index = th.tensor(all_index, dtype=th.long).to(device)
            
            _, all_index2 = th.sort(all_index)
            
            # embed_e_feat = self.edge_emb(e_feat)
            embed_e_feat = th.cat([e_feat[sum(e_count[0:i]):sum(e_count[0:i+1])].unsqueeze(-1) * self.edge_emb[e_feat[sum(e_count[0:i])]] for i in range(len(e_count))], 0)
            embed_e_feat = self.fc_e(embed_e_feat).view(-1, self._num_heads, self._edge_feats)
            # ee = (embed_e_feat * self.attn_e).sum(dim=-1).unsqueeze(-1)
            # el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            # er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            
            # ee = th.cat([embed_e_feat[sum(e_count[0:i]):sum(e_count[0:i+1])] * self.attn_e(e_feat[sum(e_count[0:i])]).view(-1, self._num_heads, self._edge_feats) for i in range(len(e_count))], 0).sum(dim=-1).unsqueeze(-1)
            # el = th.cat([feat_src[sum(n_count[0:i]):sum(n_count[0:i+1])] * self.attn_l(th.tensor([i], dtype=th.long).to(device)[0]).view(-1, self._num_heads, self._out_feats) for i in range(len(n_count))], 0).sum(dim=-1).unsqueeze(-1)
            # er = th.cat([feat_dst[sum(n_count[0:i]):sum(n_count[0:i+1])] * self.attn_r(th.tensor([i], dtype=th.long).to(device)[0]).view(-1, self._num_heads, self._out_feats) for i in range(len(n_count))], 0).sum(dim=-1).unsqueeze(-1)
            
            ee = th.cat([embed_e_feat[sum(e_count[0:i]):sum(e_count[0:i+1])] * self.attn_e[e_feat[sum(e_count[0:i])]] for i in range(len(e_count))], 0).sum(dim=-1).unsqueeze(-1)
            el = th.cat([feat_src[sum(n_count[0:i]):sum(n_count[0:i+1])] * self.attn_l[th.tensor([i], dtype=th.long).to(device)[0]] for i in range(len(n_count))], 0).sum(dim=-1).unsqueeze(-1)
            er = th.cat([feat_dst[sum(n_count[0:i]):sum(n_count[0:i+1])] * self.attn_r[th.tensor([i], dtype=th.long).to(device)[0]] for i in range(len(n_count))], 0).sum(dim=-1).unsqueeze(-1)
            
            
            ee = ee.index_select(0, all_index2)
            
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.edata.update({'ee': ee})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e')+graph.edata.pop('ee'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            if res_attn is not None:
                graph.edata['a'] = graph.edata['a'] * (1-self.alpha) + res_attn * self.alpha
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias:
                rst = rst + self.bias_param
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst, graph.edata.pop('a').detach()

