import torch
from torch import nn
from utils import givens_rotation, euc_distance
from torch.nn import functional as F


from dgl.nn import RelGraphConv, GATConv, GraphConv
# Models for global graph 

class GNN(nn.Module):
    def __init__(self, in_dim, out_dim, layer_num, num_rel, num_head, gnn='rgat', att_drop=0.2, fea_drop=0.2,):
        super(GNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.layer_num = layer_num
        self.num_rel = num_rel
        self.num_head = num_head
        self.gnn = gnn
        self.att_drop = att_drop
        self.feature_drop = fea_drop

        if self.gnn == "rgcn":
            self.gnn_layer = nn.ModuleList(RelGraphConv(
                self.in_dim, self.out_dim, num_rels=self.num_rel, regularizer='basis',
                num_bases=10, activation=F.relu, dropout=0.3
            ) for _ in range(self.layer_num))

        elif self.gnn == "gat":
            """
            input: (num, in_dim)
            out: (num, num_head, out_dim)
            """
            self.gnn_layer = nn.ModuleList(GATConv(
                self.in_dim, int(self.out_dim / self.num_head), num_head, self.feature_drop,
                self.att_drop, activation=F.relu 
            ) for _ in range(self.layer_num))

        elif self.gnn == 'rgat':
            self.gnn_layer = nn.ModuleList(RGAT(
                self.in_dim, self.out_dim, self.att_drop, self.feature_drop
            ) for _ in range(self.layer_num))

    
    def forward(self, g, feature,):
        for fn in self.gnn_layer:
            feature = fn(g, feature)
        return feature
        


class RGAT(nn.Module):
    def __init__(self, in_dim, out_dim, att_drop, fea_drop):
        super(RGAT, self).__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        self.w_r = nn.Linear(in_dim, out_dim, bias=False)
        self.att = nn.Linear(out_dim * 3, 1, bias=False)
        self.att_drop = nn.Dropout(att_drop)
        self.feature_drop = nn.Dropout(fea_drop)
        self.loop_weight = nn.Parameter(torch.Tensor(out_dim, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.w.weight, gain=gain)
        nn.init.xavier_uniform_(self.att.weight, gain=gain)

        nn.init.xavier_uniform_(self.loop_weight, gain=gain)

    def edge_attention(self, edges):
        edges.src['h'] = edges.src['h'].squeeze(1)
        edges.dst['h'] = edges.dst['h'].squeeze(1)
        z3 = torch.cat([edges.src['h'], edges.dst['h'], edges.data['r_h']], dim=1)
        att_ = self.att(z3)
        return {'e' : F.leaky_relu(att_)} # attention score before softmax

    def message_func(self, edges):
        return {'h' : edges.src['h'], 'e': edges.data['e'], 'r_h': edges.data['r_h']}

    def reduce_func(self, nodes):
        alpha = self.att_drop(F.softmax(nodes.mailbox['e'], dim=1)) # \alpha
        h = self.feature_drop(torch.sum(alpha * (nodes.mailbox['h'] + nodes.mailbox['r_h']), dim=1) + torch.mm(nodes.data['h'], self.loop_weight))
        return {'h': h}

    def forward(self, g, h):
        h = self.w(h)
        with g.local_scope():
            g.ndata['h'] = h
            g.edata['r_h'] = self.w_r(g.edata['r_h'])
            g.apply_edges(self.edge_attention) # create att
            g.update_all(self.message_func, self.reduce_func)
            return F.relu(g.ndata.pop('h'))

class gnn_kge(nn.Module):
    """
    gnn_kge: finish pretrained embeddings using R-GCN like network and KGE
    """
    def __init__(self, graph, num_nodes, num_rels, hidden_dim, score_func, 
                 layer_num, num_head, gnn='rgat', att_drop=0.2, fea_drop=0.2,):
        super().__init__()
        self.graph = graph
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.hidden_dim = hidden_dim
        self.ent_embedding = nn.Embedding(self.num_nodes, self.hidden_dim)
        self.rel_embedding = nn.Embedding(self.num_rels * 2, self.hidden_dim)
        nn.init.xavier_normal_(self.ent_embedding.weight, gain=1.414)
        nn.init.xavier_normal_(self.rel_embedding.weight, gain=1.414)
        self.score_func = score_func

        self.gnn_model = GNN(hidden_dim, hidden_dim, layer_num, num_rels, num_head, gnn, att_drop, fea_drop)

    def gnn_forward(self, ):
        total_e = self.ent_embedding(self.graph.ndata['id'].squeeze(1))
        # self.graph.edata['r_h'] = self.rel_embedding(self.graph.edata['type'])
        # total_e = self.gnn_model(self.graph, total_e)
        return total_e
        
    def forward(self, triples, new_feature):
        h = new_feature[triples[:, 0]]
        r = self.rel_embedding(triples[:, 1])
        t = new_feature
        if self.score_func == "RotatE":
            target = givens_rotation(r, h)
            score = -euc_distance(target, t, eval_mode=True)
        return score