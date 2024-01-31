
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import numpy as np
from scipy.sparse import find
import seaborn as sns
import matplotlib.pyplot as plt
class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """

    def __init__(self, n_hops, n_users, interact_mat,
                 edge_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        self.interact_mat = interact_mat
        self.n_users = n_users
        self.n_hops = n_hops
        self.edge_dropout_rate = edge_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        
        self.dropout = nn.Dropout(p=mess_dropout_rate)  # mess dropout

    def _sparse_dropout(self, x, rate=0.617):
        noise_shape = x._nnz()

        random_tensor = rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - rate))
    

    def forward(self, user_embed, item_embed,
                mess_dropout=True, edge_dropout=True):
        # user_embed: [n_users, channel]
        # item_embed: [n_items, channel]

        # all_embed: [n_users+n_items, channel]
        all_embed = torch.cat([user_embed, item_embed], dim=0)
        agg_embed = all_embed
        embs = [all_embed]

        for hop in range(self.n_hops):
            interact_mat = self._sparse_dropout(self.interact_mat,
                                                self.edge_dropout_rate) if edge_dropout \
                else self.interact_mat
            agg_embed = torch.sparse.mm(interact_mat, agg_embed)
            if mess_dropout:
                agg_embed = self.dropout(agg_embed)
                # random_noise = torch.rand_like(agg_embed).cuda()
                # agg_embed += torch.sign(agg_embed) * F.normalize(random_noise, dim=-1) * self.eps
            # agg_embed = F.normalize(agg_embed)
            embs.append(agg_embed)
        embs = torch.stack(embs, dim=1)  # [n_entity, n_hops+1, emb_size]
        return embs[:self.n_users, :], embs[self.n_users:, :]


class LightGCN1(nn.Module):
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN1, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat

        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.mess_dropout = args_config.mess_dropout
        self.mess_dropout_rate = args_config.mess_dropout_rate
        self.edge_dropout = args_config.edge_dropout
        self.edge_dropout_rate = args_config.edge_dropout_rate
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.ns = args_config.ns
        self.K = args_config.K
        self.alpha = args_config.alpha
        self.warmup = args_config.warmup
        self.beta = args_config.beta
        self.eps = args_config.eps
        # gating
        self.gamma = args_config.gamma
        self.choose=args_config.choose
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self.user_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.item_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.mlp_q_layers = nn.Linear(self.n_users, self.emb_size).to(self.device)
        self.mlp_p_layers = nn.Linear(self.n_items, self.emb_size).to(self.device)
        self.pos_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)
        self.neg_gate = nn.Linear(self.emb_size, self.emb_size).to(self.device)

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.bias = nn.Parameter(self.bias)

        self.gcn = self._init_model()
        #这里的self.GCN就是在调用图卷积
    def _init_weight(self):
        #随机初始化U和I的embeding
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))
        self.bias = initializer(torch.empty(1, self.n_users))
        
        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        #这里是调用图卷积的地方
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         interact_mat=self.sparse_norm_adj,
                         edge_dropout_rate=self.edge_dropout_rate,
                         mess_dropout_rate=self.mess_dropout_rate)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)
    

    def forward(self, cur_epoch, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]
        
        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=self.edge_dropout,
                                              mess_dropout=self.mess_dropout)

        if self.ns == 'rns':  # n_negs = 1
            neg_gcn_embs = item_gcn_emb[neg_item[:, :self.K]]
        elif self.ns == 'novel':
            neg_gcn_embs = []
            for k in range(self.K):
                neg_gcn_embs.append(self.phase_negative_sampling(cur_epoch, user_gcn_emb, item_gcn_emb,
                                                                user,
                                                                neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                pos_item))
            neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)
        n_neg_gcn_embs = [] 
        for k in range(self.K): 
            n_neg_gcn_embs.append(self.n_negative_sampling(cur_epoch, user_gcn_emb, item_gcn_emb,
                                                                user,
                                                                neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                pos_item)) 
        n_neg_gcn_embs = torch.stack(n_neg_gcn_embs, dim=1)
        return self.create_bpr_loss(cur_epoch, user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs,n_neg_gcn_embs)
    

    
    def n_negative_sampling(self, cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        gate_n = torch.sigmoid(self.neg_gate(n_e))
        gated_n_e = n_e * gate_n  # [batch_size, n_negs, n_hops+1, channel]

        n_e_sel = (1 - min(1, cur_epoch / self.warmup)) * n_e - gated_n_e  # [batch_size, n_negs, n_hops+1, channel]

        """Add random noise to s_e"""
        random_noise1 = torch.rand_like(s_e).cuda()
        normalized_noise1 = F.normalize(random_noise1, dim=-1)  # Normalize random noise
        s_e1 = s_e + torch.sign(s_e) * normalized_noise1 * self.eps

        random_noise2 = torch.rand_like(s_e).cuda()
        normalized_noise2 = F.normalize(random_noise2, dim=-1)  # Normalize random noise
        s_e2 = s_e - torch.sign(s_e) * normalized_noise2 * self.eps

        """Dynamic negative sampling"""
        scores1 = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        scores2 = (s_e1.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        scores3 = (s_e2.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]

        scores_avg = (scores1 + scores2 + scores3) / 3.0  # Average scores
        indices = torch.min(scores_avg, dim=1)[1].detach()
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]

        return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]
    
    def phase_negative_sampling(self, cur_epoch, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        import torch.nn.functional as F

        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]

        
        gate_n = torch.sigmoid(self.neg_gate(n_e) )
        gated_n_e = n_e * gate_n  # [batch_size, n_negs, n_hops+1, channel]

        n_e_sel = (1 - min(0.6, cur_epoch / self.warmup)) * n_e - min(0.6, cur_epoch / self.warmup)*gated_n_e  # [batch_size, n_negs, n_hops+1, channel]

        """Add random noise to s_e"""
        random_noise1 = torch.rand_like(s_e).cuda()
        normalized_noise1 = F.normalize(random_noise1, dim=-1)  # Normalize random noise
        s_e1 = s_e + torch.sign(s_e) * normalized_noise1 * 0.005

        random_noise2 = torch.rand_like(s_e).cuda()
        normalized_noise2 = F.normalize(random_noise2, dim=-1)  # Normalize random noise
        s_e2 = s_e - torch.sign(s_e) * normalized_noise2 * 0.005
        
        
        
        
        """Dynamic negative sampling"""
        scores1 = (s_e.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        scores2 = (s_e1.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        scores3 = (s_e2.unsqueeze(dim=1) * n_e_sel).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]

        scores_avg = (scores1 + scores2 + scores3)/3.0   # Average scores
        indices = torch.max(scores_avg, dim=1)[1].detach()
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]

        return neg_items_emb_[[[i] for i in range(batch_size)], range(neg_items_emb_.shape[1]), indices, :]

    
    

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed,
                                              self.item_embed,
                                              edge_dropout=False,
                                              mess_dropout=False)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())
#也有可能这里的rating'是固定的,不是固定的每次都会再计算，固定的原因是设定了种子
    def create_bpr_loss(self, cur_epoch, user_gcn_emb, pos_gcn_embs, p_neg_gcn_embs, n_neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        #neg_e 的维度是 [batch_size, K, channel] 
        neg_e = self.pooling(p_neg_gcn_embs.view(-1, p_neg_gcn_embs.shape[2], p_neg_gcn_embs.shape[3])).view(batch_size,
                                                                                                       self.K, -1)

        
       
        n_e = p_neg_gcn_embs * (1-min(0.9,cur_epoch/(self.warmup+50))) if (cur_epoch / self.warmup) > self.beta else p_neg_gcn_embs
        hard_gcn_embs=pos_gcn_embs.unsqueeze(dim=1)*min(0.9,max(0,cur_epoch/(self.warmup+50)))+n_e
        
        
        hard_neg_e = self.pooling(hard_gcn_embs.view(-1, hard_gcn_embs.shape[2], hard_gcn_embs.shape[3])).view(batch_size,
                                                                                                       self.K, -1)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]
        hard_neg_scores=torch.sum(torch.mul(u_e.unsqueeze(dim=1), hard_neg_e), axis=-1)   
        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=1)
        neg_scores = torch.sum(torch.mul(u_e.unsqueeze(dim=1), neg_e), axis=-1)  # [batch_size, K]
        if self.choose==1:
            mf_loss = max(-0.25,1.5-1.0*cur_epoch/self.warmup)*(torch.mean(torch.log(1 + torch.exp(hard_neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1))))
        elif  self.choose==2:
            mf_loss = max(-0.25,1.5-1.0*cur_epoch/100)*(torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1))))
        elif  self.choose==3:
            mf_loss = (torch.mean(torch.log(1 + torch.exp(hard_neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1))))
        
        if self.ns == 'novel' and self.gamma > 0.:
            
            
            

            gate_neg = torch.sigmoid(self.neg_gate(p_neg_gcn_embs))
            gated_neg_e_r = p_neg_gcn_embs * gate_neg
            gated_neg_e_ir = p_neg_gcn_embs - gated_neg_e_r
            
            
            
            # gated_hard_e_r = self.pooling(gated_hard_e_r.view(-1, hard_gcn_embs.shape[2], hard_gcn_embs.shape[3])).view(batch_size, self.K, -1)
            # gated_hard_e_ir = self.pooling(gated_hard_e_ir.view(-1, hard_gcn_embs.shape[2], hard_gcn_embs.shape[3])).view(batch_size, self.K, -1)

            
            
            gated_neg_e_r = self.pooling(gated_neg_e_r.view(-1, n_neg_gcn_embs.shape[2], n_neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)
            
            
            gated_neg_e_ir = self.pooling(gated_neg_e_ir.view(-1, n_neg_gcn_embs.shape[2], n_neg_gcn_embs.shape[3])).view(batch_size, self.K, -1)

            
            gated_neg_scores_r = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_r), axis=-1)  # [batch_size, K]
            
            
            gated_neg_scores_ir = torch.sum(torch.mul(u_e.unsqueeze(dim=1), gated_neg_e_ir), axis=-1)  # [batch_size, K]
            
        
            
           
            # BPR
            # mf_loss += self.gamma * torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_n_neg_scores_r.unsqueeze(dim=1)).sum(dim=1)))
            
            mf_loss += self.gamma * (torch.mean(torch.log(1 + torch.exp(gated_neg_scores_r - gated_neg_scores_ir).sum(dim=1))))#+torch.mean(torch.log(1 + torch.exp(score2 - score1).sum(dim=1)))
            
            
            
            
            

        # cul regularizer
        regularize = ((torch.norm(user_gcn_emb[:, 0, :]) ** 2)
                      + (torch.norm(pos_gcn_embs[:, 0, :]) ** 2)
                      + (torch.norm(p_neg_gcn_embs[:, :, 0, :]) ** 2)+(torch.norm(n_neg_gcn_embs[:, :, 0, :]) ** 2)) / 2  # take hop=0,目前来看“-”是效果最好的改动,这个改动目前看来只能说明，这个地方是加或者减号都没用任何影响
        emb_loss =self.decay * regularize / batch_size #+cur_epoch*torch.mm(self.bias,self.bias.T)#这里的loss和regularize有改进空间
        
        return mf_loss + emb_loss, mf_loss, emb_loss
