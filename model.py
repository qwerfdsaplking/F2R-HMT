# -*- coding: utf-8 -*-
import torch
import math
from torch.nn import init
import json
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm




def sparse_id2multihot(x):
    #split the x to dense_x and sparse_x
    d_ds = 100
    d_sp=int((x.shape[-1]-d_ds)/2)
    bound = 8790


    x_ds = x[:, :d_ds].float()
    x_sp_id = x[:, d_ds:d_ds+d_sp].long()
    x_sp_val = x[:, d_ds+d_sp:].float()

    filter_mat = x_sp_id >= bound
    x_sp_id = x_sp_id.masked_fill(filter_mat, 0)
    x_sp_val = x_sp_val.masked_fill(filter_mat, 0)

    x_sp_mat = torch.zeros(x.shape[0] * bound, device=x_ds.device)
    x_sp_id = x_sp_id + (torch.arange(x.shape[0], dtype=torch.long, device=x_ds.device) * bound).reshape(x.shape[0], 1)
    x_sp_id = x_sp_id.reshape(x.shape[0] * d_sp)
    x_sp_val = x_sp_val.reshape(x.shape[0] * d_sp)
    x_sp_mat[x_sp_id] = x_sp_val
    x_sp_mat = x_sp_mat.reshape(x.shape[0], bound)
    x_sp = x_sp_mat[:,d_ds:bound]

    return x_ds,x_sp




def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, input_size, hidden_size, activation="PReLU", dropout=0.1, out_size=None):
        """Initialization.

        :param input_size: the input dimension.
        :param hidden_size: the hidden dimension.
        :param activation: the activation function.
        :param dropout: the dropout rate.
        :param out_size: the output dimension, the default value is equal to input_size.
        """
        super(PositionwiseFeedForward, self).__init__()
        if out_size is None:
            out_size = input_size
        # By default, bias is on.
        self.W_1 = nn.Linear(input_size, hidden_size)
        self.W_2 = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.act_func = get_activation_function(activation)

    def forward(self, x):
        """
        The forward function
        :param x: input tensor.
        :return:
        """
        return self.W_2(self.dropout(self.act_func(self.W_1(x))))



class ScaledDotProductAttention(nn.Module):
    """
    Compute 'Scaled Dot Product SelfAttention
    """
    def forward(self, query, key, value, mask=None, dropout=None):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :param dropout:
        :return:
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))
        #print(scores.shape,mask.shape)
        if mask is not None:
            if scores.shape==mask.shape:
                scores = scores * mask
                scores = scores.masked_fill(scores == 0, -1e9)
            else:
                scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, input_size, dropout=0.1,use_ffn=False):
        """

        :param h:
        :param input_size:
        :param dropout:
        :param bias:
        """
        super().__init__()
        assert input_size % head_num == 0

        # We assume d_v always equals d_k
        self.d_k = input_size // head_num
        self.head_num = head_num  # number of heads

        self.q_linear = nn.Linear(input_size, input_size)
        self.k_linear = nn.Linear(input_size, input_size)
        self.v_linear = nn.Linear(input_size, input_size)
        self.a_linear = nn.Linear(input_size, input_size)
        self.layer_norm = nn.LayerNorm(input_size,eps=1e-6)
        #self.output_linear = nn.Linear(input_size, input_size, bias)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)

        self.use_norm = True
        self.use_ffn=use_ffn
        if use_ffn:
            self.pos_ffn_layer = PositionwiseFeedForward(input_size, input_size*2, dropout=dropout)

    def forward(self, x, mask):
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        batch_size = x.size(0)
        residual = x

        query = self.q_linear(x)
        key = self.k_linear(x)
        value = self.v_linear(x)

        query = query.view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)

        if mask is not None and len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        # 2) Apply attention on all the projected vectors in batch.
        x, att = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.d_k)

        #residual connection
        x = self.dropout(self.a_linear(x))
        x += residual

        x = self.layer_norm(x)

        if self.use_ffn:
            x = self.pos_ffn_layer(x)

        #return self.output_linear(x)
        return x, att

class TransformerEncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, input_size,hidden_size, head_num, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.attn_layer = MultiHeadAttention(head_num, input_size, dropout)
        self.pos_ffn_layer = PositionwiseFeedForward(input_size, hidden_size, dropout=dropout)

    def forward(self, inputs, attn_masks=None):
        outputs, attns = self.slf_attn_layer(
            inputs, inputs, inputs, mask=attn_masks)
        outputs = self.pos_ffn_layer(outputs)
        return outputs, attns





class HgsMultiHeadAttention(nn.Module):
    """
    The heterogeneous multi-head attention module. Take in model size and number of heads.
    """
    def __init__(self, head_num, input_size, node_type_num, dropout=0.1, use_ffn=False):
        """

        :param h:
        :param input_size:
        :param dropout:
        :param bias:
        """
        super().__init__()
        assert input_size % head_num == 0

        # We assume d_v always equals d_k
        self.d_k = input_size // head_num
        self.head_num = head_num  # number of heads
        self.node_type_num = node_type_num
        self.q_linears = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.node_type_num)])
        self.k_linears = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.node_type_num)])
        self.v_linears = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.node_type_num)])
        self.a_linears = nn.ModuleList([nn.Linear(input_size, input_size) for _ in range(self.node_type_num)])
        self.norms = nn.ModuleList([nn.LayerNorm(input_size) for _ in range(self.node_type_num)])
        #self.output_linear = nn.Linear(input_size, input_size, bias)
        self.attention = ScaledDotProductAttention()

        self.dropout = nn.Dropout(p=dropout)

        self.skip = nn.Parameter(torch.ones(self.node_type_num))
        self.use_norm = True
        self.use_ffn=use_ffn
        #self.posfeedforward = PositionwiseFeedForward(input_size, input_size, activation="PReLU", dropout=0.1, out_size=None)
        if use_ffn:
            self.pos_ffn_layer = PositionwiseFeedForward(input_size, input_size*2, dropout=dropout)


    def forward(self, x, x_type, mask):
        """

        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """

        residual = x.clone()
        batch_size = x.size(0)
        x = x.reshape(-1,x.shape[-1])


        query = torch.zeros(x.shape).to(x.device)#.reshape(-1,x.shape[-1]).to(x.device)
        key = torch.zeros(x.shape).to(x.device)#.reshape(-1,x.shape[-1]).to(x.device)
        value = torch.zeros(x.shape).to(x.device)#.reshape(-1,x.shape[-1]).to(x.device)
        x_type = x_type.reshape(-1)

        #print(x_type.shape)
        #print(x_type)
        #print(x)
        for i in range(self.node_type_num):
            idx = (x_type==i).nonzero().squeeze()

            if len(idx.shape)>0 and idx.shape[0]>0:
                query[idx] = self.q_linears[i](x[idx])
                key[idx] = self.k_linears[i](x[idx])
                value[idx] = self.v_linears[i](x[idx])




        query = query.view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.head_num, self.d_k).transpose(1, 2)

        if mask is not None and len(mask.shape)==3:
            mask = mask.unsqueeze(1)
        # 2) Apply attention on all the projected vectors in batch.
        x, _ = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(-1, self.head_num * self.d_k)#.view(batch_size, -1, self.head_num * self.d_k)

        residual = residual.view(-1, self.head_num * self.d_k)
        #residual connection

        out = torch.zeros(x.shape).to(x.device)

        for i in range(self.node_type_num):
            idx = (x_type == i).nonzero().squeeze()
            x_i = self.a_linears[i](x[idx])
            x_i = self.dropout(x_i)

            #alpha = torch.sigmoid(self.skip[i])
            if self.use_norm:
                #x[idx] = self.norms[i](alpha * x[idx] + (1 - alpha) * residual[idx])
                out[idx] = self.norms[i](x_i+residual[idx])
            else:
                #x[idx] = alpha * x[idx] + (1 - alpha) * residual[idx]
                out[idx] = x_i+residual[idx]

        #return self.output_linear(x)
        x = x.view(batch_size, -1, self.head_num * self.d_k)

        #ffn
        if self.use_ffn:
            x = self.pos_ffn_layer(x)

        return x, _


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)



class Transformer_Model(torch.nn.Module):
    def __init__(self, args,
                 trace_func=print, rank=0):
        super(Transformer_Model, self).__init__()
        self.hidden_size = args.hidden_size
        self.out_size = args.out_size
        self.node_type_num = args.node_type_num  #
        self.gnn_type = args.gnn_type
        self.gnn_num = args.gnn_num
        self.dense_size = 224 if 'dense' in args.feature_mode else 0
        self.sparse_size = 8690 if 'sparse' in args.feature_mode else 0
        self.n_heads = 4
        self.args = args
        self.dense_offset_list = [0, 97, 112, 125, 224]
        self.pos_size = 3 if args.pos_encoding else 0
        self.max_len = args.max_node_num
        self.export = False
        self.get_att = False
        if rank == 0:
            trace_func('++++++ONNX++++Model %s+++++%s++++' % (
            args.gnn_type, self.dense_size + self.sparse_size+self.pos_size))
            if self.args.pos_encoding:
                trace_func('use pos encoding')
                trace_func(args.head_masks)


        # feature mapping layer
        self.input_linear = nn.Linear(self.dense_size + self.sparse_size + self.pos_size, self.hidden_size)

        self.drop = nn.Dropout(p=args.dropout_rate)  #
        self.relu = nn.PReLU()
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=1e-6)

        self.sigmoid = nn.Sigmoid()
        if args.gnn_type in ['transformer','transformermasked']:
            self.att_layers = nn.ModuleList([
                MultiHeadAttention(self.n_heads, self.hidden_size, dropout=0.1,use_ffn=args.use_ffn) for _ in range(args.gnn_num)])
        elif args.gnn_type in ['hgstransformer','hgstransformermasked']:
            self.att_layers = nn.ModuleList([
                HgsMultiHeadAttention(self.n_heads, self.hidden_size, self.node_type_num,dropout=0.1,use_ffn=args.use_ffn) for _ in range(args.gnn_num)])


        if args.pool_type=='split':
            out_mid_size = args.hidden_size*4
            self.mid_linear = nn.Linear(out_mid_size, args.hidden_size)  # 
            self.out_linear = nn.Linear(args.hidden_size, args.out_size)
        elif args.pool_type=='all':
            out_mid_size = args.hidden_size*3
            self.mid_linear = nn.Linear(out_mid_size, args.hidden_size)  # 
            self.out_linear = nn.Linear(args.hidden_size, args.out_size)
        elif args.pool_type=='cross':
            self.bi_mat_list = nn.ParameterList([nn.Parameter(torch.randn(args.hidden_size,args.hidden_size)) for _ in range(3)])
            for i in range(len(self.bi_mat_list)):#
                glorot(self.bi_mat_list[i])
        elif args.pool_type=='att':
            self.v_att_bi = nn.Parameter(torch.randn(args.hidden_size,args.hidden_size))
            self.u_att_bi = nn.Parameter(torch.randn(args.hidden_size, args.hidden_size))
            self.out_bi = nn.Parameter(torch.randn(args.hidden_size, args.hidden_size))
            glorot(self.v_att_bi)
            glorot(self.u_att_bi)
            glorot(self.out_bi)
        else:
            raise ValueError('Pool type Error!')





    def forward(self, x,adj,x_mask, x_type,x_in_vsg,x_in_usg, isout):
        x =x.float()
        batch_size = x.shape[0]
        x_type = x_type.long()
        x_mask = x_mask.float()
        x_type = x_type.long()
        x_in_usg=x_in_usg.bool()
        x_in_vsg=x_in_vsg.bool()
        isout=isout.bool()
        mask = torch.matmul(x_mask.unsqueeze(-1), x_mask.unsqueeze(1)).bool()
        #adj add_self_loop and bidirected edges
        adj += torch.eye(adj.shape[-1]).to(adj.device)
        adj += adj.transpose(1,2)
        adj *= mask
        adj = adj.bool()


        #cross attention mask
        cross_mask = x_in_vsg.unsqueeze(-1).float()@x_in_usg.unsqueeze(1).float()
        cross_mask = cross_mask+cross_mask.transpose(1,2)




        #===========================feature processing=================================
        x = x.reshape(batch_size*self.max_len,x.shape[-1])
        x_type = x_type.reshape(batch_size*self.max_len)
        x_ds= torch.zeros(x.shape[0],224).to(x.device)
        for i in range(self.node_type_num):
            s=self.dense_offset_list[i]
            e=self.dense_offset_list[i+1]
            x_ds[(x_type == i).nonzero().squeeze(), s:e] = x[(x_type == i).nonzero().squeeze(), :e - s]

        if self.args.feature_mode=='dense':
            x = x_ds
        elif self.args.feature_mode=='sparse':
            _, x_sp = sparse_id2multihot(x)
            x =x_sp
        elif self.args.feature_mode=='dense+sparse':
            _, x_sp_all = sparse_id2multihot(x)
            x_sp = x_sp_all[:,:1690]
            x = torch.cat([x_ds, x_sp], dim=1)

        x = x.reshape(batch_size, self.max_len, x.shape[-1])
        # ============================graph position embedding===============================
        if self.args.pos_encoding:
            x = torch.cat([x, x_in_usg.unsqueeze(-1).float(), x_in_vsg.unsqueeze(-1).float(),isout.unsqueeze(-1).float()],dim=-1)


        # ===========================feature mapping start=================================
        x = self.input_linear(x)


        #===========================sparse adjacent matrix=============================
        if 'masked' in self.args.gnn_type:
            x_norm = torch.norm(x_sp_all,dim=-1,p=2)
            x_sp_all = x_sp_all.reshape(batch_size, self.max_len, -1)
            sparse_adj = torch.matmul(x_sp_all, x_sp_all.transpose(1,2))/(torch.matmul(x_norm.unsqueeze(-1),x_norm.unsqueeze(-1).transpose(1,2)))
            #sparse_adj += torch.eye(sparse_adj.shape[-1]).to(sparse_adj.device)
        else:
            sparse_adj=0



        x = self.drop(x)
        x = self.layer_norm(x)


        mask_dict = {'adj': adj.float(),
                     'sparse': sparse_adj,
                     'full': mask.float(),
                     'cross': cross_mask.float()}
        if 'center' in self.args.head_masks:
            isout_mask = isout.repeat(1, isout.shape[-1]).reshape(adj.shape)
            isout_mask = isout_mask + isout_mask.permute(0, 2, 1)
            center = (adj * isout_mask).bool()
            mask_dict['center']=center

        for layer in self.att_layers:
            if self.args.gnn_type == 'transformer':
                mask = (adj.float() + mask.float()).bool().float()
                masks = torch.cat([mask.float().unsqueeze(1) for _ in range(4)],dim=1)
                x,att = layer(x,masks)
            elif self.args.gnn_type == 'hgstransformer':
                x,att = layer(x,x_type, mask)
            elif self.args.gnn_type == 'hgstransformermasked':
                head_masks = self.args.head_masks.split('+')
                masks = torch.cat([mask_dict[name].unsqueeze(1) for name in head_masks],dim=1)
                x, att = layer(x,x_type,masks)
            elif self.args.gnn_type == 'hgstransformermaskedv2':
                head_masks = self.args.head_masks.split('+')
                masks = torch.cat([mask_dict[name].unsqueeze(1) for name in head_masks],dim=1)
                x,att = layer(x,x_type,masks)
            elif self.args.gnn_type == 'transformermasked':
                head_masks = self.args.head_masks.split('+')
                masks = torch.cat([mask_dict[name].unsqueeze(1) for name in head_masks],dim=1)
                x, att = layer(x, masks)
            else:
                raise ValueError('gnn type error')


        x = x.reshape(-1, x.shape[-1])
        isout = isout.reshape(-1)
        vrid = (isout & (x_type == 1)).nonzero().squeeze()
        vx_roots = x[vrid].reshape(-1, x.shape[-1])
        urid = (isout & (x_type == 0)).nonzero().squeeze()
        ux_roots = x[urid].reshape(-1, x.shape[-1])


        if self.args.pool_type in ['split','all']:
            x = x.reshape(batch_size, -1, x.shape[-1])
            if self.args.pool_type=='split':
                v_mean_pool = (x * x_in_vsg.float().unsqueeze(-1)).sum(1)/x_in_vsg.float().sum(1).unsqueeze(-1)
                u_mean_pool = (x * x_in_usg.float().unsqueeze(-1)).sum(1) / x_in_usg.float().sum(1).unsqueeze(-1)
                x_meanpool = torch.cat([v_mean_pool,u_mean_pool],dim=1)
            else:#一起pooling
                x_meanpool = x.sum(1) / x_mask.float().sum(-1).unsqueeze(1)

            #print(vx_roots.shape,ux_roots.shape,x_meanpool.shape)
            x_out = torch.cat([vx_roots, ux_roots, x_meanpool], dim=1)
            #print(vx_roots.shape, x_meanpool.shape,x_out.shape)
            x_out = self.mid_linear(x_out)
            x_out = self.drop(self.relu(x_out))
            x_out = self.out_linear(x_out)

        elif self.args.pool_type == 'cross':#cross&inner attention
            x = x.reshape(batch_size,-1,x.shape[-1])
            v_v_pool, v_u_pool, u_v_pool, u_u_pool = self.cross_pooling(x, vx_roots, ux_roots, x_in_vsg, x_in_usg)
            score_1 = (v_v_pool.unsqueeze(1)@self.bi_mat_list[0]@u_u_pool.unsqueeze(2)).squeeze(-1)
            score_2 = (v_u_pool.unsqueeze(1) @ self.bi_mat_list[1] @ v_v_pool.unsqueeze(2)).squeeze(-1)
            score_3 = (u_v_pool.unsqueeze(1) @ self.bi_mat_list[2] @ u_u_pool.unsqueeze(2)).squeeze(-1)
            #score_4 = (u_v_pool.unsqueeze(1) @ self.bi_mat_list[3] @ v_u_pool.unsqueeze(2)).squeeze(-1)
            x_out = score_1+score_2+score_3
        elif self.args.pool_type == 'att':
            x = x.reshape(batch_size,-1,x.shape[-1])
            v_v_pool, u_u_pool = self.attentive_pooling(x, vx_roots, ux_roots, x_in_vsg, x_in_usg)
            x_out = (v_v_pool.unsqueeze(1) @ self.out_bi @ u_u_pool.unsqueeze(2)).squeeze(-1)
        else:
            raise ValueError('pool type error')

        if self.export:
            x_out = self.sigmoid(x_out)#+adj.float()[0][0][0]*0.0
        if self.get_att:
            return x_out, att

        return x_out



    def attentive_pooling(self, x, vx_roots, ux_roots, x_in_vsg, x_in_usg):
        v_score = (x@self.v_att_bi@vx_roots.unsqueeze(-1)).squeeze()
        u_score = (x @ self.u_att_bi @ ux_roots.unsqueeze(-1)).squeeze()

        v_v_score = torch.masked_fill(v_score, ~x_in_vsg, -1e12)
        u_u_score = torch.masked_fill(u_score, ~x_in_usg, -1e12)
        v_v_att = F.softmax(v_v_score, dim=-1)  # nxl
        u_u_att = F.softmax(u_u_score, dim=-1)
        v_v_pool = (x * v_v_att.unsqueeze(-1)).sum(1)
        u_u_pool = (x * u_u_att.unsqueeze(-1)).sum(1)
        return v_v_pool, u_u_pool

    def cross_pooling(self, x, vx_roots, ux_roots, x_in_vsg, x_in_usg):
        #x nxlxd

        v_score = torch.matmul(x, vx_roots.unsqueeze(-1)).squeeze()#nxl
        u_score = torch.matmul(x, ux_roots.unsqueeze(-1)).squeeze()#nxl

        v_v_score = torch.masked_fill(v_score, ~x_in_vsg, -1e12)
        v_u_score = torch.masked_fill(v_score, ~x_in_usg, -1e12)
        u_v_score = torch.masked_fill(u_score, ~x_in_vsg, -1e12)
        u_u_score = torch.masked_fill(u_score, ~x_in_usg, -1e12)

        v_v_att = F.softmax(v_v_score, dim=-1)#nxl
        v_u_att = F.softmax(v_u_score, dim=-1)
        u_v_att = F.softmax(u_v_score, dim=-1)
        u_u_att = F.softmax(u_u_score, dim=-1)

        v_v_pool = (x * v_v_att.unsqueeze(-1)).sum(1)
        v_u_pool = (x * v_u_att.unsqueeze(-1)).sum(1)
        u_v_pool = (x * u_v_att.unsqueeze(-1)).sum(1)
        u_u_pool = (x * u_u_att.unsqueeze(-1)).sum(1)
        return v_v_pool, v_u_pool, u_v_pool, u_u_pool



