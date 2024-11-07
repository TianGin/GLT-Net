import os
import pickle
import time
import random
import shutil
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
import  torch
import  torch.nn as nn
from datetime import datetime
import pandas as pd
import math
import torch.nn.init as init
seed = 6669
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(42)
torch.set_printoptions(precision=8)
def f1_(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    f1 = f1_score(y_true=y_true_hot, y_pred=result, average=metrics, zero_division=1)
    return (f1)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks), ))
    r = np.zeros((len(ks), ))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / min(k, len(t))
    return a / len(y_true_hot), r / len(y_true_hot)
def medical_codes_loss(y_pred, y_true):
    focal_loss = torch.nn.BCEWithLogitsLoss(reduction='none')  
    return torch.mean(torch.sum(focal_loss(y_pred, y_true.float()), axis=-1))

# class CodeEmbedding(nn.Module):
#     def __init__(self, code_num, embedding_size, embedding_init=None):
#         super(CodeEmbedding, self).__init__()
#         if embedding_init is not None:
#             self.code_embedding = nn.Parameter(torch.tensor(embedding_init))
#         else:
#             self.code_embedding = nn.Parameter(torch.Tensor(code_num + 1, embedding_size))
#             nn.init.xavier_uniform_(self.code_embedding)

#     def forward(self, inputs=None):
#         return self.code_embedding

class Embedding(torch.nn.Embedding):

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(Embedding, self).__init__(num_embeddings, embedding_dim, padding_idx=padding_idx,
                                        max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq,
                                        sparse=sparse, _weight=_weight)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)



class CodeEmbedding(nn.Module):
    def __init__(self, code_num, embedding_size, embedding_init=None):
        super(CodeEmbedding, self).__init__()
        if embedding_init is not None:
            # print(torch.tensor(embedding_init).shape)
            self.code_embedding = nn.Parameter(torch.tensor(embedding_init))
        else:
            # self.code_embedding = nn.Parameter(torch.Tensor(4930 + 1, embedding_size))
            # nn.init.xavier_uniform_(self.code_embedding)
            self.pre_embedding = Embedding(code_num + 1, embedding_size)
            indices = list(range(code_num + 1))
            self.bias_embedding = torch.nn.Parameter(torch.Tensor(embedding_size))
            bound = 1 / (code_num + 1)
            init.uniform_(self.bias_embedding, -bound, bound)
            # self.code_embedding = nn.Parameter(self.pre_embedding(torch.tensor(indices)))
            self.code_embedding = self.pre_embedding(torch.tensor(indices)).cuda() + self.bias_embedding.cuda()

    def forward(self, inputs=None):
        return self.code_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, adj, hiddens, dropout_rate=0.):
        super(GraphConvolution, self).__init__()
        self.adj = torch.Tensor(adj).cuda()  # (n, n)
        self.denses = nn.ModuleList([nn.Linear(in_features, out_features) for in_features, out_features in zip(hiddens[:-1], hiddens[1:])])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = x.to(torch.float32)  # (n, dim)
        for dense in self.denses:
            output = self.dropout(output)
            output = torch.matmul(self.adj, output)  # (n, dim)
            output = dense(output)
        return output


def masked_softmax(inputs, mask):
    # 将输入减去最大值以防止指数爆炸
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True)[0]
    exp = torch.exp(inputs) * mask
    # 计算softmax
    # result = exp / torch.sum(exp, dim=-1, keepdim=True)
    # 计算softmax
    sum_exp = torch.sum(exp, dim=-1, keepdim=True)
    result = exp / torch.where(sum_exp == 0, torch.tensor(1.0).cuda(), sum_exp)
    return result

class Attention(nn.Module):
    def __init__(self, attention_size):
        super(Attention, self).__init__()
        self.attention_size = attention_size
        self.b_omega = nn.Parameter(torch.zeros(attention_size))
        self.u_omega = nn.Parameter(torch.zeros(attention_size))
        self.w_omega = nn.Parameter(torch.zeros(256, self.attention_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.b_omega.unsqueeze(0))
        nn.init.xavier_uniform_(self.u_omega.unsqueeze(0))
        nn.init.xavier_uniform_(self.w_omega)

        
    def forward(self, x, mask=None):
        v = torch.tanh(torch.matmul(x, self.w_omega) + self.b_omega)  # (**size, attention_size)
        vu = torch.tensordot(v, self.u_omega, dims=1)  # (**size)
        
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = F.softmax(vu, dim=-1)  # (**size)
        
        output = torch.sum(x * alphas.unsqueeze(-1), dim=-2)  # (**size, dim)
        return output
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):

        super(PositionalEncoding, self).__init__()


        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])

        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.from_numpy(position_encoding.astype(np.float32))

        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))

        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):


        max_len = 49
        # tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor

        pos = np.zeros([len(input_len), max_len])
        for ind, length in enumerate(input_len):
            for pos_ind in range(1, length + 1):
                pos[ind, pos_ind - 1] = pos_ind
        input_pos = torch.LongTensor(pos).cuda()
        return self.position_encoding(input_pos).clone().detach(), input_pos


class PositionalWiseFeedForward(nn.Module):
    def __init__(self, model_dim=412, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Linear(model_dim, ffn_dim, 1)
        self.w2 = nn.Linear(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, x):
        output = x
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(x + output).clone().detach()
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim=412, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim, eps=1e-6)

    def forward(self, key, value, query, attn_mask=None):
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class EncoderLayer(nn.Module):
    def __init__(self, model_dim=412, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


def padding_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    # pad_mask = ~seq_k.eq(0)
    # pad_mask = pad_mask.unsqueeze(1)  # shape [B, L_q, L_k]
    pad_mask = seq_k.eq(0)
    # print(pad_mask)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        attention = torch.bmm(q, k.transpose(1, 2))
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, v)
        return context, attention
    
class EncoderNew(nn.Module):
    def __init__(self,
                 num_layers=1,
                 model_dim=256,
                 num_heads=4,
                 ffn_dim=1024,
                 max_len=49,
                 dropout=0.0):
        super(EncoderNew, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])

        self.pos_embedding = PositionalEncoding(model_dim, max_len)
    def forward(self, code_embeddings, time_embeddings,  mask_code, mask_visit, visits_len):
        output = (code_embeddings * mask_code).sum(dim=2) + time_embeddings
        # print(output)
        # output = (code_embeddings * mask_code).sum(dim=2) + time_embeddings
        output_pos, ind_pos = self.pos_embedding(visits_len.unsqueeze(1))
        output += output_pos
        # print(output)
        self_attention_mask = padding_mask(ind_pos, ind_pos)

        attentions = []
        outputs = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            outputs.append(output)
        # weight = torch.softmax(self.weight_layer(outputs[-1]), dim=1)
        # weight = weight * mask - 255 * (1 - mask)
        return output
    
    
class FFN(nn.Module):
    def __init__(self, hid_units, output_dim):
        super(FFN, self).__init__()
        self.hid_units = hid_units
        self.output_dim = output_dim 
        # 定义模型的参数
        # self.W1 = nn.Parameter(torch.Tensor(1, self.hid_units))
        # self.b1 = nn.Parameter(torch.zeros(self.hid_units))
        # self.W2 = nn.Parameter(torch.Tensor(self.hid_units, self.output_dim))
        # # 初始化模型的参数
        # self.reset_parameters()
        self.self_layer1 = torch.nn.Linear(1, self.hid_units)
        self.self_layer2 = torch.nn.Linear(self.hid_units, self.output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    # def reset_parameters(self):
    #     nn.init.xavier_uniform_(self.W1)
    #     nn.init.xavier_uniform_(self.b1.unsqueeze(0))
    #     nn.init.xavier_uniform_(self.W2)
    
    def forward(self, x):
        # 将输入数据添加一个维度
        # x = x.unsqueeze(-1).to(torch.float32)
        # # 计算第一层
        # x = torch.matmul(torch.tanh(torch.add(torch.matmul(x, self.W1), self.b1)), self.W2)
        # outputs = self.self_layer2(torch.tanh(self.self_layer1(x)))
        seq_time_step = torch.Tensor(x).unsqueeze(2) / 360
        time_feature = 1 - self.tanh(torch.pow(self.self_layer1(seq_time_step), 2))
        outputs = self.self_layer2(time_feature)
        return outputs
    
    
class AttentionVisit(nn.Module):
    def __init__(self, input_dim, attention_size, output_dim):
        super(AttentionVisit, self).__init__()
        self.attention_size = attention_size
        self.u_omega = nn.Parameter(torch.zeros(attention_size))
        self.u_omega_o = nn.Parameter(torch.zeros(attention_size, output_dim))
        self.w_omega = nn.Parameter(torch.Tensor(input_dim, attention_size))
        self.b_omega = nn.Parameter(torch.zeros(attention_size))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.b_omega.unsqueeze(0))
        
        nn.init.xavier_uniform_(self.u_omega.unsqueeze(0))
        # nn.init.xavier_uniform_(self.u_omega_o.unsqueeze(0))
        nn.init.xavier_uniform_(self.w_omega)    
    def forward(self, x, mask=None):
        # print(x.shape, self.w_omega.shape, self.b_omega.shape)
        # print((torch.matmul(x, self.w_omega)).shape)
        # print((torch.matmul(x, self.w_omega)+ self.b_omega).shape)
        t = F.normalize(torch.matmul(x, self.w_omega) + self.b_omega, p=2, dim=-1)
        v = torch.tanh(t)
        # print(v.shape, self.u_omega.shape, self.u_omega_o.shape)
        vu = torch.tensordot(v, self.u_omega, dims=([-1], [0]))
        vu_o = torch.tensordot(v, self.u_omega_o, dims=([-1], [0]))
        
        if mask is not None:
            # print(mask.shape, vu.shape)
            vu *= mask
            mask_o = mask.unsqueeze(-1)
            vu_o *= mask_o
            alphas = masked_softmax(vu, mask)
            betas = masked_softmax(vu_o, mask_o)
        else:
            alphas = F.softmax(vu, dim=-1)
            betas = F.softmax(vu_o, dim=-1)
        
        w = alphas.unsqueeze(-1) * betas
        # print(w.shape, x.shape)
        # print((x * w).shape)
        output = torch.sum(x * w, dim=-2)
        # print(output.shape)
        # print(type(output))
        return output, alphas, betas
    
def sequence_mask(batch_size, sequence_length, max_length, dtype=torch.float32):
    # 创建一个形状为(batch_size, max_length)的张量，所有元素初始化为0
    mask = torch.zeros(sequence_length.size(0), max_length, dtype=dtype)
    
    # 生成一个形状为(batch_size, max_length)的逐渐增加的序列，用于和sequence_length进行比较
    arange_seq = torch.arange(max_length).unsqueeze(0).expand(sequence_length.size(0), -1).cuda()
    
    # 将序列长度作为条件，将对应位置的mask元素设为1
    mask = mask.to(sequence_length.device)
    mask[arange_seq < sequence_length.unsqueeze(1)] = 1
    
    return mask


class TransformerTime(nn.Module):
    def __init__(self, conf, options):
        super(TransformerTime, self).__init__()
        self.max_visit_num = conf['max_visit_num']
        # self.attention_code = Attention(conf['attention_size'])
        # self.feature_encoder = Encoder(num_layers=options['layer'])
        self.feature_encoder = EncoderNew(num_layers=options['layer'], model_dim=options['visit_embedding_size'], max_len = conf['max_visit_num'])
        self.time_encoder = FFN(options['time_hidden'], options['time_embedding_size'])
        
        self.fusion_attention = AttentionVisit(options['visit_embedding_size'], options['attention_size'], options['patients_embedding_size'])
    def forward(self, code_embedding, seq_dignosis_codes, seq_time_step, batch_labels, options, visits_len):
        seq_code_embeddings = code_embedding[seq_dignosis_codes]
        mask_code = (seq_dignosis_codes > 0).to(seq_code_embeddings.dtype)
        mask_code_expanded = mask_code.unsqueeze(-1)
        mask_visit = sequence_mask(options['batch_size'], visits_len, self.max_visit_num, dtype=torch.float32)
        # mask_visit = (torch.arange(self.max_visit_num).expand(len(visits_len), self.max_visit_num) < visits_len.unsqueeze(1)).to(torch.float32)
        # mask_final = (torch.arange(self.max_visit_num).expand(len(visits_len), self.max_visit_num) == (visits_len-1).unsqueeze(1)).to(torch.float32).cuda()
        # masked_embeddings = seq_code_embeddings * mask_code_expanded # 32, 49, 256
        # visit_embeddings = self.attention_code(masked_embeddings, mask_code)  # x: (batch_size, max_seq_len, code_dim)
        time_embeddings = self.time_encoder(seq_time_step)
        # print('time_embeddings', time_embeddings)
        # features = self.feature_encoder(visit_embeddings, mask_visit, time_embeddings, visits_len)
        features = self.feature_encoder(seq_code_embeddings, time_embeddings, mask_code_expanded, mask_visit, visits_len)
        output, alphas, betas = self.fusion_attention(features, mask_visit)
        
        return output
    
class Decoder(nn.Module):
    def __init__(self, dim, output_dim, dropout_rate=0.):
        super(Decoder, self).__init__()
        self.dense = nn.Linear(192, output_dim) #dim
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, patient_embedding, patients_emb):
        result = torch.cat((patient_embedding, patients_emb), dim=1)
        # patient_embedding = self.dropout(patient_embedding)
        # output = self.dense(patient_embedding)
        result = self.dropout(result)
        output = self.dense(result)
        return output
    
class SatNet(nn.Module):
    def __init__(self, conf, hyper_params):
        super(SatNet, self).__init__()
        self.code_embedding = CodeEmbedding(code_num = conf['code_num_pretrain'], 
                                            embedding_size=hyper_params['code_embedding_size'],
                                            embedding_init=conf['code_embedding_init'])
        self.graph_convolution = GraphConvolution(adj=conf['adj'], hiddens=hyper_params['hiddens'],
                                                  dropout_rate=hyper_params['gnn_dropout_rate'])
        self.transformertime = TransformerTime(conf, hyper_params)
        self.decoder = Decoder(hyper_params['patients_embedding_size'], conf['output_dim'], hyper_params['decoder_dropout_rate'])
        
    def forward(self, seq_dignosis_codes, seq_time_step, batch_labels, options, visits_len, patients_emb):
        embeddings = self.code_embedding(None)
        embeddings = self.graph_convolution(embeddings)
        # print(embeddings)
        patient_embedding = self.transformertime(embeddings, seq_dignosis_codes, seq_time_step, batch_labels, options, visits_len)
        # print(patient_embedding)
        outputs = self.decoder(patient_embedding, patients_emb)
        # print(outputs)
        return outputs, batch_labels
    
def load_data(dataset_path):
    code_maps = pickle.load(open(os.path.join(dataset_path, '_code2idx.pickle'), 'rb'))
    train_codes_data = pickle.load(open(os.path.join(dataset_path, '_training-27.pickle'), 'rb'))
    test_codes_data = pickle.load(open(os.path.join(dataset_path, '_testing-27.pickle'), 'rb'))
    valid_codes_data = pickle.load(open(os.path.join(dataset_path, '_validation-27.pickle'), 'rb'))
    
    
    adj = pickle.load(open(os.path.join(dataset_path, '_code_code_adj.pickle'), 'rb'))
    embedding = pickle.load(open(os.path.join(dataset_path, 'leaf_embeddings_'), 'rb'))
    return code_maps, train_codes_data, test_codes_data, valid_codes_data, adj, embedding

def calculate_cost_tran(model, data, options, max_len, device, name, loss_function=None):
    batch_size = options['batch_size']
    n_batches = len(data)
    cost_sum = 0.0
    y_true = []
    y_pred = []
    for batch_diagnosis_codes, batch_labels, batch_time_step, batch_visits_len, batch_pe in tqdm(data, desc=f"Eval",
                                                                                       unit="batch", ncols=80):
        src, tgt, invls, visits_len, p_emb = batch_diagnosis_codes.to(device), batch_labels.to(device), batch_time_step.to(
            device), batch_visits_len.to(device), batch_pe.to(device)
        logit, labels = model(src, invls, tgt, options, visits_len, p_emb)
        loss = loss_function(logit, labels)
        cost_sum += loss.cpu().data.numpy()
        labels = labels.data.cpu().numpy()
        y_true.append(labels)
        prediction = torch.argsort(logit, dim=-1, descending=True).data.cpu().numpy()
        y_pred.append(prediction)
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    f1 = f1_(y_true, y_pred)
    prec, recall = top_k_prec_recall(y_true, y_pred, ks=[5, 10, 15, 20])
    print(f"Data: {name}, f1_score: {f1:.4f}, Recall: {recall}")
    return cost_sum / n_batches

def train_model(conf, options, model_file, model_name, output_file):

    n_epoch = hyper_params['epoch']
    print('building the model ...')
    model = model_file(conf, options)

    print('constructing the optimizer ...')
    optimizer = torch.optim.Adam(model.parameters(), lr=options['lr'], weight_decay=options['L2_reg'])
    print('done!')

    print('loading data ...')
    
    train_dataset = TensorDataset(torch.tensor(train_codes_x), torch.tensor(train_codes_y), torch.tensor(train_codes_intervals),
                                  torch.tensor(train_visit_lens), torch.tensor(train_embedding)
    )
    valid_dataset = TensorDataset(torch.tensor(valid_codes_x), torch.tensor(valid_codes_y), torch.tensor(valid_codes_intervals),
                                  torch.tensor(valid_visit_lens), torch.tensor(valid_embedding)
    )
    test_dataset = TensorDataset(torch.tensor(test_codes_x), torch.tensor(test_codes_y), torch.tensor(test_codes_intervals),
                                  torch.tensor(test_visit_lens), torch.tensor(test_embedding)
    )
    train_loader = DataLoader(train_dataset, batch_size=options['batch_size'], shuffle=False, num_workers=8,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=options['batch_size'], shuffle=False, num_workers=8,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=options['batch_size'], shuffle=False, num_workers=8,
                             pin_memory=True)

    print('training start')
    best_train_cost = 0.0
    best_validate_cost = 1000000000.000000
    best_test_cost = 0.0
    epoch_duaration = 0.0
    best_epoch = 0.0
    max_len = conf['max_visit_num']
    best_parameters_file = ''
    # 将模型移至 GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device) 
    model.train()
    for epoch in range(options['epoch']):
        cost_vector = []
        start_time = time.time()
        model.train()
        for batch_diagnosis_codes, batch_labels, batch_time_step, batch_visits_len, batch_pe in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epoch}", unit="batch", ncols=80):
            src, tgt, invls, visits_len, p_emb = batch_diagnosis_codes.to(device), batch_labels.to(device), batch_time_step.to(device), batch_visits_len.to(device), batch_pe.to(device)
            predictions, labels = model(src, invls, tgt, options, visits_len, p_emb)
            # print(predictions.shape, labels.shape)
            
            optimizer.zero_grad()
            loss = medical_codes_loss(predictions, labels)
            loss.backward()
            optimizer.step()
            cost_vector.append(loss.cpu().data.numpy())
        duration = time.time() - start_time
        print('epoch:%d, mean_cost:%f, duration:%f' % (epoch + 1, np.mean(cost_vector), duration))

        train_cost = np.mean(cost_vector)
        model.eval()
        validate_cost = calculate_cost_tran(model, valid_loader, options, max_len, device, 'valid', medical_codes_loss)
        test_cost = calculate_cost_tran(model, test_loader, options, max_len, device, 'test', medical_codes_loss)
        print('epoch:%d, validate_cost:%f, duration:%f' % (epoch + 1, validate_cost, duration))
        epoch_duaration += duration

        train_cost = np.mean(cost_vector)
        epoch_duaration += duration

        if validate_cost < best_validate_cost:
            best_validate_cost = validate_cost
            best_train_cost = train_cost
            best_test_cost = test_cost
            best_epoch = epoch + 1

            output_file_ = output_file + model_name + '/'
            if os.path.isdir(output_file_):
                shutil.rmtree(output_file_)
            os.mkdir(output_file_)

            torch.save(model.state_dict(), output_file_ + model_name + '.' + str(epoch+1))
            best_parameters_file = output_file_ + model_name + '.' + str(epoch+1)
        buf = 'Best Epoch:%d, Train_Cost:%f, Valid_Cost:%f, Test_Cost:%f' % (
        best_epoch, best_train_cost, best_validate_cost, best_test_cost)
        print(buf)
    # testing
    # best_parameters_file = output_file + model_name + '.19'
    #best_parameters_file = output_file + model_name + '.' + str(8)
    model.load_state_dict(torch.load(best_parameters_file))
    model.eval()
    n_batches = len(test_loader)
    y_true = []
    y_pred = []
    for batch_diagnosis_codes, batch_labels, batch_time_step, batch_visits_len, batch_pe in tqdm(test_loader, desc=f"Eval",
                                                                                       unit="batch", ncols=80):
        src, tgt, invls, visits_len, p_emb = batch_diagnosis_codes.to(device), batch_labels.to(device), batch_time_step.to(
            device), batch_visits_len.to(device), batch_pe.to(device)
        logit, labels = model(src, invls, tgt, options, visits_len, p_emb)
        labels = labels.data.cpu().numpy()
        y_true.append(labels)
        prediction = torch.argsort(logit, dim=-1, descending=True).data.cpu().numpy()
        y_pred.append(prediction)
    y_pred = np.vstack(y_pred)
    y_true = np.vstack(y_true)
    f1 = f1_(y_true, y_pred)
    prec, recall = top_k_prec_recall(y_true, y_pred, ks=[5, 10, 15, 16, 20])
    print(f"f1_score: {f1:.4f}, Recall: {recall}")
    return f1, recall

if __name__ == '__main__':
    
    dataset = 'Mental'
    dataset_path = 'data'
    path = os.path.join(dataset_path, dataset)
    code_maps, train_codes_data, test_codes_data, valid_codes_data, adj, embedding = load_data(path)
    print("Load Done!\nProcessing...")

    # (train_codes_x, train_codes_y, train_codes_intervals, train_visit_lens) = train_codes_data
    # (valid_codes_x, valid_codes_y, valid_codes_intervals, valid_visit_lens) = valid_codes_data
    # (test_codes_x, test_codes_y, test_codes_intervals, test_visit_lens) = test_codes_data
    
    (train_codes_x, train_codes_y, train_codes_intervals, train_visit_lens, train_embedding) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_codes_intervals, valid_visit_lens, valid_embedding) = valid_codes_data
    (test_codes_x, test_codes_y, test_codes_intervals, test_visit_lens, test_embedding) = test_codes_data



    op_conf = {
        'pretrain': False, # 进行选择  
        'from_pretrain': False,
        'path': './data/Mental/',
        'use_embedding_init': False,
        'task': 'm',  # m: medical codes, h: heart failure
    }


    task_conf = {
        'm': {
            'output_dim': len(code_maps),
            'loss_fn': medical_codes_loss,
        },
    }

    model_conf = {
        'code_embedding_init': None,
        'adj': adj,
        'max_visit_num': train_codes_x.shape[1],
        'code_num_pretrain': len(embedding) - 1,
        'output_dim': task_conf[op_conf['task']]['output_dim'],
        'loss' : task_conf[op_conf['task']]['loss_fn']
    }

    hyper_params = {
        'code_embedding_size': 256,
        'hiddens': [256, 128],
        'time_hidden': 64,
        'time_embedding_size': 128,
        'visit_embedding_size': 128,
        'attention_size': 64,
        'patients_embedding_size': 128,
        'epoch': 300,
        'batch_size': 32,
        'gnn_dropout_rate': 0.8,
        'decoder_dropout_rate': 0.17,
        'L2_reg' : 0,

        
        'lr': 1e-2,
        'layer' : 3,
    }
    print("Load parameters!")
    if op_conf['use_embedding_init']:
        if op_conf['pretrain'] or (not op_conf['from_pretrain']):
            model_conf['code_embedding_init'] = embedding

    model_choice_ = ['SatNet'] # name of the proposed HiTANet in our paper
    for model_choice in model_choice_:
        model_file = eval(model_choice)
        current_time = datetime.now().strftime('%m%d_%H:%M:%S')
        # Create the model name string
        model_name = 'GT-Ec_%s_L%d_wt_%.4f_patient_embedding_%s' % (model_choice, hyper_params['layer'], hyper_params['lr'], current_time)
        print(model_name)
        
        path = 'data/' + 'Mental-1/'

        output_file_path = 'cache/' + model_choice + '_GT-Ec/'
        log_file = output_file_path + '/GT-Ec.csv'
        if os.path.exists(log_file):
            df = pd.read_csv(log_file)
        else:
            df = pd.DataFrame(columns=['time','model_name','f1','recall'])
            
        if os.path.isdir(output_file_path):
            pass
        else:
            os.mkdir(output_file_path)
        results = []
        for k in range(1):
            f1, recall = train_model(model_conf, hyper_params, model_file, model_name, output_file_path)
            # results.append([f1, recall[0], recall[1], recall[2], recall[3]])
        # print(f1, recall)
        print(f"f1_score: {f1}, Recall: {recall}")
        # print(np.mean(results, 0))
        # print(np.std(results, 0))
        # with open(log_file, 'a') as file:
        #     # file.write(current_time + model_name + str(results) + '\n')
        #     file.write(current_time + ' ' + model_name + ' ' + str(f1) + ' ' + str(recall) + '\n')
        new_row = pd.DataFrame([{'time': current_time, 'model_name': model_name, 'f1': f1, 'recall': recall}], index=[0])
        df = pd.concat([df, new_row], ignore_index=True)

        # 保存DataFrame到文件
        df.to_csv(log_file, index=False, encoding='utf-8')