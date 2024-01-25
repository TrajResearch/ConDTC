import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn.parameter import Parameter



def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size, 1, seq_len]
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size, seq_len, seq_len]


def gelu(x):
    """
      Implementation of the gelu activation function.
      For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
      0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
      Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, embed_size):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TimeEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, timestamp):
        time_encode = timestamp.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        time_encode = torch.cos(time_encode)
        return self.div_term * time_encode


class Embedding(nn.Module):
    def __init__(self, args, max_len, vocab_size,dropout=0.1,timespan=86404): # 86400/15 +4 = 5760+4
        super(Embedding, self).__init__()
        self.args = args
        
        self.tok_embed = nn.Embedding(vocab_size, self.args.d_model)  # token embedding
        if self.args.freezeLemb:
            for param in self.tok_embed.parameters():
                param.requires_grad = False

        if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
            self.time_embed = nn.Embedding(timespan, self.args.d_model) 
            if self.args.freezeTemb:
                for param in self.time_embed.parameters():
                    param.requires_grad = False

        if self.args.embedding == 'position' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
            self.pos_embed = PositionalEncoding(max_len, self.args.d_model)
        

        self.linear = nn.Linear(self.args.d_model*2,self.args.d_model)
        self.norm = nn.LayerNorm(self.args.d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x,timestamp):
        # print(x[1])
        # print(timestamp[1])
        # exit()
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)#.to(device)  # [seq_len] -> [batch_size, seq_len]

        if self.args.embedding == 'temporal':
            embedding = self.tok_embed(x) + self.time_embed(timestamp)
        elif self.args.embedding == 'position':
            embedding = self.tok_embed(x) + self.pos_embed(pos)
        elif  self.args.embedding == 'both':
            embedding = self.tok_embed(x) + self.time_embed(timestamp) + self.pos_embed(pos)
        elif self.args.embedding == 'onlyt':
            embedding =  self.time_embed(timestamp) + self.pos_embed(pos)
        embedding = self.norm(embedding)
        return self.dropout(embedding)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask,d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # scores : [batch_size, n_heads, seq_len, seq_len]
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.args = args
        self.d_k = self.d_v = self.args.d_model // self.args.head
        self.W_Q = nn.Linear(self.args.d_model, self.d_k * self.args.head, bias=False)
        self.W_K = nn.Linear(self.args.d_model, self.d_k * self.args.head, bias=False)
        self.W_V = nn.Linear(self.args.d_model, self.d_v * self.args.head, bias=False)
        self.fc = nn.Linear(self.args.head * self.d_v, self.args.d_model, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(self.args.d_model)

    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size, seq_len, d_model], k: [batch_size, seq_len, d_model], v: [batch_size, seq_len, d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.args.head, self.d_k).transpose(1, 2)  # q_s: [batch_size, n_heads, seq_len, d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.args.head, self.d_k).transpose(1, 2)  # k_s: [batch_size, n_heads, seq_len, d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.args.head, self.d_v).transpose(1, 2)  # v_s: [batch_size, n_heads, seq_len, d_v]

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.args.head, 1,
                                                    1)  # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, seq_len, d_v], attn: [batch_size, n_heads, seq_len, seq_len]
        context = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask,self.d_k)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1,
                                                            self.args.head * self.d_v)  # context: [batch_size, seq_len, n_heads * d_v]
        output = self.fc(context)
        output = self.dropout(output)
        return self.layernorm(output + residual)  # output: [batch_size, seq_len, d_model]



class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(PoswiseFeedForwardNet, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(self.args.d_model, self.args.d_model*4)
        self.dropout1 = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(self.args.d_model*4, self.args.d_model)
        self.dropout2 = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(self.args.d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.fc2(self.dropout1(gelu(self.fc1(x))))
        x = x + self.dropout2(x)

        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, args, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(args, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(args, dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                         enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size, seq_len, d_model]
        return enc_outputs


class BERT(nn.Module):
    def __init__(self, args,max_len,vocab_size, dropout=0.1):
        super(BERT, self).__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embedding = Embedding(self.args,max_len,vocab_size, dropout,timespan=self.args.timesize+4)
        self.layers = nn.ModuleList([EncoderLayer(self.args, dropout) for _ in range(self.args.layer)])
        
        self.linear = nn.Linear(self.args.d_model, self.args.d_model)
        self.activ2 = gelu
        # fc2 is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(self.args.d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

        # temporal_fc is shared with embedding layer
        if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
            tiem_embed_weight = self.embedding.time_embed.weight
            timestamp_size = self.args.timesize+4
            self.temporal_fc = nn.Linear(self.args.d_model, timestamp_size, bias=False)
            self.temporal_fc.weight = tiem_embed_weight

    def forward(self, input_ids, masked_pos,timestamp ):
        output = self.embedding(input_ids,timestamp)  # [bach_size, seq_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size, maxlen, maxlen]
        for layer in self.layers:
            # output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)

        masked_pos = masked_pos[:, :, None].expand(-1, -1, self.args.d_model)  # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size, max_pred, d_model]
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size, max_pred, d_model] 
        logits_lm = self.fc2(h_masked)  # [batch_size, max_pred, vocab_size]
        logits_lm_tm = None
        if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
            logits_lm_tm = self.temporal_fc(h_masked)  # temporal layer [batch_size, max_pred, timestamp_size]
        return logits_lm,logits_lm_tm

class ClusertLayer(nn.Module):
    def __init__(self, args,n_clusters,alpha=1):
        super(ClusertLayer,self).__init__()

        # self.clusters = Parameter(torch.Tensor(n_clusters, cluster_d), requires_grad=True).cuda()
        self.clusters = Parameter(torch.Tensor(n_clusters, args.d_model), requires_grad=True)#.cuda()

        self.alpha = alpha

    def forward(self, context):
        # clustering: caculate Student’s t-distribution
        # clusters (n_clusters, hidden_size * num_directions)
        # context (batch, hidden_size * num_directions)
        # q (batch,n_clusters): similarity between embedded point and cluster center
        # distance = torch.sum(torch.pow(context.unsqueeze(1) - self.clusters, 2), 2)
        
        distance = context.unsqueeze(1).cpu() - self.clusters.cpu()
        distance = torch.sum(torch.pow(distance, 2), 2)
        q = 1.0 / (1.0 + distance / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q


class BERT_CLUSTER(nn.Module):
    def __init__(self, args,n_clusters,model_mlm:BERT, dropout=0.1):
        super(BERT_CLUSTER, self).__init__()
        self.args = args
        self.bert = model_mlm
        self.clusterlayer = ClusertLayer(args,n_clusters)
        self.dropout = dropout
        self.mlp_in = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.args.d_model, 128)
        )
        self.mlp_cl = nn.Sequential(
            nn.Linear(self.args.d_model, self.args.d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self.args.d_model, n_clusters)
        )
        self.norm = nn.LayerNorm(n_clusters)
    
    def forward(self, input_ids_original, lengths, id_mask,timestamp,momentum_encoder):
        input_ids = input_ids_original
        model = momentum_encoder if momentum_encoder else self.bert
        output = model.embedding(input_ids,timestamp) 
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.bert.layers:
            output = layer(output, enc_self_attn_mask)
        # it will be decided by first token(CLS)
        id_masks = id_mask.unsqueeze(2).repeat(1,1,self.args.d_model)#.cuda()
        lengths=lengths.unsqueeze(1).repeat(1,self.args.d_model)#.cuda()
        output = (output*id_masks).sum(1)/lengths
        output = output.view(output.size()[0],-1) #256,256
        q = self.clusterlayer(output)
        head_in = self.mlp_in(output)
        head_cl = self.mlp_cl(output).t()

        return output, q, head_in, head_cl

class BERT_ETA(nn.Module):
    def __init__(self, args,model_mlm:BERT, dropout=0.1):
        super(BERT_ETA, self).__init__()
        self.args = args
        self.bert = model_mlm
        self.dropout = dropout
        self.linear = nn.Linear(self.args.d_model, 1)
    
    def forward(self, input_ids, lengths, id_mask, timestamp, momentum_encoder):
        model = momentum_encoder if momentum_encoder else self.bert
        output = model.embedding(input_ids,timestamp) 
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.bert.layers:
            #output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)

        id_masks = id_mask.unsqueeze(2).repeat(1,1,self.args.d_model)#.cuda()
        lengths=lengths.unsqueeze(1).repeat(1,self.args.d_model)#.cuda()
        output = (output*id_masks).sum(1)/lengths
        output = output.view(output.size()[0],-1) #256,256
        
        output = self.linear(output) # batch,1
        return output

class BERT_sim(nn.Module):
    def __init__(self, args,model_mlm:BERT, dropout=0.1):
        super(BERT_sim, self).__init__()
        self.args = args
        self.bert = model_mlm
        self.dropout = dropout
    
    def forward(self, input_ids, lengths, id_mask, timestamp, momentum_encoder):
        # model = momentum_encoder if momentum_encoder else self.bert
        model = self.bert
        output = model.embedding(input_ids,timestamp) 
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        for layer in self.bert.layers:
            #output: [batch_size, max_len, d_model]
            output = layer(output, enc_self_attn_mask)

        id_masks = id_mask.unsqueeze(2).repeat(1,1,self.args.d_model)#.cuda()
        lengths=lengths.unsqueeze(1).repeat(1,self.args.d_model)#.cuda()
        # output = (output*id_masks).sum(1)/lengths
        # output = output.view(output.size()[0],-1) #256,256 ,batch, d_model
        output = output.mean(dim=1)
        
        return output
