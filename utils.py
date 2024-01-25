import collections
import math
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, accuracy_score, fowlkes_mallows_score, normalized_mutual_info_score, adjusted_mutual_info_score
import copy
import random
randomSeed = 2023
random.seed(randomSeed)

from sklearn.metrics import r2_score, explained_variance_score

def masked_mae_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                    null_val=null_val))

def masked_mape_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def r2_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return r2_score(labels, preds)

def r2_score_adjust_torch(preds,labels):
    n,p = preds.shape[0],1
    
    return 1-((1-r2_score(labels,preds))*(n-1))/(n-p-1)

def masked_mse_torch(preds, labels, null_val=np.nan):
    labels[torch.abs(labels) < 1e-4] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def explained_variance_score_torch(preds, labels):
    preds = preds.cpu().flatten()
    labels = labels.cpu().flatten()
    return explained_variance_score(labels, preds)

def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]

def get_slot_by_datetime(datetime):
    format = '%Y-%m-%d %H:%M:%S'
    format_time = time.strptime(datetime, format)
    day, hour, minute = format_time.tm_mday, format_time.tm_hour, format_time.tm_min
    slot = (day - 1) * 48 + hour * 2 + (0 if minute <= 30 else 1)
    return slot

def get_slot_by_unix(ts):
    dt = time.localtime(ts)
    day, hour, minute = dt.tm_mday, dt.tm_hour, dt.tm_min
    slot = hour * 2 + (0 if minute <= 30 else 1)
    return slot

def list_to_array(x):
    dff = pd.concat([pd.DataFrame({'{}'.format(index): labels}) for index, labels in enumerate(x)], axis=1)
    return dff.fillna(0).values.T.astype(int)

def ts_to_slot(ts):
    ans = [0] * 48
    for t in ts:
        slot = get_slot_by_unix(t)
        ans[slot] = 1
    return ans

def traj_to_slot(trajectory, ts, pad=0):
    ans = [pad] * 48
    for i in range(len(ts)):
        slot_id = get_slot_by_unix(ts[i])
        ans[slot_id] = trajectory[i]
    return ans

def topk(ground_truth, logits_lm, k):
    pred_topk = logits_lm[:, :, 0:k]
    pred_topk = torch.flatten(pred_topk, start_dim=0, end_dim=1).cpu().data.numpy()
    topk_token = 0
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_topk[i]:
            topk_token += 1
    topk_score = topk_token / len(ground_truth)
    return topk_token, topk_score

def map_score(ground_truth, logits_lm):
    MAP = 0
    pred_topk = torch.flatten(logits_lm, start_dim=0, end_dim=1).cpu().data.numpy()
    for i in range(len(ground_truth)):
        if ground_truth[i] in pred_topk[i]:
            a = ground_truth[i]
            b = pred_topk[i]
            rank = np.argwhere(ground_truth[i] == pred_topk[i]) + 1
            MAP += 1.0 / rank[0][0]
    return MAP / len(ground_truth)

def get_evalution(ground_truth, logits_lm, exchange_matrix):
    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()
    accuracy_token = 0
    for i in range(len(ground_truth)):
        if pred_acc[i] == ground_truth[i]:
            accuracy_token += 1
    accuracy_score = accuracy_token / len(ground_truth)
    print("top1:", accuracy_token, accuracy_score)

    pred_acc = logits_lm[:, :, 0]
    pred_acc = pred_acc.flatten().cpu().data.numpy()

    fuzzy_accuracy_token = 0
    for i in range(len(pred_acc)):
        a = int(pred_acc[i])
        b = ground_truth[i]
        if exchange_matrix[b][a] > 0 or exchange_matrix[a][b] > 0:
            fuzzy_accuracy_token += 1
    fuzzy_score = fuzzy_accuracy_token / len(ground_truth)
    print("fuzzy:", fuzzy_accuracy_token, fuzzy_score)

    top3_token, top3_score = topk(ground_truth, logits_lm, 3)
    print("top3:", top3_token, top3_score)

    top5_token, top5_score = topk(ground_truth, logits_lm, 5)
    print("top5:", top5_token, top5_score)

    top10_token, top10_score = topk(ground_truth, logits_lm, 10)
    print("top10:", top10_token, top10_score)

    top30_token, top30_score = topk(ground_truth, logits_lm, 30)
    print("top30:", top30_token, top30_score)

    top50_token, top50_score = topk(ground_truth, logits_lm, 50)
    print("top50:", top50_token, top50_score)

    top100_token, top100_score = topk(ground_truth, logits_lm, 100)
    print("top100:", top100_token, top100_score)

    MAP = map_score(ground_truth, logits_lm)
    print("MAP score:", MAP)

    return accuracy_score, fuzzy_score, top3_score, top5_score, top10_score, top30_score, top50_score, top100_score, MAP

def get_acc(ground_truth, logits_lm):
    pre = torch.argmax(logits_lm,dim = 1)
    # # for i,each in enumerate(pre):
    # #     if each == ground_truth[i]:
    result = sum([1 if each == ground_truth[i] else 0 for i,each in enumerate(pre) ])/len(pre)
    return result

def target_distribution(q):
    # clustering target distribution for self-training
    # q (batch,n_clusters): similarity between embedded point and cluster center
    # p (batch,n_clusters): target distribution
    weight = q**2 / q.sum(0)
    p = (weight.t() / weight.sum(1)).t()
    return p

def update_cluster(model,dataloader,device,momentum_model):
    q = []
    model.eval()
    
    labels = []
    vecs = []
    torch.save(model.bert.state_dict(), 'before_update_cluster.pt')
    for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,label) in enumerate(dataloader):
        # context, q_i, head_in, head_cl = model(input_ids_o.to(device),lengths.to(device),id_mask.to(device),timestamp.to(device),momentum_model)
        context, q_i, head_in, head_cl = model(input_ids_o.to(device),lengths.to(device),id_mask.to(device),timestamp_o.to(device),momentum_model)

        q.append(q_i.cpu().data)
        vecs.append(context.cpu().data)
        labels.append(label)

    # (datasize,n_clusters)
    q = torch.cat(q)
    labels = torch.cat(labels)
    vecs = torch.cat(vecs)

    torch.save(model.bert.state_dict(), 'after_update_cluster.pt')
    model.train()

    return  q, target_distribution(q), labels, vecs

def cluster_acc(y_true, y_pred,record = False):
    """
    Calculate unsupervised clustering accuracy. Requires scikit-learn installed

    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = np.array(y_true)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    if record:
        if os.path.exists('cluster_w.txt'):
            os.remove('cluster_w.txt')
        for i in range(len(w)):
            with open('cluster_w.txt','a') as f:
                f.write(' '.join([str(each) for each in w[i]])+'\n')
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def nmi_score(y, y_pred):
    return normalized_mutual_info_score(y, y_pred)

def ami_score(y, y_pred):
    return adjusted_mutual_info_score(y, y_pred)

def ari_score(y, y_pred):
    return adjusted_rand_score(y, y_pred)

def fms_score(y, y_pred):
    return fowlkes_mallows_score(y, y_pred)


class Loss_Function(nn.Module):
    def __init__(self):
        super(Loss_Function, self).__init__()



    def clusterloss(self,q, p, loss_cuda):
        '''
        caculate the KL loss for clustering
        '''
        q, p = q.to(loss_cuda), p.to(loss_cuda)
        criterion = nn.KLDivLoss(reduction='sum').to(loss_cuda)
        return criterion(q.log(), p)

    def clusteringLoss(self,clusterlayer, context, p, cuda2, loss_cuda,q):
        """
        One batch cluster KL loss

        Input:
        context: (batch, hidden_size * num_directions) last hidden layer from encoder 
        clusterlayer: caculate Student’s t-distribution with clustering center

        p: (batch_size,n_clusters)target distribution

        Output:loss
        """
        batch = context.size(0)
        assert batch == p.size(0)
        kl_loss = self.clusterloss(q, p, loss_cuda)

        return kl_loss.div(batch)

    def Cross_Entropy_Loss(self, logit_lm, ground_truth):
        _, num_classes = logit_lm.size()
        p_i = torch.softmax(logit_lm, dim=1)
        y = F.one_hot(ground_truth, num_classes=num_classes)
        loss = y * torch.log(p_i + 0.0000001)
        loss = torch.sum(loss, dim=1)
        loss = -torch.mean(loss, dim=0)
        return loss

    def infonce(self, features_1, features_2, temperature=0.1):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        features = F.normalize(features,dim=1)
        features_1, features_2 = features.chunk(2,0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)

        neg_mean = torch.mean(neg)
        pos_n = torch.mean(pos)
        Ng = neg.sum(dim=-1)

        loss_pos = (- torch.log(pos / (Ng+pos))).mean()

        return loss_pos

    def nce(self, z1, z2, device):
        sim_mat = torch.mm(z1, z2.t())
        batch_size = z1.shape[0]
        pos_mask = torch.eye(batch_size).to(device)#.cuda()
        return nn.BCEWithLogitsLoss(reduction='none')(sim_mat, pos_mask).sum(1).mean()
    
    def contrastive_loss_simclr(self, z1, z2, temperature=0.1, similarity='inner'):
        """

        Args:
            z1(torch.tensor): (batch_size, d_model)
            z2(torch.tensor): (batch_size, d_model)

        Returns:

        """
        assert z1.shape == z2.shape
        batch_size, d_model = z1.shape
        features = torch.cat([z1, z2], dim=0)  # (batch_size * 2, d_model)

        labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        #labels = labels.to(self.device)

        features = F.normalize(features, dim=1)
        if similarity == 'inner':
            similarity_matrix = torch.matmul(features, features.T)
        elif similarity == 'cosine':
            similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=-1)
        else:
            similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool)#.to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)  # [batch_size * 2, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)  # [batch_size * 2, 2N-2]

        logits = torch.cat([positives, negatives], dim=1)  # (batch_size * 2, batch_size * 2 - 1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)  # (batch_size * 2, 1)
        logits = logits / temperature

        loss_res = Cross_Entropy_Loss(logits, labels)
        return loss_res



def make_exchange_matrix(token_list, token_size, alpha=98, theta=1000):
    token_list = [list(filter(lambda x: x > 3, token)) for token in token_list]  # 去掉pad mask等字符
    exchange_matrix = np.zeros(shape=(token_size, token_size))
    for token in token_list:
        for i in range(1, len(token)):
            if token[i] == token[i - 1]:
                continue
            exchange_matrix[token[i - 1]][token[i]] += 1  # 按照轨迹的方向统计
    print(np.min(exchange_matrix), np.max(exchange_matrix))
    exchange_matrix = np.where(exchange_matrix >= alpha, exchange_matrix, 0)  # 大于alpha的作为临近基站
    exchange_matrix = exchange_matrix / theta  # 做theta缩放
    exchange_matrix = np.where(exchange_matrix > 0, np.exp(exchange_matrix), 0)
    # exp(x)
    print(np.min(exchange_matrix), np.max(exchange_matrix))
    for i in range(token_size):
        row_sum = sum(exchange_matrix[i]) + np.exp(1)
        for j in range(token_size):
            if exchange_matrix[i][j] != 0:
                exchange_matrix[i][j] = exchange_matrix[i][j] / row_sum  # 除对角元素外softmax
    print(np.min(exchange_matrix), np.max(exchange_matrix))
    for i in range(token_size):
        exchange_matrix[i][i] = 1  # 对角元素置1

    for i in range(token_size):
        for j in range(token_size):
            exchange_matrix[i][j] = max(exchange_matrix[i][j], exchange_matrix[j][i])
    return exchange_matrix

def gen_train(train_df):
    # ['trajectory', 'user_index', 'day']
    records = []
    for index, row in train_df.iterrows():
        seq, user_index, day ,label= row['trajectory'], row['user_index'], row['year'] + row['month'] + row['day'],row['label']
        records.append([seq, user_index, day,label])
    print("All data length is " + str(len(records)))
    return records

def gen_data(train_df,col_list):
    # ['trajectory', 'user_index', 'day']
    records = []
    for index, row in train_df.iterrows():
        # seq, timestamp ,label= row['trajectory'], row['time'], row['label']
        records.append([row[col] for col in col_list])
    return records

def get_near(train_df):
    records = []
    for index, row in train_df.iterrows():
        near_traj,near_length,near_traj_id,timestamp = row['near_traj'],row['near_length'],row['near_traj_id'],row['time']
        records.append([near_traj ,near_length,near_traj_id,timestamp])
    return records


def gen_test(self):
    # ['trajectory', 'masked_pos', 'masked_tokens']
    test_df = self.test_df
    records = []
    for index, row in test_df.iterrows():
        seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
        user_index, day = row['user_index'], row['day']
        seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                         list(map(int, masked_tokens.split()))
        records.append([seq, masked_pos, masked_tokens, user_index, day])
    print("All test length is " + str(len(records)))
    return records

class DataAug(nn.Module):
    def __init__(self, data, lengths,timestamp,id_mask):
        super(DataAug, self).__init__()
        self.oriData = data
        self.lengths = lengths
        self.timestamp = timestamp
        self.id_mask = id_mask
    
    # drop some trajectory points
    def drop(self, droprate=0.1):   
        data = self.oriData.numpy()
        timestamp = self.timestamp.numpy()
        datasize = self.lengths.numpy()
        idmask = self.id_mask.numpy()
        batchsize = len(data)
        
        mask = [random.sample(range(datasize[i]), int(datasize[i]*droprate)) for i in range(batchsize)]
        # drop points and pad zeros
        newdata = [[data[i][j] for j in range(len(data[i])) if j not in mask[i]]+[0]*len(mask[i]) for i in range(len(data))]
        #后续可能需要drop 对应的时间戳。
        newtimestamp = [[timestamp[i][j] for j in range(len(timestamp[i])) if j not in mask[i]]+[0]*len(mask[i]) for i in range(len(timestamp))]

        new_idmask = [[idmask[i][j] for j in range(len(idmask[i])) if j not in mask[i]]+[0]*len(mask[i]) for i in range(len(idmask))]
        #需要输出新的长度？ 但是bert里是根据长度截取，这个0是嵌入式
        return torch.Tensor(newdata).int(),torch.Tensor(newtimestamp).int(),self.lengths , torch.Tensor(new_idmask).int()
    
    #单条轨迹的轨迹点偏移
    def offsetSingle(self,traj,rate = 0.2):
        '''
        输入 含0，1，2，3特殊token 的单条轨迹
        只挑选非 1，2，3的token进行偏移 偏移后也是非 1，2，3的token
        '''
        size= len(traj)
        for i in range(1,size-1):#除去起始终止点。
            p = random.uniform(0,1)
            if p <= rate:
                #选择前后一个的
                token = random.choice([traj[i-1],traj[i+1]])
                if token not in [1,2,3]:
                    traj[i] = token
        return traj

    # 对轨迹偏移
    def offsetTraj(self,rate= 0.2):
        '''
        输入原始轨迹，则不含3
        输入 含0，1，2，3特殊token 
        只挑选非 1，2，3的token进行偏移 偏移后也是非 1，2，3的token
        '''
        data = self.oriData.numpy()
        #rate概率用前后相邻的轨迹点替换
        newdata = [self.offsetSingle(traj,rate=rate) for traj in data]
        return torch.Tensor(newdata).int(),self.timestamp,self.lengths
    
    # 时间偏移
    def offsetTime(self,rate = 0.2,timesize = 1440,o_timestamp = None):
        '''
        deltaT 时间偏移量
        只挑选非 1，2，0的token进行偏移
        整体轨迹时间偏移
        平移轨迹周期的1/10
        在这里用 timestamp后 计算轨迹周期可能会有被mask的时间无法计算
        '''
        timestamp = self.timestamp.numpy()
        
        if o_timestamp == None:
            deltaT = int((1800*rate)/(86400//timesize) )#分类标准2小时 //15
            deltat = random.randint(0, deltaT) # 偏移量 如果希望对 timetoken 有影响 且在范围内
            newtimestamp = [[ t if t in [0,1,2] else (t + deltat ) % timesize  for t in each]for each in timestamp]
        else:
            o_timestamp = o_timestamp.numpy()

            deltat = [ random.randint(-int((each[self.lengths[i]] - each[1])*rate),int((each[self.lengths[i]] - each[1])*rate)) for i,each in enumerate(o_timestamp)]
            newtimestamp = [[ t if t in [0,1,2] else ((t + deltat[i] ) % timesize)  for t in each]for i,each in enumerate(timestamp)]
        
        return self.oriData,torch.Tensor(newtimestamp).int(),self.lengths
    
    #单条轨迹插值
    def insertSingle(self,traj,timestamp,id_mask,pad_len,rate = 0.2):
        '''
        输入原始轨迹 不含mask 3
        插值需要保证轨迹长度不变
        边插值边删除值=> 删除 0 1 2
        插值后会影响length 所以需要返回新length
        插值数量小于等于 pad 的数量。 whole size * rate <= num pad
        '''
        
        #插入附近轨迹，计算新长度
        #补充pad 0 和 开头结尾 1，2
        #删除轨迹中,时间戳中的 0,1,2 
        index = np.argwhere(traj < 3)
        traj = np.delete(traj, index)
        timestamp = np.delete(timestamp, index) 
        #计算出要插入的个数
        puresize = len(traj)
        insertsize = int((pad_len-puresize)*rate)
        #在剩余的点中随机选取点并插入
        insert_index = random.sample(range(puresize), insertsize)
        insert_traj_value = traj[insert_index]
        insert_time_value = timestamp[insert_index]
        newtraj = np.insert(traj,insert_index,insert_traj_value)
        newtimestamp = np.insert(timestamp,insert_index,insert_time_value)
        newlenght = len(newtraj)
        #再补0，1，2
        newtraj = np.insert(newtraj,[len(newtraj)],[0]*(pad_len-2-newlenght))#补0
        newtraj = np.insert(newtraj,[0,len(newtraj)],[1,2])#补1，2
        newtimestamp = np.insert(newtimestamp,[len(newtimestamp)],[0]*(pad_len-2-newlenght)) #补0
        newtimestamp = np.insert(newtimestamp,[0,len(newtimestamp)],[1,2])#补1，2

        new_idmask = np.array([0]+[1]*newlenght+[0]*(len(newtimestamp)-newlenght-2)+[0]) #构建 id mask
        return  newtraj,newtimestamp,newlenght,new_idmask
    # 插值
    def inserV(self,rate = 0.2):
        '''
        插值需要保证轨迹总长度不变(包含pad 1 2)
        '''
        batch_traj = self.oriData.numpy()
        batch_timestamp = self.timestamp.numpy()
        pure_len = self.lengths.numpy()
        id_mask = self.id_mask.numpy()
        newtraj,newtimestamp,newlength,new_idmask= [],[],[],[]
        pad_len = len(batch_traj[0])
        for i in range(len(batch_traj)):
            tmptraj,tmptime,tmplength,tmp_idmask = self.insertSingle(batch_traj[i],batch_timestamp[i],id_mask[i],pad_len,rate)
            newtraj.append(copy.deepcopy(tmptraj))
            newtimestamp.append(copy.deepcopy(tmptime))
            newlength.append(copy.deepcopy(tmplength))
            new_idmask.append(copy.deepcopy(tmp_idmask))

        return torch.Tensor(newtraj).int(),torch.Tensor(newtimestamp).int(),torch.Tensor(newlength).int() ,torch.Tensor(new_idmask).int()

if __name__ == '__main__':
    pass
