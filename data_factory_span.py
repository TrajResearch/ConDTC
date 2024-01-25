import collections
import random
random.seed(0)
# from random import shuffle
import pandas as pd
import os
# import time
# import csv
import pickle
from utils import traj_to_slot,gen_train,get_near,gen_data
import torch.utils.data as Data
from random import *
import os
import torch
import numpy as np
class data_provider():
    def __init__(self,args,sub = None):
        self.args = args
        # self.time_size = 5764
        self.time_size = self.args.timesize + 4

        if sub == None:
            if args.dataset == 'porto_sigma0001':
                dataset_path = './data/porto_sigma0001'
            elif  args.dataset == 'mobile':
                dataset_path = './data/mobile'
            elif  args.dataset == 'porto':
                dataset_path = './data/porto'
            elif  args.dataset == 'qd':
                dataset_path = './data/qd'
            elif args.dataset == 'cdr':
                dataset_path = './data/cdr'
            elif args.dataset == 'qd_half_label':
                dataset_path = './data/qd_half_label'

            elif args.dataset == 'cdr_q': # query
                dataset_path = './data/cdr_st2vec_q'
            elif args.dataset == 'cdr_d': # detour
                dataset_path = './data/cdr_st2vec_d'
            elif args.dataset == 'cdr_o': # other detour
                dataset_path = './data/cdr_st2vec_o'
            elif args.dataset == 'cdr_b': # other detour
                dataset_path = './data/cdr_st2vec_b'

            elif args.dataset == 'qdhalf_q': # query
                dataset_path = './data/qdhalf_st2vec_q'
            elif args.dataset == 'qdhalf_d': # detour
                dataset_path = './data/qdhalf_st2vec_d'
            elif args.dataset == 'qdhalf_o': # other detour
                dataset_path = './data/qdhalf_st2vec_o'
            
            elif args.dataset =='qdhalf':
                dataset_path = './data/qdhalf'
            
            elif args.dataset =='qddebug':
                dataset_path = './data/qddebug'
            
            elif args.dataset =='portodebug':
                dataset_path = './data/portodebug'

            elif args.dataset =='mobiledebug':
                dataset_path = './data/mobiledebug'

            elif args.dataset =='qddrop':
                dataset_path = './data/qddrop'
            
            elif args.dataset =='mobiletest':
                dataset_path = './data/mobiletest'
            
            elif args.dataset =='qdtest':
                dataset_path = './data/qdtest'

            elif args.dataset =='qdTimeNoise':
                dataset_path = './data/qdTimeNoise'  


            else:
                dataset_path = './data/'+args.dataset

         
        else:
            if args.dataset.split('_')[0] == 'cdr':
                dataset_path = './data/cdr_' + sub
            elif args.dataset.split('_')[0] == 'qdhalf':
                dataset_path = './data/qdhalf_' + sub

        self.dataset = os.path.join(dataset_path ,'data_k3.h5')
        self.traj_len = os.path.join(dataset_path ,'trj_length.csv')
        print('data path :',dataset_path)

        self.preprocess()
    
    
    def get_loader(self,flag,args):
        
        total_data = self.make_mask_data(flag,self.train_token_list,self.timestamp_list,self.args.datalen)
        input_ids, masked_tokens, masked_pos, input_ids_o,timestamp,time_masked_tokens,timestamp_o = zip(*total_data)
        
        id_mask=[]
        print('-'*50,'traj length',self.trajL,'-'*50)
        print('total data',len(input_ids))
        for l in self.trj_lengths['length'].values.tolist():
            id_mask.append([0]+[1]*int(l)+[0]*(self.trajL-1-int(l))) # 1 means loc token 0 means pad and cls sep

        data = []

        if flag == 'pretrain' :
            if self.args.task != 'cluster':
                data.append(torch.LongTensor(input_ids))
                data.append(torch.LongTensor(masked_tokens))
                data.append(torch.LongTensor(masked_pos))
                data.append(torch.LongTensor(timestamp))
                data.append(torch.LongTensor(time_masked_tokens))
            else:
                data.append(torch.LongTensor(input_ids))
                data.append(torch.LongTensor(masked_tokens))
                data.append(torch.LongTensor(masked_pos))
                data.append(torch.LongTensor(input_ids_o))
                data.append(torch.LongTensor(timestamp))
                print('time max value',torch.max(torch.LongTensor(timestamp)))
                print('time total num',torch.LongTensor(timestamp).unique().size())

                data.append(torch.LongTensor(time_masked_tokens))
                data.append(torch.LongTensor(timestamp_o))
                data.append(torch.LongTensor(id_mask))
                data.append(torch.LongTensor(self.trj_lengths['length'].values))
                data.append(torch.LongTensor(self.label_data ))

        elif flag == 'cluster':
            data.append(torch.LongTensor(input_ids))
            data.append(torch.LongTensor(masked_tokens))
            data.append(torch.LongTensor(masked_pos))
            data.append(torch.LongTensor(input_ids_o))
            data.append(torch.LongTensor(timestamp))
            data.append(torch.LongTensor(time_masked_tokens))
            data.append(torch.LongTensor(timestamp_o))
            data.append(torch.LongTensor(id_mask))
            data.append(torch.LongTensor(self.trj_lengths['length'].values))
            data.append(torch.LongTensor(self.label_data ))

        elif flag == 'eta':
            '''
            return Travel time estimation dataset 
            '''
            data.append(torch.LongTensor(input_ids))
            data.append(torch.LongTensor(masked_tokens))
            data.append(torch.LongTensor(masked_pos))
            data.append(torch.LongTensor(input_ids_o))
            data.append(torch.LongTensor(timestamp))
            data.append(torch.LongTensor(time_masked_tokens))
            data.append(torch.LongTensor(timestamp_o))
            data.append(torch.LongTensor(id_mask))
            data.append(torch.LongTensor(self.trj_lengths['length'].values))
            data.append(torch.FloatTensor(self.duration))
        
        elif flag == 'sim' :
            data.append(torch.LongTensor(input_ids_o))
            data.append(torch.LongTensor(masked_tokens))
            data.append(torch.LongTensor(masked_pos))
            data.append(torch.LongTensor(timestamp_o))
            data.append(torch.LongTensor(time_masked_tokens))
            data.append(torch.LongTensor(id_mask))
            data.append(torch.LongTensor(self.trj_lengths['length'].values))
            data.append(torch.LongTensor(self.traj_id))


        dataset = MyDataSet(data) 
      
        train_size = int(len(dataset) * self.args.trainP)
        valid_size =  (len(dataset)-train_size) // 2
        test_size = len(dataset) - train_size -valid_size
        train_dataset,valid_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size,valid_size, test_size],generator=torch.Generator().manual_seed(0))
        train_loader = Data.DataLoader(train_dataset, args.bs, False)
        valid_loader = Data.DataLoader(valid_dataset, args.bs, False)
        test_loader = Data.DataLoader(test_dataset, args.bs, False)
        alldata_loader = Data.DataLoader(dataset, args.bs, False)
        return alldata_loader,train_loader,valid_loader,test_loader
    
    def make_mask_data(self,flag,token_list,timestamp_list,max_pred):
        # 找到连续子数组，按照长度大到小返回子数组
        def find_consecutive_subarrays(arr):
            subarrays = []
            start_index = 0
            for i in range(1, len(arr)):
                if arr[i] != arr[i - 1] + 1:
                    subarrays.append(arr[start_index:i])
                    start_index = i

            subarrays.append(arr[start_index:])
            return sorted(subarrays, key=lambda x: len(x),reverse=True)

        # 尽可能选择连续token，连续长度不超过 max_con， 选择个数总长度不超过 pred_len
        def select_consecutive_values(arr, pred_len,max_con):
            if len(arr) < pred_len:
                raise ValueError("The array is too small to select the desired number of values.")
            selected = []
            remain = pred_len
            #当选中的个数小于 设定值时，那么就继续进行选择，
            while remain :
                candi = find_consecutive_subarrays(arr)
                candisize = len(candi[0])
                if candisize > max_con:
                    randid = randint(0,candisize-max_con)
                    res = candi[0][randid:randid+max_con]
                    selected = selected + res
                    for v in res:
                        arr.remove(v)
                else:
                    selected = selected + candi[0]
                    for v in candi[0]:
                        arr.remove(v)
                if len(selected) > pred_len :
                    selected = selected[:pred_len]
                remain = pred_len - len(selected)
                
            return selected
        
        total_data = []
        # max_pred -= 1
        maxT = np.max(timestamp_list)
        minT = np.unique(timestamp_list)[4]
        time_test = []
        for i in range(len(token_list)):
            tokens_a_index = i  # sample random index in sentences
            tokens_a = token_list[tokens_a_index]
            input_ids = [self.word2idx['[CLS]']] + tokens_a + [self.word2idx['[SEP]']]
            input_ids_o = [self.word2idx['[CLS]']] + tokens_a + [self.word2idx['[SEP]']]
            timestamp = [self.time2idx['[CLS]']] + timestamp_list[tokens_a_index] + [self.time2idx['[SEP]']]
            if flag == 'eta':
                st_time = timestamp_list[tokens_a_index][:1] + [0 for i in range(len(timestamp_list[tokens_a_index])-1)]
                timestamp_o = [self.time2idx['[CLS]']] +  st_time + [self.time2idx['[SEP]']]
            else:
                timestamp_o = [self.time2idx['[CLS]']] + timestamp_list[tokens_a_index] + [self.time2idx['[SEP]']]
            # MASK LM
            # n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
            cand_maked_pos = [i for i, token in enumerate(input_ids)
                            if token != self.word2idx['[CLS]'] and token != self.word2idx['[SEP]'] and token != self.word2idx[
                                '[PAD]']]  # candidate masked position
            # shuffle(cand_maked_pos)
            
            max_con = self.args.max_con
            masked_pos = select_consecutive_values(cand_maked_pos,max_pred,max_con)
            masked_tokens  = []
            time_masked_tokens = []

            for pos in masked_pos:
                masked_tokens.append(input_ids[pos])
                time_masked_tokens.append(timestamp[pos])
                if random() < 0.8:  # 80%
                    input_ids[pos] = self.word2idx['[MASK]']  # make mask
                    timestamp[pos] = self.time2idx['[MASK]']

                elif random() > 0.9:  # 10%
                    index = randint(4, self.vocab_size - 1)  # random index in vocabulary
                    input_ids[pos] = index  # replace
                    timestamp[pos] = randint(minT,maxT)
                    # timestamp[pos] = randint(4,maxT)

            if i == 1:
                print('-'*100)
                print(masked_pos,masked_tokens,input_ids)
                print(masked_pos,time_masked_tokens,timestamp)
            # if len(masked_tokens) == n_pred: #ValueError: expected sequence of length 5 at dim 1 (got 2)
                # total_data.append([input_ids, masked_tokens, masked_pos, input_ids_o,timestamp,time_masked_tokens,timestamp_o])
            time_test.append(timestamp)
            total_data.append([input_ids, masked_tokens, masked_pos, input_ids_o,timestamp,time_masked_tokens,timestamp_o])
        return total_data
    
    def get_vocabsize(self):
        return self.vocab_size
    
    def get_label_size(self):
        return self.label_size

    def get_max_len(self):
        return self.trajL

    def get_exchange_map(self):
        return self.exchange_map
    
    def preprocess(self):
        train_df = pd.read_hdf(self.dataset)
        col_list = ['trajectory','time']

        train_data = gen_data(train_df,col_list)

        if self.args.task == 'cluster':
            self.label_data = gen_data(train_df,['label'])
            self.label_data = [ each[0] for each in self.label_data]
            self.label_size = len(list(set(self.label_data)))
        
        elif self.args.task == 'eta':
            self.duration = gen_data(train_df,['duration'])
            self.duration = [ each[0] for each in self.duration]
        elif self.args.task == 'sim':
            self.traj_id = gen_data(train_df,['traj_id'])

        # 初始化 word2idx time2idx  字典
        self.word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        dict1 = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        dict2 = {str(i):i+4 for i in range(self.time_size)}
        self.time2idx = dict( dict1, **dict2 )#{'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        
        if self.args.loadpickle:
            print('load',self.args.dataset.split('_')[0]+'_vocab.pkl file')
            train_word_list = pickle.load(open('mid_data/'+self.args.dataset.split('_')[0]+'_vocab.pkl','rb'))

            for k,v in train_word_list.items():
                self.word2idx[k] = v 

        else:
            train_word_list = list(
                set(word for i in range(len(train_data)) for word in train_data[i][0].split()))
            
            train_word_list.remove('[PAD]')
            
            train_word_list_int = [int(i) for i in train_word_list]
            train_word_list_int.sort()
        
            for i, w in enumerate(train_word_list_int):
                if w == '[PAD]' or w == '[MASK]':
                    print("error")
                self.word2idx[str(w)] = i + 4
            # save vocab dict
            # with open('mid_data/'+self.args.dataset.split('_')[0]+'_vocab.pkl', 'wb') as f:
            #     pickle.dump(self.word2idx, f)
            #     print('save pkl ','mid_data/'+self.args.dataset.split('_')[0]+'_vocab.pkl')

        self.train_token_list = list()
        self.timestamp_list = list()

        # maxt = 0
        for sentence in train_data:
            seq, timestamp = sentence
            seq = seq.split()
            arr = [self.word2idx[s] for s in seq]
            self.train_token_list.append(arr)
            self.timestamp_list.append([(((int(t)//(5760//self.args.timesize))% self.args.timesize) + 4 ) if t != '[PAD]' else 0 for t in timestamp.split()])
        self.trj_lengths = pd.read_csv(self.traj_len)

        self.vocab_size = len(self.word2idx)
        self.trajL = len(self.train_token_list[0])+2



class MyDataSet(Data.Dataset):
    def __init__(self,data):
        self.data = data
    
    def __len__(self):
        return len(self.data[0]) 
    
    def __getitem__(self, idx):
        return [ self.data[i][idx] for i in range(len(self.data))]
