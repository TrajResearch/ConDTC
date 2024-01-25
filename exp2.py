from bert import BERT, BERT_CLUSTER,BERT_ETA,BERT_sim

import os
import torch
import torch.nn as nn
from torch import optim
from data_factory_span import data_provider
from utils import  Loss_Function,cluster_acc, nmi_score, ari_score,update_cluster, DataAug,get_acc ,masked_mae_torch, masked_rmse_torch, masked_mape_torch ,r2_score_torch ,explained_variance_score_torch,r2_score_adjust_torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import pair_confusion_matrix

import numpy as np
import time
import copy
import torch.nn.init as init
import pickle
import pandas as pd
    
class Exp():
    def __init__(self,args) :
        super(Exp, self).__init__()
        self.args = args
        self.device = self._acquire_device()
        self.timelossRate = args.tRate

        # 设置随机数种子
        seed = args.seed
        torch.manual_seed(seed)
        # 如果您在代码中使用了 GPU（CUDA）
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        if not os.path.exists('pth_model'):
            os.mkdir('pth_model')
        if not os.path.exists('result'):
            os.mkdir('result')
        if not os.path.exists('pth_model/span2'):
            os.mkdir('pth_model/span2')

        if self.args.dataset not in [ 'cdr_sep' ,'qdhalf_sep']or self.args.task=='sim':
            self.data_provider = data_provider(args)
            self.vocab_size = self.data_provider.get_vocabsize()
            max_len = self.data_provider.get_max_len()

        elif self.args.dataset  in [ 'cdr_sep' ,'qdhalf_sep']:
            self.train_data_loader = data_provider(args,sub = 'train')
            self.test_data_loader = data_provider(args,sub = 'test')
            self.valid_data_loader = data_provider(args,sub = 'valid')
            self.vocab_size = self.train_data_loader.get_vocabsize()
            max_len = self.train_data_loader.get_max_len()

        print('vocab size: ',self.vocab_size )

        self.pretrain_model = BERT(self.args,max_len,self.vocab_size).to(self.device)
        #初始化模型参数
        self.pretrain_model.apply(self.init_weights)

        if self.args.task == 'cluster':
            if self.args.wrongK == 0:
                self.n_cluster = self.data_provider.get_label_size()
            else:
                self.n_cluster  = self.args.wrongK
            print('label size: ',self.n_cluster )
            self.cluster_model = BERT_CLUSTER(self.args,self.n_cluster,self.pretrain_model).to(self.device)
        elif self.args.task == 'eta':
            best_model_path = './pth_model/test/cdr_sep_model_embedding_%s_timesize_1440_tRate_1.0_kl_1.0_in_1.0_cl_1.0_momentum_0.995.pth'% (str(self.args.embedding))
            self.pretrain_model = self.load_weight(self.pretrain_model,best_model_path)
            self.ETA_model = BERT_ETA(self.args,self.pretrain_model).to(self.device)
        elif self.args.task == 'sim':
            
            best_model_path = './pth_model/span2/porto_pretrain1031_model_embedding_both_timesize_1440_tRate_1.0_kl_1.0_in_0.0_cl_0.0_momentum_0.0_seed_19721013_gamma_2.0.pth'
            
            self.pretrain_model = self.load_weight(self.pretrain_model,best_model_path)
            self.sim_model = BERT_sim(self.args,self.pretrain_model).to(self.device)

        if not os.path.exists('result'):os.mkdir('result')

        self.momentum = args.momentum
        if self.momentum >0:
            self.momentum_model = copy.deepcopy(self.pretrain_model).to(self.device)
            for param, param_m in zip(self.pretrain_model.parameters(), self.momentum_model.parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient  
        else:
            self.momentum_model = None
        self.time_embed_size = self.args.timesize + 4


    def _select_optimizer(self,flag):
        if self.args.lrsep:
            seplr = self.args.lr*3
        else:
            seplr = self.args.lr
            
        if flag == 'pretrain':
            optimizer = optim.AdamW(self.pretrain_model.parameters(), lr=self.args.prelr)
        elif flag == 'train' :
            optimizer = optim.AdamW(self.cluster_model.parameters(), lr=self.args.lr)
        elif flag == 'train_seperate':
            optimizer = optim.AdamW(
                [{'params':self.cluster_model.bert.parameters(),'lr':self.args.lr},
                 {'params':self.cluster_model.clusterlayer.parameters()},
                 {'params':self.cluster_model.mlp_in.parameters()},
                 {'params':self.cluster_model.norm.parameters()}
                 ],
                lr=seplr
            )
        elif flag == 'eta':
            # optimizer = optim.Adam(self.ETA_model.parameters(), lr=self.args.lr*10)
            optimizer = optim.AdamW(
                [{'params':self.ETA_model.bert.parameters(),'lr':self.args.lr},
                 {'params':self.ETA_model.linear.parameters()}
                 ],
                lr=self.args.lr*10
            )
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=0)
        # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #             optimizer, mode='min', patience=105,
        #             factor=0.8, threshold=1e-4)
        lr_scheduler = None
        # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)

        
        return optimizer,lr_scheduler

    def _select_criterion(self):
        criterion = Loss_Function()
        return criterion
    
    def test(self,test_loader):
        acc = []
        loc_acc,time_acc = [],[]
        self.pretrain_model.eval()
        with torch.no_grad():
            if self.args.task != 'cluster':
                for i, (input_ids, masked_tokens, masked_pos, timestamp, time_masked_tokens) in enumerate(test_loader):
                    mlm_pred,mlm_temporal = self.pretrain_model(input_ids.to(self.device), masked_pos.to(self.device),timestamp.to(self.device))
                    lacc = get_acc( masked_tokens.view(-1).to(self.device),mlm_pred.view(-1, self.vocab_size).to(self.device))
                    loc_acc.append(lacc)
                    acc.append(lacc)
                    if self.args.embedding == 'temporal' or self.args.embedding == 'both':
                        timestamp_size = self.time_embed_size
                        tacc = get_acc(time_masked_tokens.view(-1).to(self.device),mlm_temporal.view(-1, timestamp_size).to(self.device) )
                        time_acc.append(tacc)
                        acc.append(tacc)
            else:
                for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,label) in enumerate(test_loader):
                    mlm_pred,mlm_temporal = self.pretrain_model(input_ids.to(self.device), masked_pos.to(self.device),timestamp.to(self.device))
                    lacc = get_acc( masked_tokens.view(-1).to(self.device),mlm_pred.view(-1, self.vocab_size).to(self.device))
                    loc_acc.append(lacc)
                    acc.append(lacc)
                    if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
                        timestamp_size = self.time_embed_size
                        tacc = get_acc(time_masked_tokens.view(-1).to(self.device),mlm_temporal.view(-1, timestamp_size).to(self.device) )
                        time_acc.append(tacc)
                        acc.append(tacc)
        self.pretrain_model.train()
        acc = sum(acc)/len(acc)
        loc_acc = sum(loc_acc)/len(loc_acc)
        if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
            time_acc= sum(time_acc)/len(time_acc)
        else:time_acc = 0
        return acc, loc_acc, time_acc
 
    def pretrain(self,setting):
        pretrain_st = time.time()
        if self.args.dataset not in  ['cdr_sep','qdhalf_sep']:
            loader,train_loader,valid_loader,test_loader = self.data_provider.get_loader(flag='pretrain',args = self.args)
            if self.args.task == 'cluster':
                self.cluster_loader = loader

        else:
            train_loader,_,_,_ = self.train_data_loader.get_loader(flag='pretrain',args = self.args)
            test_loader ,_,_,_ =  self.test_data_loader.get_loader(flag='pretrain',args = self.args)
            valid_loader,_,_,_ = self.valid_data_loader.get_loader(flag='pretrain',args = self.args)
        model_optim,lr_scheduler = self._select_optimizer('pretrain') 
        criterion = self._select_criterion()

        earlystop = 0
        if self.args.dataset in ['qdhalf']:
            best_model_path =  './pth_model/test/'+self.args.dataset+'_sep_model_embedding_both.pth'

        elif self.args.dataset in ['qddebug','qddrop']:
            best_model_path ='./pth_model/test/qd_model_embedding_both.pth'

        else:
            if self.args.loadmodel == '':
                best_model_path = './pth_model/span2/%s_model_embedding_%s_timesize_%s_tRate_%s_kl_%s_in_%s_cl_%s_momentum_%s_seed_%s_gamma_%s_epoch_%s.pth'% (self.args.dataset,str(self.args.embedding),str(self.args.timesize),str(self.args.tRate),str(self.args.kl),str(self.args.ncein),str(self.args.ncecl),str(self.args.momentum),str(self.args.seed),str(self.args.gamma),str(self.args.pretrain_epoch))
                
            else:
                best_model_path  = './pth_model/span2/'+self.args.loadmodel

        if self.args.pretrain_epoch == 0:
            best_model_path = './pth_model/span2/porto_pretrain1031_model_embedding_both_timesize_1440_tRate_1.0_kl_1.0_in_0.0_cl_0.0_momentum_0.0_seed_19721013_gamma_2.0_epoch_10.pth'

            self.pretrain_model = self.load_weight(self.pretrain_model,best_model_path)
            print('****Loaded pretrained model********',best_model_path)
            try:
                best_acc = torch.load(best_model_path,map_location=self.device)['best_acc']
                print('load model best acc:',best_acc)
                acc,loc_acc,time_acc = self.test(valid_loader)
                print('valid dataset acc: ',acc,'loc acc: ',loc_acc,'time acc: ',time_acc)
            except:
                best_acc = 0 
        else:
            '''
            try: 
                best_acc = torch.load(best_model_path,map_location=self.device)['best_acc']
                self.pretrain_model = self.load_weight(self.pretrain_model,best_model_path)
                print('model best acc:',best_acc)
            except:
                best_acc = 0
            '''
            best_acc = 0

            avg_train_time_list = []
            avg_valid_time_list = []
            train_token = []
            train_time_token = []
            for epoch in range(self.args.pretrain_epoch):
                each_epoch_st_t = time.time()
                train_loss = []
                self.pretrain_model.train()

                if self.args.task != 'cluster':
                    for i, (input_ids, masked_tokens, masked_pos, timestamp, time_masked_tokens) in enumerate(train_loader):
                        model_optim.zero_grad()
                        mlm_pred,mlm_temporal = self.pretrain_model(input_ids.to(self.device), masked_pos.to(self.device),timestamp.to(self.device))
                        mlm_loss_lm = criterion.Cross_Entropy_Loss(mlm_pred.view(-1, self.vocab_size).to(self.device), masked_tokens.view(-1).to(self.device)) 
                        timestamp_size = self.time_embed_size
                        mlm_loss = (mlm_loss_lm.float()).mean().cpu() * 1
                        
                        loss = mlm_loss.clone()
                        if self.args.embedding == 'temporal' or self.args.embedding == 'both':
                            mlm_temporal_loss = criterion.Cross_Entropy_Loss(mlm_temporal.view(-1, timestamp_size).to(self.device), time_masked_tokens.view(-1).to(self.device))
                            # for masked LM
                            mlm_temporal_loss = (mlm_temporal_loss.float()).mean().cpu() * 1
                            loss +=  mlm_temporal_loss * self.timelossRate 
                        else: mlm_temporal_loss = torch.zeros(1)
                        print("Epoch: %d, batch: %d, loss: %f, mlm loss: %f, mlm temporal loss: %f" %(epoch + 1, i, loss, mlm_loss, mlm_temporal_loss))
                        train_loss.append(loss)
                        loss.backward()
                        model_optim.step()
                else:
                    for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,label) in enumerate(train_loader):
                        if epoch == 0:
                            train_token.append(input_ids)
                            train_time_token.append(timestamp)
                        model_optim.zero_grad()
                        mlm_pred,mlm_temporal = self.pretrain_model(input_ids.to(self.device), masked_pos.to(self.device),timestamp.to(self.device))
                        mlm_loss_lm = criterion.Cross_Entropy_Loss(mlm_pred.view(-1, self.vocab_size).to(self.device), masked_tokens.view(-1).to(self.device)) 
                        timestamp_size = self.time_embed_size
                        mlm_loss = (mlm_loss_lm.float()).mean().cpu() * 1
                        if self.args.embedding == 'onlyt':
                            mlm_loss = torch.zeros(1)
                        else:
                            loss = mlm_loss.clone()
                        
                        if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
                            mlm_temporal_loss = criterion.Cross_Entropy_Loss(mlm_temporal.view(-1, timestamp_size).to(self.device), time_masked_tokens.view(-1).to(self.device))
                            # for masked LM
                            mlm_temporal_loss = (mlm_temporal_loss.float()).mean().cpu() * 1
                            if self.args.embedding == 'onlyt':
                                loss =  mlm_temporal_loss * self.timelossRate 
                            else:
                                loss +=  mlm_temporal_loss * self.timelossRate 
                        else: mlm_temporal_loss = torch.zeros(1)
                        print("Epoch: %d, batch: %d, loss: %f, mlm loss: %f, mlm temporal loss: %f, lr: %f " %(epoch + 1, i, loss, mlm_loss, mlm_temporal_loss,model_optim.state_dict()['param_groups'][0]['lr']))
                        
                        train_loss.append(loss)
                        loss.backward()
                        model_optim.step()
                        # lr_scheduler.step()

                train_loss = torch.mean(torch.stack(train_loss))
                print("Epoch: {} | Train Loss: {}  ".format(epoch + 1, train_loss))
                each_epoch_train_finish_t = time.time()
                if (epoch+1) % 1 == 0:   
                    acc,loc_acc,time_acc = self.test(valid_loader)
                    print('valid dataset acc: ',acc,'loc acc: ',loc_acc,'time acc: ',time_acc)
                    # # 需注掉
                    # tacc,tloc_acc,ttime_acc = self.test(test_loader)
                    # print('test dataset acc: ',tacc,'loc acc: ',tloc_acc,'time acc: ',ttime_acc)

                    if acc >= best_acc:
                        best_acc = acc
                        torch.save({'model': self.pretrain_model,'best_acc':best_acc}, best_model_path)
                        earlystop = 0
                    if best_acc > acc:
                        earlystop += 1
                    
                    if earlystop > 10:
                        print('early stop')
                        break
                # adjust_learning_rate(model_optim, epoch + 1, self.args)
                each_epoch_test_finish_t = time.time()
            avg_train_time_list.append(each_epoch_train_finish_t-each_epoch_st_t)
            avg_valid_time_list.append(each_epoch_test_finish_t-each_epoch_train_finish_t)
            if not os.path.exists('./pth_model'):
                os.mkdir('pth_model')
            # torch.save({'model': self.pretrain_model}, model_path)
            # torch.save(self.pretrain_model, model_path)
            train_token = torch.cat(train_token,dim =0)
            train_time_token = torch.cat(train_time_token,dim=0)
            print('-'*100)
            print('train token len',train_token.unique().size())
            print('train time token len',train_time_token.unique().size())
            
            if self.momentum > 0:
                self.momentum_model = copy.deepcopy(self.pretrain_model).to(self.device)
                for param, param_m in zip(self.pretrain_model.parameters(), self.momentum_model.parameters()):
                    param_m.data.copy_(param.data)  # initialize
                    param_m.requires_grad = False  # not update by gradient  
            else:
                self.momentum_model = None 
        
        # 加载best模型
        print('load best model',best_model_path)
        self.pretrain_model = self.load_weight(self.pretrain_model,best_model_path)
        
        test_time_st = time.time()
        if  self.args.dataset == 'cdr_sep':
            test_acc,test_loc_acc,test_time_acc = self.test(test_loader)
        else:
            test_acc,test_loc_acc,test_time_acc = self.test(test_loader)
            # test_acc,test_loc_acc,test_time_acc = self.test(valid_loader)
        test_time_ed = time.time()
        if self.args.pretrain_epoch != 0:
            print('test dataset acc: ',test_acc,'loc acc: ',test_loc_acc,'time acc: ',test_time_acc)
            print('avg train time',round(sum(avg_train_time_list)/self.args.pretrain_epoch,3),'s')
            print('avg eval time',round(sum(avg_valid_time_list)/self.args.pretrain_epoch,3),'s')
            print('test cost ',round(test_time_ed- test_time_st,3),'s')
            print('total pretrain time',round(test_time_ed - pretrain_st),'s')
        
        # 0316
        self.momentum = self.args.momentum
        if self.momentum >0:
            self.momentum_model = copy.deepcopy(self.pretrain_model).to(self.device)
            for param, param_m in zip(self.pretrain_model.parameters(), self.momentum_model.parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient  
        else:
            self.momentum_model = None

        # torch.save(self.pretrain_model.state_dict(), 'pretrain.pt')

        return

    def train(self,setting):

        model_optim,lr_scheduler = self._select_optimizer('train_seperate') 

        criterion = self._select_criterion()
        vecsForKmeans, kmeansCenters, init_acc = self.init_clusterlayer()
        if self.args.epoch == 0 :
            return init_acc
        self.cluster_model.train()
        best_acc = 0
        best_nmi = 0
        best_ari = 0
        best_epoch = 0
        best_context = None
        for ep in range(self.args.epoch):
            if ep % 1 == 0:
                    with torch.no_grad():
                        # q (datasize,n_clusters)
                        tmp_q, p, labels, vecs = update_cluster(self.cluster_model,self.cluster_loader,self.device,None)
                    # cc = nn.KLDivLoss(reduction='sum').cuda()
                    y = labels.cpu()
                    y_pred = tmp_q.numpy().argmax(1)
                    acc = cluster_acc(y, y_pred)
                    nmi = nmi_score(y, y_pred)
                    ari = ari_score(y, y_pred)
                   
                    # 保存pertain 后的 轨迹表征
                    if ep == 0 :
                        np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_init_vec',vecs)
                        np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_init_true_label',y.numpy())
                        np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_init_pre_label',y_pred)
                    
                    if(best_acc < acc):
                        best_acc = acc
                        best_nmi = nmi
                        best_ari = ari
                        best_epoch = ep + 1
                        best_context = vecs 
                        best_true_label = y.numpy()
                        best_pre_label = y_pred
                        # 保存结果最好的 轨迹表征
                        np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_best_vec',best_context)
                        np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_best_pre_label',best_pre_label)
                        np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_best_true_label',best_true_label)
                        matrix = pair_confusion_matrix(y.numpy(), y_pred)
                        # 将矩阵写入文本文件
                        with open('pair_confusion_matrix.txt', 'w') as f:
                            for row in matrix:
                                line = ' '.join(str(x) for x in row)  # 将每个元素转换为字符串，并用空格分隔
                                f.write(line + '\n')  # 写入文件，并在每行末尾添加换行符
                    print('Epoch: ', ep+1)
                    print('acc',acc)
                    print('nmi', nmi)
                    print('ari', ari)
            # timestampList = []
            # epoch_lable = []
            print('use data aug method', self.args.dataaug)
            for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,label) in enumerate(self.cluster_loader):
                # if len(label) == self.args.bs:
                #     epoch_lable.append(label.cpu().detach().numpy())
                model_optim.zero_grad()
                dataAug = DataAug(input_ids_o, lengths,timestamp,id_mask)
                # augT1 = time.time()
                if self.args.dataaug == 'drop':
                    id_mask2 = id_mask
                    data1,timestamp1,length1,id_mask1 = dataAug.drop(self.args.droprate)
                    data2,timestamp2,length2 = dataAug.offsetTime(self.args.droprate,self.args.timesize,timestamp_o)

                    # data2,timestamp2,length2 = dataAug.offsetTraj(self.args.droprate)
                elif self.args.dataaug == 'offsetTraj':
                    id_mask1 = id_mask2 = id_mask
                    data1,timestamp1,length1 = dataAug.offsetTraj(self.args.droprate)
                    data2,timestamp2,length2 = dataAug.offsetTime(self.args.droprate,self.args.timesize,timestamp_o)

                    # data2,timestamp2,length2 = dataAug.offsetTime(self.args.droprate)

                elif self.args.dataaug == 'offsetTime':
                    id_mask1 = id_mask2 = id_mask
                    data1,timestamp1,length1 = dataAug.offsetTime(self.args.droprate,self.args.timesize,timestamp_o)
                    data2,timestamp2,length2 = dataAug.offsetTime(self.args.droprate,self.args.timesize,timestamp_o)

                elif self.args.dataaug == 'inserV':
                    id_mask2 = id_mask
                    data1,timestamp1,length1,id_mask1 = dataAug.inserV(self.args.droprate)
                    data2,timestamp2,length2 = dataAug.offsetTime(self.args.droprate,self.args.timesize,timestamp_o)


                else:
                    id_mask1 = id_mask2 = id_mask
                    data1,timestamp1,length1 = input_ids_o,timestamp_o,lengths
                    data2,timestamp2,length2 = input_ids_o,timestamp_o,lengths
                
                loss = 0
                mlm_pred,mlm_temporal = self.cluster_model.bert(input_ids.to(self.device), masked_pos.to(self.device),timestamp.to(self.device))

                mlm_loss_lm = criterion.Cross_Entropy_Loss(mlm_pred.view(-1, self.vocab_size).to(self.device), masked_tokens.view(-1).to(self.device))  # for masked LM
                mlm_loss = (mlm_loss_lm.float()).mean()
                timestamp_size = self.time_embed_size
                if self.args.embedding == 'temporal' or self.args.embedding == 'both' or self.args.embedding == 'onlyt':
                    mlm_temporal_loss = criterion.Cross_Entropy_Loss(mlm_temporal.view(-1, timestamp_size).to(self.device), time_masked_tokens.view(-1).to(self.device))
                    mlm_temporal_loss = (mlm_temporal_loss.float()).mean() 
                    if self.args.embedding == 'onlyt':
                        mlm_loss = mlm_temporal_loss * self.timelossRate
                    else:
                        mlm_loss += mlm_temporal_loss * self.timelossRate 
                else: mlm_temporal_loss = torch.zeros(1)
                if self.args.mlm:
                    loss+=mlm_loss

                #loss = 0
                context, q, head_in, head_cl = self.cluster_model(data1.to(self.device), length1.to(self.device), id_mask1.to(self.device),timestamp1.to(self.device),None)
                context1, q1, head_in1, head_cl1 = self.cluster_model(data2.to(self.device), length2.to(self.device), id_mask2.to(self.device),timestamp2.to(self.device),self.momentum_model)
                p_select = p[i*self.args.bs:(i+1)*self.args.bs]

                if self.args.ncein:
                    nce_loss_in = criterion.infonce(head_in1, head_in, 0.5)
                    loss+=nce_loss_in
                if self.args.ncecl:
                    nce_loss_cl = criterion.infonce(head_cl1, head_cl, 1)
                    loss+=nce_loss_cl

               
                if self.args.kl:
                    kl_loss = criterion.clusteringLoss(self.cluster_model.clusterlayer, context, p_select, self.device, self.device, q)
                    kl_loss1 = criterion.clusteringLoss(self.cluster_model.clusterlayer, context1, p_select, self.device, self.device, q1)
                    kl_loss = kl_loss + kl_loss1
                    loss+= kl_loss * self.args.gamma 

             

                self.cluster_model.zero_grad()
                loss.backward()
                model_optim.step()
                if lr_scheduler != None:
                    lr_scheduler.step(nce_loss_cl)
              


                if self.momentum>0:
                    momentum=self.momentum
                    for param, param_m in zip(self.cluster_model.bert.parameters(), self.momentum_model.parameters()):
                        param_m.data = param_m.data * momentum + param.data * (1. - momentum)

                print("Epoch: %d, batch: %d, loss: %f, mlm_loss: %f, kl_loss: %f, nce_loss_cl: %f, nce_loss_in: %f, lr1: %.8f, lr2: %.8f"
                        %(ep+1, i, loss.cpu(), mlm_loss.cpu() if self.args.mlm else 0\
                            , kl_loss.cpu() if self.args.kl else 0\
                                , nce_loss_cl.cpu() if self.args.ncecl else 0\
                                    , nce_loss_in.cpu() if self.args.ncein else 0\
                                        ,model_optim.state_dict()['param_groups'][0]['lr'],model_optim.state_dict()['param_groups'][1]['lr']))
                
            

            print('Best Epoch: ', best_epoch)
            print('Best Acc: ', best_acc)
            print('Best Nmi: ', best_nmi)
            print('Best Ari: ', best_ari)

        # 保存训练后的 轨迹表征
        # np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_best_vec',best_context)
        # np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_best_pre_label',best_pre_label)
        # np.save('./result/'+self.args.dataset+'_'+self.args.embedding+'_best_true_label',best_true_label)
        # if ep == 0:
        #         np.save('./result/timestamp.npy',timestampList)

    def init_clusterlayer(self):
        self.cluster_model.eval()
       
        vecs = []
        y = []
        data_dict = {'traj':[],'time':[],'label':[]}
        with torch.no_grad():
            for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,label) in enumerate(self.cluster_loader):
                context, q_i, head_in, head_cl = self.cluster_model(input_ids_o.to(self.device), lengths.to(self.device), id_mask.to(self.device),timestamp_o.to(self.device),None)
                # context, q_i, head_in, head_cl = self.cluster_model(input_ids.to(self.device), lengths.to(self.device), id_mask.to(self.device),timestamp.to(self.device))
                data_dict['traj'].append(input_ids_o.cpu().data)
                data_dict['time'].append(timestamp_o.cpu().data)
                data_dict['label'].append(label.cpu().data)

                vecs.append(context.cpu().data)
                y.append(label)
            vecs = torch.cat(vecs).cpu().numpy()
            y = torch.cat(y).cpu().numpy()

            # data_dict['traj'] = torch.cat(data_dict['traj']).cpu().numpy().tolist()
            # data_dict['time'] = torch.cat(data_dict['time']).cpu().numpy().tolist()
            # data_dict['label'] = torch.cat(data_dict['label']).cpu().numpy().tolist()
            # df = pd.DataFrame(data_dict)
            # df.to_csv('my_data2.csv', index=False)

            # torch.save(self.cluster_model.bert.state_dict(), 'before_initcluster.pt')
            if True:
                print('-'*20+'init cluster layer'+'-'*20)
                kmeans = KMeans(n_clusters=self.n_cluster, n_init=10,
                                    random_state=58).fit(vecs)
                y_pred = kmeans.fit_predict(vecs)
                acc = cluster_acc(y, y_pred,True)
                nmi = nmi_score(y, y_pred)
                ari = ari_score(y, y_pred)
                print('KMeans\tAcc: {0:.4f}\tnmi: {1:.4f}\tari: {2:.4f}'.format(acc, nmi, ari))
                self.cluster_model.clusterlayer.clusters.data = torch.Tensor(
                        kmeans.cluster_centers_).to(self.device)
                # 保存KMeans模型
                # with open(self.args.kname+".pkl", "wb") as file:
                #     pickle.dump(kmeans, file)
                # np.save(self.args.kname+'_vec',vecs )
                # np.save(self.args.kname+'_y',y )

                # torch.save(self.cluster_model.clusterlayer.clusters.data, './pth_model/init_clusterLayer_epoch_%s.pth' % str(self.args.pretrain_epoch))
                # torch.save({'model': self.cluster_model}, './pth_model/init_clusterLayer_epoch_%s.pth' % str(self.args.pretrain_epoch))
            # torch.save(self.cluster_model.bert.state_dict(), 'after_initcluster.pt')
        
        self.cluster_model.train()

        return vecs, kmeans.cluster_centers_ , acc
    
    def eta(self,setting):
        # self.eta_loader,self.eta_train_loader,self.eta_valid_loader,self.eta_test_loader = self.data_provider.get_loader(flag='eta',args = self.args)
        
        if self.args.dataset not in  ['cdr_sep','qdhalf_sep']:
            loader,self.eta_train_loader,self.eta_valid_loader,self.eta_test_loader = self.data_provider.get_loader(flag='eta',args = self.args)
            # if self.args.task == 'cluster':
            #     self.cluster_loader = loader

        else:
            self.eta_train_loader,_,_,_ = self.train_data_loader.get_loader(flag='eta',args = self.args)
            self.eta_test_loader ,_,_,_ =  self.test_data_loader.get_loader(flag='eta',args = self.args)
            self.eta_valid_loader,_,_,_ = self.valid_data_loader.get_loader(flag='eta',args = self.args)
        
        criterion = self._select_criterion()
        eta_criterion = torch.nn.MSELoss(reduction='none')
        model_optim,lr_scheduler = self._select_optimizer('eta') 

        best_mae = float('inf')
        best_model_path =  'best_model_'+'proposed' if self.args.embedding == 'both' else 'bert' +'.pth'
        for ep in range(self.args.epoch):
            self.ETA_model.train()
            for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,duration) in enumerate(self.eta_train_loader):

                model_optim.zero_grad()

                # mlm_pred,mlm_temporal = self.ETA_model.bert(input_ids.to(self.device), masked_pos.to(self.device),timestamp_o.to(self.device))
                mlm_pred,mlm_temporal = self.ETA_model.bert(input_ids_o.to(self.device), masked_pos.to(self.device),timestamp_o.to(self.device))
                mlm_loss_lm = criterion.Cross_Entropy_Loss(mlm_pred.view(-1, self.vocab_size).to(self.device), masked_tokens.view(-1).to(self.device))  # for masked LM
                mlm_loss = (mlm_loss_lm.float()).mean().cpu()

                prediction = self.ETA_model(input_ids_o.to(self.device), lengths.to(self.device), id_mask.to(self.device),timestamp_o.to(self.device),None)
                prediction = prediction.squeeze()
                eta_loss = eta_criterion(prediction.cpu().float(),duration.cpu().float()).mean().cpu()
                loss = mlm_loss + eta_loss
                # loss = eta_loss
                print('ETA Epoch: %d, batch: %d,: loss: %f, mlm_loss: %f, eta_loss: %f'%((ep+1),i,loss.cpu(),mlm_loss.cpu(),eta_loss.cpu()))
                # print('ETA Epoch: %d, batch: %d,: loss: %f,  eta_loss: %f'%((ep+1),i,loss.cpu(),eta_loss.cpu()))


                loss.backward()
                model_optim.step()
                
                # iter_pred = prediction.cpu().float()
                # iter_true = duration.cpu().float()
                # if i  == 0:
                #     all_pred,all_true = iter_pred.clone().detach(),iter_true.clone().detach()
                # else:
                #     all_pred = torch.cat((all_pred,iter_pred.clone().detach()))
                #     all_true = torch.cat((all_true,iter_true.clone().detach()))

            print('Train: prediction',prediction.detach().cpu().numpy()[0],'true duration',duration.detach().cpu().numpy()[0])

            mae = self.eta_test(self.eta_valid_loader)
            if mae <= best_mae:
                best_mae = mae
                torch.save({'model': self.ETA_model,'best_mae':best_mae},best_model_path )
            # mae = masked_mae_torch(all_pred.cpu().float(),all_true.cpu().float())
            # rmse = masked_rmse_torch(all_pred.cpu().float(),all_true.cpu().float())
            # mape = masked_mape_torch(all_pred.cpu().float(),all_true.cpu().float())
            # r2 = r2_score_torch(all_pred.cpu().float(),all_true.cpu().float())
            # evar = explained_variance_score_torch(all_pred.cpu().float(),all_true.cpu().float())
            
            # print('test MAE: %f, RMSE: %f, MAPE: %f, R2: %f, EVAR: %f'%(mae,rmse,mape,r2,evar))
        # self.ETA_model = self.load_weight(self.ETA_model,best_model_path)
        self.eta_test(self.eta_test_loader)
        return 
    
    def eta_test(self,loader):
        self.ETA_model.eval()
        with torch.no_grad():
            for i, (input_ids, masked_tokens, masked_pos,input_ids_o, timestamp, time_masked_tokens, timestamp_o, id_mask, lengths,duration) in enumerate(loader):
                    prediction = self.ETA_model(input_ids_o.to(self.device), lengths.to(self.device), id_mask.to(self.device),timestamp_o.to(self.device),None)
                    prediction = prediction.squeeze()
                    
                    iter_pred = prediction.cpu().float()
                    iter_true = duration.cpu().float()
                    if i  == 0:
                        all_pred,all_true = iter_pred.clone().detach(),iter_true.clone().detach()
                    else:
                        all_pred = torch.cat((all_pred,iter_pred.clone().detach()))
                        all_true = torch.cat((all_true,iter_true.clone().detach()))

            print('Test: prediction',prediction.detach().cpu().numpy()[0],'true duration',duration.detach().cpu().numpy()[0])

            mae = masked_mae_torch(all_pred.cpu().float(),all_true.cpu().float())
            rmse = masked_rmse_torch(all_pred.cpu().float(),all_true.cpu().float())
            mape = masked_mape_torch(all_pred.cpu().float(),all_true.cpu().float())
            r2 = r2_score_torch(all_pred.cpu().float(),all_true.cpu().float())
            # r2_adjusted = r2_score_adjust_torch(all_pred.cpu().float(),all_true.cpu().float())
            evar = explained_variance_score_torch(all_pred.cpu().float(),all_true.cpu().float())
            
            print('test MAE: %f, RMSE: %f, MAPE: %f, R2: %f, EVAR: %f'%(mae,rmse,mape,r2,evar))
            # print('test MAE: %f, RMSE: %f, MAPE: %f, R2: %f, R2_ad: %f, EVAR: %f'%(mae, rmse, mape, r2, r2_adjusted, evar))
        return mae
    def sim(self,seeting):
        eta_loader,_,_,_ = self.data_provider.get_loader(flag='sim',args = self.args)
        pred_list,id_list = [],[]
        self.sim_model.eval()
        with torch.no_grad():
            for i, (input_ids, masked_tokens, masked_pos, timestamp, time_masked_tokens,id_mask,lengths,traj_id) in enumerate(eta_loader):
                predictions = self.sim_model(input_ids.to(self.device), lengths.to(self.device), id_mask.to(self.device), timestamp.to(self.device), None)

                pred_list.append(predictions.cpu().numpy())
                id_list.append(traj_id.cpu().numpy())
            pred_list = np.concatenate(pred_list)  # (n, dim)
            id_list = np.concatenate(id_list) 

            if self.args.dataset.split('_')[-1] == 'q':
                ids_path = 'query_id_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
                pred_list_path = 'query_pred_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
            elif self.args.dataset.split('_')[-1] == 'd':
                ids_path = 'detour_id_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
                pred_list_path = 'detour_pred_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
            elif self.args.dataset.split('_')[-1] == 'o':
                ids_path = 'database_id_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
                pred_list_path = 'database_pred_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
            
            elif self.args.dataset.split('_')[-2] == 'q':
                ids_path = 'query_id_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
                pred_list_path = 'query_pred_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
            elif self.args.dataset.split('_')[-2] == 'd':
                ids_path = 'detour_id_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
                pred_list_path = 'detour_pred_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
            elif self.args.dataset.split('_')[-2] == 'o':
                ids_path = 'database_id_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)
                pred_list_path = 'database_pred_{}_{}_{}.npy'.format(self.args.modelname,self.args.dataset.split('_')[0],self.args.d_model)

            # elif self.args.dataset.split('_')[1] == 'b':
            #     ids_path = 'bigdatabase_id_{}_{}_{}.npy'.format(self.args.modelname,'cdr',self.args.d_model)
            #     pred_list_path = 'bigdatabase_pred_{}_{}_{}.npy'.format(self.args.modelname,'cdr',self.args.d_model)

            pred_list_path = 'simTask/data/' + pred_list_path
            ids_path = 'simTask/data/' + ids_path
            print(pred_list_path)
            print(ids_path)
            np.save(pred_list_path, pred_list)  # len=b
            np.save(ids_path, id_list)
        return 
    


    
    def load_weight(self,model,model_path):
        print('load model',model_path)
        # model = torch.load(model_path)
        if type(torch.load(model_path,map_location=self.device)).__name__=='dict':
            pretrained_dict = torch.load(model_path,map_location=self.device)['model'].state_dict()
        else: pretrained_dict = torch.load(model_path,map_location=self.device).state_dict()
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        return model



    def _acquire_device(self):
        if self.args.use_gpu:
          
            device = torch.device('cuda:{}'.format(self.args.device))
            print('Use GPU: cuda:{}'.format(self.args.device))
        else:
            device = torch.device('cpu')
        return device

    # 初始化函数
    def init_weights(self,m):
        # if type(m) == nn.Linear:
        #     # init.xavier_uniform_(m.weight)
        #     # m.bias.data.fill_(0)
        #     nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            # m.bias.data.fill_(0)
        # elif type(m) == nn.Embedding:#3
            # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') #2
            # init.uniform_(m.weight, a=-0.05, b=0.05) #_init
        if type(m)==nn.Embedding: # 4
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if type(m) == nn.Linear:
            if self.args.kname =='c':
                nn.init.xavier_uniform_(m.weight) #c
            if self.args.kname =='d':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu') #d
            if self.args.kname =='e':
                nn.init.uniform_(m.weight, a=-0.05, b=0.05) #e
