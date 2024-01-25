import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
import torch
import os
import json
import time
import random
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--modelname', default='proposed', type=str, help='bert / proposed')
parser.add_argument('--sim_mode', default='most', type=str, help='knn / most')
parser.add_argument('--dataset', default='porto1031', type=str, help='cdr,qdhalf,porto1031')

args = parser.parse_args()
modelname = args.modelname # bert / proposed /st2vec
d_model = '256'
dataset = args.dataset
sim_mode = args.sim_mode # knn / most

print('use model: {} sim_mode: {} dataset: {}'.format(modelname,sim_mode,dataset))
topk = [1,5,10]
sim_select_num = 5
result = {}

# query_id_path = './data/query_id_{}_{}_{}.npy'.format(modelname, dataset, d_model)
# detour_id_path = './data/detour_id_{}_{}_{}.npy'.format(modelname, dataset, d_model)
# database_id_path = './data/database_id_{}_{}_{}.npy'.format(modelname, dataset, d_model)
# database_id_path = './data/bigdatabase_id_{}_{}_{}.npy'.format(modelname, dataset, d_model)


query_pred_path = './data/query_pred_{}_{}_{}.npy'.format(modelname, dataset, d_model)
detour_pred_path = './data/detour_pred_{}_{}_{}.npy'.format(modelname, dataset, d_model)
database_pred_path = './data/database_pred_{}_{}_{}.npy'.format(modelname, dataset, d_model)
# database_pred_path = './data/bigdatabase_pred_{}_{}_{}.npy'.format(modelname, dataset, d_model)
print(query_pred_path)
print(detour_pred_path)
print(database_pred_path)



euclidean_path = './data/evaluate_cache/euclidean_{}_{}_{}_most.npy' \
            .format(modelname, dataset, d_model)
euclidean_index_path = './data/evaluate_cache/euclidean_index_{}_{}_{}_most.npy' \
            .format(modelname, dataset, d_model)
evaluate_res_path = './data/evaluate_cache/evaluate_res_{}_{}_{}_{}.json' \
            .format(modelname, dataset, d_model, sim_mode)
qgis_res_path = './data/evaluate_cache/qgis_res_{}_{}_{}_{}.csv' \
            .format(modelname, dataset, d_model, sim_mode)

euclidean_path_truth = './data/evaluate_cache/euclidean_{}_{}_{}_truth_knn.npy'.format(modelname, dataset, d_model) 
euclidean_path_pred = './data/evaluate_cache/euclidean_{}_{}_{}_pred_knn.npy'.format(modelname, dataset, d_model) 
knn_hit_path = './data/evaluate_cache/knn_hit_{}_{}_{}.npy'.format(modelname, dataset, d_model)

true_id_list = []#np.load('/home/xy/traj/cdr/true_id.npy')
#most
# rootpath = '/home/xy/d/d/mySTART/libcity/cache/202302192/evaluate_cache/'
# query_id_path = rootpath + '202302192_query_ids_LinearSim_cdr_256.npy'
# detour_id_path = rootpath + '202302192_detour_ids_LinearSim_cdr_256.npy'
# database_id_path = rootpath + '202302192_database_ids_LinearSim_cdr_256.npy'

# query_pred_path = rootpath + '202302192_query_vec_LinearSim_cdr_256.npy'
# detour_pred_path = rootpath + '202302192_detour_vec_LinearSim_cdr_256.npy'
# database_pred_path = rootpath + '202302192_database_vec_LinearSim_cdr_256.npy'


# evaluate_res_path = rootpath + 'res.json'
# qgis_res_path = rootpath + 'qgis.csv'
# euclidean_path =  rootpath + 'euclidean_path.npy'
# euclidean_index_path = rootpath + 'euclidean_index_path.npy'



# # knn
# rootpath = '/home/xy/d/d/mySTART/libcity/cache/202302225/evaluate_cache/'
# query_id_path = rootpath + '202302225_query_ids_LinearSim_cdr_256.npy'
# detour_id_path = rootpath + '202302225_detour_ids_LinearSim_cdr_256.npy'
# database_id_path = rootpath + '202302225_database_ids_LinearSim_cdr_256.npy'

# query_pred_path = rootpath + '202302225_query_vec_LinearSim_cdr_256.npy'
# detour_pred_path = rootpath + '202302225_detour_vec_LinearSim_cdr_256.npy'
# database_pred_path = rootpath + '202302225_database_vec_LinearSim_cdr_256.npy'

# ## result file
# evaluate_res_path = rootpath + 'res.json'
# qgis_res_path = rootpath + 'qgis.csv'
# euclidean_path_truth = rootpath + 'euclidean_truth_cdr_256_knn.npy'
# euclidean_path_pred =  rootpath + 'euclidean_pred_cdr_256_knn.npy'
# knn_hit_path = rootpath + 'knn_hit_cdr_256.npy'


# query_id_list = np.load(query_id_path)
# detour_id_list = np.load(detour_id_path)
# database_id_list = np.load(database_id_path)

query_pred_list = np.load(query_pred_path)
print(query_pred_list.shape)
detour_pred_list = np.load(detour_pred_path)
database_pred_list = np.load(database_pred_path)


def evaluate_most_sim():
    t1, t2, t3 = 0, 0, 0
    # if os.path.exists(euclidean_path):
    #     eul_res = np.load(euclidean_path)
    if False:pass
    else:
        start_time = time.time()
        database_all = np.concatenate([detour_pred_list, database_pred_list], axis=0)
        # database_all = np.concatenate([database_pred_list,detour_pred_list], axis=0)
        eul_res = euclidean_distances(query_pred_list, database_all)  # (a, a+b)
        t1 = time.time() - start_time
        # self._logger.info('Euclidean_distances cost time {}.'.format(t1))
        print('Euclidean_distances cost time {}.'.format(t1))
        np.save(euclidean_path, eul_res)
        # self._logger.info('Euclidean_distances is saved at {}, shape={}.'.format(self.euclidean_path, eul_res.shape))
        print('Euclidean_distances is saved at {}, shape={}.'.format(euclidean_path, eul_res.shape))
    # if os.path.exists(euclidean_index_path):
        # sorted_eul_index = np.load(euclidean_index_path)
    if False:pass
    else:
        start_time = time.time()
        sorted_eul_index = eul_res.argsort(axis=1)
        t2 = time.time() - start_time
        # self._logger.info('Sorted euclidean_index cost time {}.'.format(t2))
        print('Sorted euclidean_index cost time {}.'.format(t2))
        np.save(euclidean_index_path, sorted_eul_index)
        print('Sorted euclidean_index is saved at {}, shape={}.'.format(euclidean_index_path, sorted_eul_index.shape))
        # self._logger.info('Sorted euclidean_index is saved at {}, shape={}.'.format(self.euclidean_index_path, sorted_eul_index.shape))

    start_time = time.time()
    total_num = eul_res.shape[0]
    hit = {}
    for k in topk:
        hit[k] = 0
    rank = 0
    rank_p = 0.0
    for i in range(total_num):
        rank_list = list(sorted_eul_index[i])
        rank_index = rank_list.index(i)
        # if i == 0: print(rank_list[:12],'i',i,'rank index',rank_index)
        # if rank_list[0] != i and rank_list[0] != i +10000:print('spatial',i,rank_list[:5])
        
        # with open('most_rank.txt','a') as f:
        #     f.write(str(rank_list[:12])+' i '+str(i)+' rank index '+str(rank_index)+'\n')

        # rank_index is start from 0, so need plus 1
        rank += (rank_index + 1)
        rank_p += 1.0 / (rank_index + 1)
        for k in topk:
            if i in sorted_eul_index[i][:k]:
                hit[k] += 1

    result['MR'] = rank / total_num
    result['MRR'] = rank_p / total_num
    for k in topk:
        result['HR@{}'.format(k)] = hit[k] / total_num
    t3 = time.time() - start_time
    # self._logger.info("Evaluate cost time is {}".format(t3))
    # self._logger.info("Evaluate result is {}".format(self.result))
    print("Evaluate cost time is {}".format(t3))
    print("Evaluate result is {}".format(result))
    json.dump(result, open(evaluate_res_path, 'w'), indent=4)
    print('Evaluate result is saved at {}'.format(evaluate_res_path))
    print("Total cost time is {}".format(t1 + t2 + t3))
    # self._logger.info('Evaluate result is saved at {}'.format(self.evaluate_res_path))
    # self._logger.info("Total cost time is {}".format(t1 + t2 + t3))

    kmax = max(topk)
    select_index = np.arange(len(sorted_eul_index))
    random.shuffle(select_index)
    # select_index = select_index[:sim_select_num]
    select_index = select_index

    # output = []
    # for i in select_index:
    #     # query
    #     output.append([str(i) + '-query', query_id_list[i], i])

    #     # detour
    #     output.append([str(i) + '-detour', detour_id_list[i], i])

    #     # tok-sim
    #     for ind, d in enumerate(sorted_eul_index[i, 0:kmax]):
    #         index_out = str(i) + '-' + str(ind)
    #         if d >= len(detour_id_list):  # 大数据库中的轨迹
    #             d -= len(detour_id_list)
    #             # output.append([index_out, self.database_id_list[d],
    #             #                self.database_wkt[str(self.database_id_list[d])], i])
    #             output.append([index_out, database_id_list[d], i])
    #         else:
    #             if d == i:
    #                 index_out += '-find'
    #             # output.append([index_out, self.detour_id_list[d],
    #             #                self.detour_wkt[str(self.detour_id_list[d])], i])
    #             output.append([index_out, detour_id_list[d], i])
    # # output = pd.DataFrame(output, columns=['index', 'id', 'wkt', 'class'])
    # output = pd.DataFrame(output, columns=['index', 'id', 'class'])

    # output.to_csv(qgis_res_path, index=False)
    # print(result)
    # return result

def evaluate_knn_sim():
    topk = [5]
    assert len(topk) == 1
    topk = topk[0]  # list to int
    t1, t2, t3 = 0, 0, 0
    # if os.path.exists(euclidean_path_truth) and os.path.exists(euclidean_path_pred):
    #     eul_res_query = np.load(euclidean_path_truth)
    #     eul_res_detour = np.load(euclidean_path_pred)
    if False:pass
    else:
        start_time = time.time()
        # database_all = np.concatenate([detour_pred_list, database_pred_list], axis=0)
        # eul_res_query  = euclidean_distances(query_pred_list, database_all)  # (a, a+b)
        eul_res_query = euclidean_distances(query_pred_list, database_pred_list)  # (a, b)
        t1 = time.time() - start_time
        # self._logger.info('Euclidean_distances Truth cost time {}.'.format(t1))
        print('Euclidean_distances Truth cost time {}.'.format(t1))
        np.save(euclidean_path_truth, eul_res_query)
        # self._logger.info('Euclidean_distances Truth is saved at {}, shape={}.'.format(
        #     self.euclidean_path_truth, eul_res_query.shape))
        print('Euclidean_distances Truth is saved at {}, shape={}.'.format(
            euclidean_path_truth, eul_res_query.shape))

        start_time = time.time()
        # eul_res_detour  = euclidean_distances(detour_pred_list, database_all)  # (a, a+b)
        eul_res_detour = euclidean_distances(detour_pred_list, database_pred_list)  # (a, b)
        t1 = time.time() - start_time
        # self._logger.info('Euclidean_distances Pred cost time {}.'.format(t1))
        print('Euclidean_distances Pred cost time {}.'.format(t1))
        np.save(euclidean_path_pred, eul_res_detour)
        print('Euclidean_distances Pred is saved at {}, shape={}.'.format(euclidean_path_pred, eul_res_detour.shape))
        # self._logger.info('Euclidean_distances Pred is saved at {}, shape={}.'.format(
        #     self.euclidean_path_pred, eul_res_detour.shape))


    eul_res_query = torch.from_numpy(eul_res_query)
    start_time = time.time()
    _, eul_res_query_index = torch.topk(eul_res_query, topk, dim=1, largest=False)
    eul_res_query_index = eul_res_query_index.cpu().numpy()
    # sorted_eul_query_index = eul_res_query.argsort(axis=1)
    # eul_res_query_index = sorted_eul_query_index
    t2 = time.time() - start_time
    # self._logger.info('Sorted euclidean_index Truth cost time {}.'.format(t2))
    # print('Sorted euclidean_index Truth cost time {}.'.format(t2))

    eul_res_detour = torch.from_numpy(eul_res_detour)
    start_time = time.time()
    _, eul_res_detour_index = torch.topk(eul_res_detour, topk, dim=1, largest=False)
    eul_res_detour_index = eul_res_detour_index.cpu().numpy()
    
    # sorted_eul_detour_index = eul_res_detour.argsort(axis=1)
    # eul_res_detour_index = sorted_eul_detour_index
    t2 = time.time() - start_time
    # self._logger.info('Sorted euclidean_index Pred cost time {}.'.format(t2))
    print('Sorted euclidean_index Pred cost time {}.'.format(t2))

    start_time = time.time()
    total_num = eul_res_query_index.shape[0]
    hit = []
    qhit , dhit = [], []
    for i in range(total_num):
        query_k = set(eul_res_query_index[i].tolist())
        # query_k = eul_res_query_index[i][:topk]
        detour_k = eul_res_detour_index[i].tolist()
        # detour_k = eul_res_detour_index[i][:topk]
        cnt = qcnt= dcnt = 0
        for ind in detour_k:
            if ind in query_k:
                cnt += 1
        
        hit.append(cnt)

        for idx in true_id_list[i]:
            if idx in detour_k:
                dcnt +=1
            if idx in query_k:
                qcnt += 1
        dhit.append(dcnt)
        qhit.append(qcnt)

    np.save(knn_hit_path, np.array(hit))
    result['Precision'] = (1.0 * sum(hit)) / (total_num * topk)
    # result['q_Precision'] = (1.0 * sum(qhit)) / (total_num * topk)
    # result['d_Precision'] = (1.0 * sum(dhit)) / (total_num * topk)

    t3 = time.time() - start_time
    # self._logger.info("Evaluate cost time is {}".format(t3))
    # self._logger.info("Evaluate result is {}".format(self.result))
    print("Evaluate cost time is {}".format(t3))
    json.dump(result, open(evaluate_res_path, 'w'), indent=4)
    print('Evaluate result is saved at {}'.format(evaluate_res_path))
    print("Total cost time is {}".format(t1 + t2 + t3))
    # self._logger.info('Evaluate result is saved at {}'.format(self.evaluate_res_path))
    # self._logger.info("Total cost time is {}".format(t1 + t2 + t3))

    select_index = np.arange(total_num)
    random.shuffle(select_index)
    select_index = select_index[:sim_select_num]
    output = []
    # for i in select_index:
    #     # query
    #     output.append([str(i) + '-query', query_id_list[i], i])

    #     # detour
    #     output.append([str(i) + '-detour', detour_id_list[i], i])

    #     # query topk-sim
    #     for ind in eul_res_query_index[i].tolist():
    #         output.append([str(i) + '-query-' + str(ind), database_id_list[ind], i])
    #     # detour topk-sim
    #     for ind in eul_res_detour_index[i].tolist():
    #         output.append([str(i) + '-detour-' + str(ind), database_id_list[ind], i])
    # output = pd.DataFrame(output, columns=['index', 'id', 'class'])

    # output.to_csv(qgis_res_path, index=False)
    print(result)
    return result


if __name__ == "__main__":
    if sim_mode == 'most':
        evaluate_most_sim()
    elif sim_mode == 'knn':
        evaluate_knn_sim()