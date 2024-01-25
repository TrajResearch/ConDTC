import pandas as pd
import argparse
import pickle
import os
parser = argparse.ArgumentParser()
# parser.add_argument('--data', default='portoTimeNoise0420', type=str, help='qd,cdr,porto,qdhalf')
parser.add_argument('--data', default='qdTimeNoise0424', type=str, help='qd,cdr,porto,qdhalf')

args = parser.parse_args()


class Vocab_setup():
    def __init__(self,args):
        self.mp_lst = []
        self.dataname = args.data

        if self.dataname == 'cdr':
            self.pathlist = ['data/cdr_format/data_k3.h5']
        elif self.dataname == 'portoTimeNoise0420':
            self.pathlist = ['data/portoTimeNoise0420/data_k3.h5']
        elif self.dataname == 'qdhalf':
            self.pathlist = ['data/qdhalf_train/data_k3.h5','data/qdhalf_test/data_k3.h5','data/qdhalf_valid/data_k3.h5']
        if self.dataname == 'cdr_sep':
            self.pathlist = ['data/cdr_d/data_k3.h5','data/cdr_q/data_k3.h5','data/cdr_o/data_k3.h5']
        else:
            self.pathlist = ['data/'+self.dataname+'/data_k3.h5']
        self.picklePath = os.path.join('mid_data',self.dataname+'_vocab.pkl')

        if not os.path.exists('mid_data'):
            os.mkdir('mid_data')


    def merge_data(self,pathlist):
        df  = pd.DataFrame()
        for each in pathlist:
            subdf = pd.read_hdf(each)
            df = pd.concat([df,subdf],ignore_index=True)
        return df

    def add_vocab(self,trajectory):
        trajectory = trajectory.split()
        for each in trajectory:
            if each != '[PAD]':
                self.mp_lst.append(int(each))

    def process(self):

        df = self.merge_data(self.pathlist)
        trajectoryList = df['trajectory'].values.tolist()
        
        for each in trajectoryList:
            self.add_vocab(each)
        vocab_list = list(set(self.mp_lst))
        vocab_list.sort()

      
        vocab_map = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for i,v in  enumerate(vocab_list):
            vocab_map[str(v)] = i + 4
        print('vocab size:',len(vocab_list),len(vocab_map))
        
        with open(self.picklePath, 'wb') as f:
            pickle.dump(vocab_map, f)

if __name__ == '__main__':
    voc = Vocab_setup(args)
    voc.process()