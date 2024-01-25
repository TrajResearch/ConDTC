import argparse
from exp2 import Exp
import time


parser = argparse.ArgumentParser()
parser.add_argument('--device', default=0, type=int, help='train device')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--epoch', default=1, type=int, help='epoch size')
parser.add_argument('--pretrain_epoch', default=2, type=int, help='pretrain epoch size')
parser.add_argument('--loss', default='loss', type=str, help='loss fun')
parser.add_argument('--datalen', default=5, type=int, help='datalen')
parser.add_argument('--max_con', default=2, type=int, help=' max con mask len')
parser.add_argument('--dataset', default="qdTimeNoise0424", type=str, help='mobile ,porto,qd,cdr')
parser.add_argument('--embed', default='256', type=str, help='loc id embed')
parser.add_argument('--d_model', default=256, type=int, help='embed size')
parser.add_argument('--head', default=2, type=int, help='multi head num')
parser.add_argument('--layer', default=2, type=int, help='layer')
parser.add_argument('--remask', default=0, type=int, help='remask')
parser.add_argument('--lr', default=1e-4, type=float, help='lr')
parser.add_argument('--prelr', default=1e-4, type=float, help='pretrain lr')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gamma', default=2, type=float, help='kl loss ratio')
parser.add_argument('--beta', default=2, type=float, help='near kl loss ratio')
parser.add_argument('--embedding', default='both', type=str, help='position,temporal,both')
parser.add_argument('--ncecl', default=1, type=float, help='ncecl')
parser.add_argument('--ncein', default=1, type=float, help='ncein')
parser.add_argument('--kl', default=1, type=float, help='kl loss')
parser.add_argument('--mlm', default=1, type=float, help='mlm loss')
parser.add_argument('--momentum', default=0.995, type=float, help='momentum ratio, 0 off, >0 on')
parser.add_argument('--droprate', default=0.1, type=float, help='trajectory drop rate')
parser.add_argument('--dataaug', default='', type=str, help='data augmentation method:drop,offsetTraj,offsetTime,inserV')
parser.add_argument('--loadmodel', default='', type=str, help='porto,mobile,qd')
parser.add_argument('--task', default='cluster', type=str, help='cluster,eta,sim')
parser.add_argument('--modelname', default='bert', type=str, help='cluster,eta,sim')
parser.add_argument('--loadpickle', default=0, type=int, help='other True, 0 False')
parser.add_argument('--tRate', default=1, type=float, help='trajectory drop rate')
parser.add_argument('--timesize', default=1440, type=int, help='5760,1440,2880')
parser.add_argument('--seed', default=0, type=int, help='5760,1440,2880')
parser.add_argument('--kname', default='a', type=str, help='a,b,c,anything')
parser.add_argument('--halfD', default=0, type=int, help='0,1')
parser.add_argument('--freezeTemb', default=0, type=int, help='0,1')
parser.add_argument('--freezeLemb', default=0, type=int, help='0,1')
parser.add_argument('--lrsep', default=0, type=int, help='0,1')
parser.add_argument('--trainP', default=0.8, type=float, help='train set percent')
parser.add_argument('--wrongK', default=0, type=int, help='0,1')


args = parser.parse_args()


setting = 'data_{}_{}_dmodel_{}_head_{}_layer_{}_loss_{}_bs_{}_pretrainEpoch_{}_epoch_{}_gamma_{}_beta_{}_momentum_{}'.format(
                args.dataset,
                args.datalen,
                args.d_model,
                args.head,
                args.layer,
                args.loss,
                args.bs,
                args.pretrain_epoch,
                args.epoch,
                args.gamma,
                args.beta,
                args.momentum
                )

exp = Exp(args)  # set experiments
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
st = time.time()
if args.task == 'cluster':
    exp.pretrain(setting)
    exp.train(setting)
elif args.task == 'eta':
    # exp.pretrain(setting)
    exp.eta(setting)
elif args.task == 'sim':
    exp.sim(setting)
elif args.task == 'pretrain':
    exp.pretrain(setting)
print('spent time :',time.time()-st,' s')