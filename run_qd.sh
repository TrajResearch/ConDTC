python run2.py --epoch 1 --pretrain_epoch 2 \
--dataset qdTimeNoise0424 --lr 5e-5 --prelr 15e-5  \
--kl 1 --ncecl 0 --ncein 0 --mlm 1 --embedding both --device 0 \
--momentum 0 --task cluster --datalen 5 --dataaug 'drop' --wrongK 12 \
--bs 32 --max_con 1 --tRate 0 --seed 19721013 --lrsep 1 --trainP 0.8 --gamma 2  \
> ./qdTimeNoise0424_k12.txt

