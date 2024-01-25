# ConDTC



## cluster task example

 could run run_qd bash file.

```
./run_qd.sh
```

also could run the following command

```
python run2.py --epoch 1 --pretrain_epoch 2 \
--dataset qdTimeNoise0424 --lr 5e-5 --prelr 15e-5  \
--embedding both --device 0 \
--task cluster  --dataaug 'drop'  \
--bs 32 --max_con 1 --tRate 0  --trainP 0.8 --gamma 2
```



## similarity task example

first run run_sim script to generate trajectory vectors

```
./run_sim.sh
```

second go to simTask folder run simTask.py to calculate similarity

```
cd simTask
python simTask.py
```



## ETA task example

```
python run2.py --task eta --dataset cdr
```

