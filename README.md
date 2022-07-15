# FedDA

'''
python Fed_D_HGN.py --prefix data_Amazon --dataset Amazon --batch-size 1028 --patience 40 --num-heads 3 --do-num 17 --communication-round 50 --c_epoch 50 --valEachStep --FedAvg --aggregateAll --active-rate 1

python Fed_D_HGN.py --prefix data_Amazon --dataset Amazon --batch-size 1028 --patience 40 --num-heads 3 --do-num 17 --communication-round 50 --c_epoch 50 --valEachStep --removeClient --aggregateAll --partiallyReturn --active-rate 1

python Fed_D_HGN.py --prefix data_Amazon --dataset Amazon --batch-size 1028 --patience 40 --num-heads 3 --do-num 17 --communication-round 50 --c_epoch 50 --valEachStep --removeClient --aggregateAll --partiallyReturn --explore --active-rate 0.667
'''

## running environment
* torch 1.6.0 cuda 10.1
* dgl 0.4.3 cuda 10.1
* networkx 2.3
* scikit-learn 0.23.2
* scipy 1.5.2