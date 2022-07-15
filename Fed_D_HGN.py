import time
import argparse

from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data
from myGNN import sepGAT
import dgl
import random

# torch.manual_seed(0)


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def run_model(args, do_id, inPara=None):
    print('='*89)
    # prepare data
    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.prefix, args.dataset, do_id, ifFed=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
        
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    
    edge2type = {}
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u, v)] = k
    for i in range(dl.nodes['total']):
        if (i, i) not in edge2type:
            edge2type[(i, i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u, v in zip(*dl.links['data'][k].nonzero()):
            if (v, u) not in edge2type:
                edge2type[(v, u)] = k + 1 + len(dl.links['count'])

    g = dgl.DGLGraph(adjM + (adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    e_count = [0 for _ in range(len(dl.links['count'])*2+1)]
    count = 0
    current_type = 0
    index = 0
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        type = edge2type[(u, v)]
        e_feat.append([type, index])
        index += 1
        e_count[type] += 1

    res_2hop = defaultdict(float)
    
    # prepare model
    train_pos, valid_pos = dl.get_train_valid_pos()#edge_types=[test_edge_type])
    num_classes = args.hidden_dim
    heads = [args.num_heads] * args.num_layers + [args.num_heads]
    net = sepGAT(g, args.edge_feats, len(dl.links['count'])*2+1, len(features_list), in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False, 0., decode=args.decoder)
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # training loop
    net.train()
    loss_func = nn.BCELoss()
        
    num_all_para = 0
    all_para = []
    for name, _ in net.named_parameters():
        t = name.split('.')
        if 'attn_l' in t or 'attn_r' in t or 'attn_e' in t or 'edge_emb' in t or 'decoder' in t or 'fc_list' in t:
            num_all_para += 1
            all_para.append(name)
    num_current_para = num_all_para
    print('Total #param: {}'.format(num_all_para))
    
    remove_list = []
    removed_clients = []
    client_list = list(range(1, args.do_num))
    performance = {}
    param_to_return = {}
    for cid in range(1, args.do_num):
        param_to_return[cid] = all_para.copy()
    for epoch in range(args.communication_round):
        performance[epoch] = {}
        t_start_e = time.time()
        
        # get the active list
        if args.removeClient:
            for cid in remove_list:
                client_list.remove(cid)
        
        # strategy 2
        if args.explore and len(removed_clients) > 0:
            if int(args.do_num * args.active_rate)-len(client_list) > 0: # we need more
                if len(removed_clients) > int(args.do_num * args.active_rate)-len(client_list): # we need sampling
                    rejoin_list = random.sample(removed_clients, int(args.do_num * args.active_rate)-len(client_list))
                    client_list.extend(rejoin_list)
                    print('Rejoining clients: {}'.format(rejoin_list))
                    # reset signal
                    for cid in rejoin_list:
                        removed_clients.remove(cid)
                        param_to_return[cid] = all_para.copy()
                else: # rejoin all
                    client_list.extend(removed_clients)
                    print('Rejoining clients: {}'.format(removed_clients))
                    # reset signal
                    for cid in removed_clients:
                        param_to_return[cid] = all_para.copy()
                    removed_clients = []
        
        if num_current_para/num_all_para < args.Tr or len(client_list)/(args.do_num-1) <= args.Tr:
            print('Reinitializing...')
            remove_list = []
            removed_clients = []
            client_list = list(range(1, args.do_num))
            
            param_to_return = {}
            for cid in range(1, args.do_num):
                param_to_return[cid] = all_para.copy()
        
        print('Current Client List: {}'.format(client_list))
        
        
        if args.FedAvg:
            active_list = random.sample(client_list, int(len(client_list) * args.active_rate))
        else:
            active_list = client_list.copy()
        print('Current Active List: {}'.format(active_list))
        if not args.FedAvg:
            for cid in active_list:
                print('We want {} params from client {}.'.format(len(param_to_return[cid]), cid))
        removed_clients.extend(remove_list)
        
        print('Potential List: {}'.format(removed_clients))

        # FedAvg preparation
        server_state = {}
        avg_state = {}
        for name, para in net.named_parameters():
            server_state[name] = para
            avg_state[name] = []
        
        # training on clients
        for cid in active_list:
            current_state = client(args, server_state, loss_func, cid)
            for name, _ in net.named_parameters():
                avg_state[name].append(current_state[name]-server_state[name])
        
        # aggregate
        num_current_para = num_all_para
        for name, para in net.named_parameters():
            t = name.split('.')
            avg_para = torch.zeros_like(para)
            if 'attn_l' in t or 'attn_r' in t or 'attn_e' in t or 'edge_emb' in t or 'decoder' in t or 'fc_list' in t:
                if args.partiallyReturn:
                    returned_values = []
                    for i, cid in enumerate(active_list):
                        if name in param_to_return[cid]:
                            returned_values.append(avg_state[name][i])
                else:
                    returned_values = avg_state[name].copy()
                if len(returned_values) == 0:
                    num_current_para -= 1
                    avg_state[name] = avg_para + server_state[name]
                    continue

                current_avg = torch.mean(torch.abs(torch.stack(returned_values)))
                
                count = 0
                for p, cid in zip(avg_state[name], active_list):
                    # aggregate returned param
                    if torch.mean(torch.abs(p)) > current_avg or args.aggregateAll:
                        if not args.partiallyReturn or name in param_to_return[cid]:
                            avg_para = avg_para + p
                            count += 1
                    # set the signal for next round
                    if torch.mean(torch.abs(p)) <= current_avg:
                        if name in param_to_return[cid]:
                            param_to_return[cid].remove(name)
                
                avg_state[name] = avg_para/count + server_state[name]
                    
            else:
                for p in avg_state[name]:
                    avg_para = avg_para + p
                avg_state[name] = avg_para/len(avg_state[name]) + server_state[name]
        
        print('We updated {} params'.format(num_current_para))
        
        if args.removeClient:
            remove_list = []
            for cid in active_list:
                print('For Client {}, we want {} params for next round'.format(cid, len(param_to_return[cid])))
                if len(param_to_return[cid]) <= num_all_para * args.Tc:
                    remove_list.append(cid)
                    print('We remove it.')
        
        print('Removing Clients {}'.format(remove_list))

        
        net.load_state_dict(avg_state)
        
        # test
        all_test_neigh, all_test_label = dl.get_test_neigh()
        print('test for communication round: {}'.format(epoch))
        for test_edge_type in dl.links_test['data'].keys():
            print(test_edge_type)
            test_logits = []
            with torch.no_grad():
                all_res = {}
                test_neigh = all_test_neigh[test_edge_type]
                test_label = all_test_label[test_edge_type]
                left = np.array(test_neigh[0])
                right = np.array(test_neigh[1])
                mid = np.zeros(left.shape[0], dtype=np.int32)
                mid[:] = test_edge_type
                labels = torch.FloatTensor(test_label).to(device)
                labels = labels.cpu().numpy()
                batch_size = args.batch_size
                for step, start in enumerate(range(0, left.shape[0], args.batch_size)):
                    current_left = left[start:start+batch_size]
                    current_right = right[start:start+batch_size]
                    current_mid = mid[start:start+batch_size]
                    logits = net(features_list, e_feat, e_count, current_left, current_right, current_mid)
                    pred = F.sigmoid(logits).cpu().numpy()
                    edge_list = np.concatenate([current_left.reshape((1,-1)), current_right.reshape((1,-1))], axis=0)
                    current_labels = labels[start:start+batch_size]
                    res = dl.evaluate(edge_list, pred, current_labels)
                    for k in res:
                        if k not in all_res:
                            all_res[k] = 0
                        all_res[k] += res[k]
                res = {}
                for k in all_res:
                    res[k] = all_res[k]/(step+1)
                    print(k, all_res[k]/(step+1))
            performance[epoch][test_edge_type] = res
        random.seed(None)
        print('*'*89)
        
    print('results: ')
    for epoch in performance:
        print('epoch: {}'.format(epoch))
        print(performance[epoch])
    torch.save(net.state_dict(), 'checkpoint/server_{}_{}_{}_{}.pt'.format(args.prefix, args.dataset, args.num_layers, args.num_heads))        
    print('='*89)


def client(args, server_state, loss_func, do_id=1):
    s = time.time()
    print('training on client ' + str(do_id))
    # load client data
    feats_type = args.feats_type
    features_list, adjM, dl = load_data(args.prefix, args.dataset, do_id, ifFed=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, save_path='checkpoint/client_{}_{}_{}_{}.pt'.format(args.prefix, args.dataset, args.num_layers, do_id))

    # prepare data
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []#[features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)


    edge2type = {}
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            edge2type[(u,v)] = k
    for i in range(dl.nodes['total']):
        if (i,i) not in edge2type:
            edge2type[(i,i)] = len(dl.links['count'])
    for k in dl.links['data']:
        for u,v in zip(*dl.links['data'][k].nonzero()):
            if (v,u) not in edge2type:
                edge2type[(v,u)] = k+1+len(dl.links['count'])

    g = dgl.DGLGraph(adjM+(adjM.T))
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    g = g.to(device)
    e_feat = []
    e_count = [0 for _ in range(len(dl.links['count'])*2+1)]
    count = 0
    current_type = 0
    index = 0
    for u, v in zip(*g.edges()):
        u = u.cpu().item()
        v = v.cpu().item()
        type = edge2type[(u, v)]
        e_feat.append([type, index])
        index += 1
        e_count[type] += 1

    res_2hop = defaultdict(float)
    res_random = defaultdict(float)
    total = len(list(dl.links_test['data'].keys()))

    # load server model
    train_pos, valid_pos = dl.get_train_valid_pos()
    
    num_classes = args.hidden_dim
    heads = [args.num_heads] * args.num_layers + [args.num_heads]
    net = sepGAT(g, args.edge_feats, len(dl.links['count'])*2+1, len(features_list), in_dims, args.hidden_dim, num_classes, args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, False, 0., decode=args.decoder)
    net.to(device)
    net.load_state_dict(server_state)
    
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    net.train()
    
    
    # training on client
    train_pos_head_full = np.array([])
    train_pos_tail_full = np.array([])
    train_neg_head_full = np.array([])
    train_neg_tail_full = np.array([])
    r_id_full = np.array([])
    for test_edge_type in dl.links_test['data'].keys():
        train_neg = dl.get_train_neg(edge_types=[test_edge_type])[test_edge_type]
        train_pos_head_full = np.concatenate([train_pos_head_full, np.array(train_pos[test_edge_type][0])])
        train_pos_tail_full = np.concatenate([train_pos_tail_full, np.array(train_pos[test_edge_type][1])])
        train_neg_head_full = np.concatenate([train_neg_head_full, np.array(train_neg[0])])
        train_neg_tail_full = np.concatenate([train_neg_tail_full, np.array(train_neg[1])])
        r_id_full = np.concatenate([r_id_full, np.array([test_edge_type]*len(train_pos[test_edge_type][0]))])

    
    train_idx = np.arange(len(train_pos_head_full))
    np.random.shuffle(train_idx)
    batch_size = args.batch_size
    
    for epoch in range(args.c_epoch):
        t_start_e = time.time()
        for step, start in enumerate(range(0, len(train_pos_head_full), args.batch_size)):
            t_start = time.time()
            train_pos_head = train_pos_head_full[train_idx[start:start+batch_size]]
            train_neg_head = train_neg_head_full[train_idx[start:start+batch_size]]
            train_pos_tail = train_pos_tail_full[train_idx[start:start+batch_size]]
            train_neg_tail = train_neg_tail_full[train_idx[start:start+batch_size]]
            r_id = r_id_full[train_idx[start:start+batch_size]]
            left = np.concatenate([train_pos_head, train_neg_head])
            right = np.concatenate([train_pos_tail, train_neg_tail])
            mid = np.concatenate([r_id, r_id])
            labels = torch.FloatTensor(np.concatenate([np.ones(train_pos_head.shape[0]), np.zeros(train_neg_head.shape[0])])).to(device)
            
            logits = net(features_list, e_feat, e_count, left, right, mid)
            logp = F.sigmoid(logits)
            train_loss = loss_func(logp, labels)

            # autograd
            optimizer.zero_grad()
            train_loss.backward()
            
            optimizer.step()
            
            pub_grad = {}
            client_grad = {}                   
            
            if args.valEachStep:
                # validation
                net.eval()
                with torch.no_grad():
                    valid_pos_head = np.array([])
                    valid_pos_tail = np.array([])
                    valid_neg_head = np.array([])
                    valid_neg_tail = np.array([])
                    valid_r_id = np.array([])
                    for test_edge_type in dl.links['data'].keys():
                        valid_neg = dl.get_valid_neg(edge_types=[test_edge_type])[test_edge_type]
                        idx = random.sample(range(len(valid_pos[test_edge_type][0])), int(len(valid_pos[test_edge_type][0])/10))
                        valid_pos_head = np.concatenate([valid_pos_head, np.array(valid_pos[test_edge_type][0])[idx]])
                        valid_pos_tail = np.concatenate([valid_pos_tail, np.array(valid_pos[test_edge_type][1])[idx]])
                        valid_neg_head = np.concatenate([valid_neg_head, np.array(valid_neg[0])[idx]])
                        valid_neg_tail = np.concatenate([valid_neg_tail, np.array(valid_neg[1])[idx]])
                        valid_r_id = np.concatenate([valid_r_id, np.array([test_edge_type]*len(idx))])

                    left = np.concatenate([valid_pos_head, valid_neg_head])
                    right = np.concatenate([valid_pos_tail, valid_neg_tail])
                    mid = np.concatenate([valid_r_id, valid_r_id])
                    labels = torch.FloatTensor(np.concatenate([np.ones(valid_pos_head.shape[0]), np.zeros(valid_neg_head.shape[0])])).to(device)
                    logits = net(features_list, e_feat, e_count, left, right, mid)
                    logp = F.sigmoid(logits)
                    val_loss = loss_func(logp, labels)
                t_end = time.time()
                
                # early stopping
                early_stopping(val_loss, net)
                if early_stopping.early_stop:
                    # print('Early stopping on client!')
                    break
        if early_stopping.early_stop:
            # print('Early stopping on client!')
            break
    net.load_state_dict(torch.load('checkpoint/client_{}_{}_{}_{}.pt'.format(args.prefix, args.dataset, args.num_layers, do_id)))    
    client_state = {}
    for name, para in net.named_parameters():
        client_state[name] = para
    
    e = time.time()
    print(e-s)
    return client_state
    
        
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=0,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                        '4 - only term features (id vec for others);' +
                        '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--communication-round', type=int, default=30, help='Number of communication rounds.')
    ap.add_argument('--c_epoch', type=int, default=1, help='Number of epochs on each local data.')
    ap.add_argument('--patience', type=int, default=40, help='Patience.')
    ap.add_argument('--patience_cr', type=int, default=10, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.01)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=1024)
    ap.add_argument('--decoder', type=str, default='mydistmult')
    ap.add_argument('--do-num', type=int, default=4)
    ap.add_argument('--Tc', type=float, default=0.5)
    ap.add_argument('--Tr', type=float, default=0.3)
    ap.add_argument('--active-rate', type=float, default=0.7)
    ap.add_argument('--agg-rate', type=float, default=1)
    ap.add_argument('--valEachStep', action="store_true")
    ap.add_argument('--shortVersion', action="store_true")
    ap.add_argument('--aggregateAll', action="store_true")
    ap.add_argument('--removeClient', action="store_true")
    ap.add_argument('--partiallyReturn', action="store_true")
    ap.add_argument('--FedAvg', action="store_true")
    ap.add_argument('--explore', action="store_true")
    ap.add_argument('--server', type=str, default='0')
    ap.add_argument('--prefix', type=str)
    ap.add_argument('--path', type=str)
    args = ap.parse_args()
    

    start = time.time()
    run_model(args, args.server)
    end = time.time()
    print(end-start)
