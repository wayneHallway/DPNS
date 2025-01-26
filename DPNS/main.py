import os
import random

import torch
import numpy as np

from time import time
from tqdm import tqdm
from copy import deepcopy
import logging
from prettytable import PrettyTable

from utils.parser import parse_args
from utils.data_loader import load_data
from utils.evaluate import test
from utils.helper import early_stopping

n_users = 0
n_items = 0


def get_feed_dict(train_entity_pairs, train_pos_set, start, end, n_negs=1):

    def sampling(user_item, train_set, n):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            negitems = []
            for i in range(n):  # sample n times
                while True:
                    negitem = random.choice(range(n_items))
                    #这个用choice随机生成一个int类型的数字
                    #随机采样，采样n次，随机选定一些物品为负样本
                    if negitem not in train_set[user]:
                        break
                negitems.append(negitem)
            neg_items.append(negitems)
        return neg_items

    feed_dict = {}
    #这个是一个字典
    entity_pairs = train_entity_pairs[start:end]
    #这里start和end设定好了实体样本的范围
    feed_dict['users'] = entity_pairs[:, 0]
    feed_dict['pos_items'] = entity_pairs[:, 1]
    
    feed_dict['neg_items'] = torch.LongTensor(sampling(entity_pairs,
                                                       train_pos_set,
                                                       n_negs*K)).to(device)
    #这个sampling函数主要就是为了生成负样本，但是这个生成的负样本数据类型是什么我不知道
    #甚至于我连字典里面users对应的值都不知道是什么类型的
    return feed_dict


if __name__ == '__main__':
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #终于找到原因了，当随机数种子被设置后，每次生成的随机数都被固定了，所以每次生成的结果相同，虽然不知道为什么还要下面cudnn的代码        
#torch.backends.cudnn.deterministic = True 会让CUDA卷积神经网络（CuDNN）在不同运行时间生成相同的输出，这是通过禁用一些算法的非确定性行为实现的。
# 该设置可以确保结果的可重复性，这对于调试和比较模型的性能很有用。

# torch.backends.cudnn.benchmark = False 取消启用CuDNN的自动寻找最适合当前硬件的高效卷积算法的功能，从而获得更准确的性能指标。
# 当输入数据大小相同时，这将使每个操作使用相同的算法，从而使结果可比较。但是，由于某些算法可能对输入大小非常敏感，
# 因此在某些情况下关闭此选项可能会使模型的性能下降。
    """read args"""
    global args, device
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
#上面都是参数设置，随机数，cuda的设置
    """build dataset"""
    train_cf, user_dict, item_dict,n_params, norm_mat, pop_weight = load_data(args)
    #这个牵扯到util的函数了，到时候再看
    train_cf_size = len(train_cf)
    train_cf = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_cf], np.int32))

    n_users = n_params['n_users']
    n_items = n_params['n_items']
    n_negs = args.n_negs
    K = args.K
#上面是数据集的初始化设置
    """define model"""
    from modules.LightGCN import LightGCN1
    from modules.LightGCN3 import LightGCN3
    from modules.LightGCN2 import LightGCN
    from modules.NGCF import NGCF1
    from modules.NGCF2 import NGCF
    from modules.APR import LGAPR
    from modules.APR_NGCF import APRNGCF
    from modules.ANS import ANS
    import torch
    if args.gnn == 'ANS':
        model = ANS(n_params, args, norm_mat).to(device)
    if args.gnn == 'lightgcn_novel':
        model = LightGCN1(n_params, args, norm_mat).to(device)
    if args.gnn == 'MF':
        model = LightGCN3(n_params, args, norm_mat).to(device)
    if args.gnn == 'lightgcn':
        model = LightGCN(n_params, args, norm_mat).to(device)
    if args.gnn == 'NGCF_novel':
        model = NGCF1(n_params, args, norm_mat).to(device)
    if args.gnn == 'NGCF':
        model = NGCF(n_params, args, norm_mat).to(device)
    if args.gnn == 'LGAPR':
        model = LGAPR(n_params, args, norm_mat).to(device)
    if args.gnn == 'NGAPR':
        model = APRNGCF(n_params, args, norm_mat).to(device)
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    cur_best_pre_0 = 0
    cur_best_pre_1 = 0
    cur_best_pre_2 = 0
    
    stopping_step = 0
    should_stop = False

    print("start training ...")
    params = list(model.parameters())
    print("Model has {} parameters".format(len(params)))
    for name, param in model.named_parameters():
        print(name, param.shape)
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        #随机给index数组的元素进行乱序处理
        train_cf_ = train_cf_[index].to(device)

        """training"""
        model.train()
        loss, s = 0, 0
        hits = 0
        train_s_t = time()
        while s + args.batch_size <= len(train_cf):
            batch = get_feed_dict(train_cf_,
                                  user_dict['train_user_set'],
                                  s, s + args.batch_size,
                                  n_negs)

            batch_loss, _, _ = model(epoch, batch)

            optimizer.zero_grad()
            
            #在使用apr时好像不要多次的使用loss.backward
            #可是这里不放backward又不对,所以还是弄上，apr的backward就不要了
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            s += args.batch_size

        train_e_t = time()

        if epoch % 5 == 0:
            """testing"""
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]

            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, n_params, mode='test')
            test_e_t = time()
            train_res.add_row(
                [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), test_ret['recall'], test_ret['ndcg'],
                 test_ret['precision'], test_ret['hit_ratio']])

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, n_params, mode='valid')
                test_e_t = time()
                train_res.add_row(
                    [epoch, train_e_t - train_s_t, test_e_t - test_s_t, loss.item(), valid_ret['recall'], valid_ret['ndcg'],
                     valid_ret['precision'], valid_ret['hit_ratio']])
            print(train_res)
            
            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            #不知道为什么原来放多个should_stop会出现问题，我要去early_stopping里面看看
            cur_best_pre_0, stopping_step, should_stop,cur_best_pre_1,cur_best_pre_2 = early_stopping(valid_ret, cur_best_pre_0,cur_best_pre_1,cur_best_pre_2,
                                                                       stopping_step, expected_order='acc',
                                                                        flag_step=10)
            cur=cur_best_pre_0*100
            print(cur)
            if should_stop:
                break 
            """save weight"""
            # if valid_ret['recall'][0] == cur_best_pre_0 and args.save:
            #     torch.save(model.state_dict(), args.out_dir + 'model_' + '.ckpt')
        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, loss.item()))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (train_e_t - train_s_t, epoch, loss.item()))

    print('early stopping at %d, recall@20:%.4f' % (epoch, cur_best_pre_0))
    # import os 
    # os.makedirs('2.1/GNN/DENS/weight', exist_ok=True) 
    # if cur_best_pre_0 > 0.0461: 
    #     torch.save(model.state_dict(), '2.1/GNN/DENS/weight/beauty_best_model.pth')
    with open('/home/wl/DENS/MF/test.txt', 'a') as f:
        print('choose=%d, eps=%.4f,warm=%d,mode=%s,gnn=%s,datan= %s,method=%s, beta= %.4f, hop= %d , neg= %d, original early stopping at %d, recall%s:%.4f, ndcg@20:%.4f, hit_ratio@20:%.4f' % (args.choose,args.gamma,args.warmup,args.mode,args.gnn,args.dataset,args.ns,args.beta,args.context_hops,args.n_negs,epoch, args.Ks,cur_best_pre_0,cur_best_pre_1,cur_best_pre_2),file=f)