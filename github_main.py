# -*- coding: utf-8 -*-
# 从main继承，改变的data读取，因为保存的数据多了x_in_vsg和x_in_usg，使用了gnn，并将边对称化
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from argparse import ArgumentParser, Namespace
from transformers.optimization import get_linear_schedule_with_warmup,get_constant_schedule_with_warmup
from transformers import AdamW

import pickle
import time
import torch
import math
from torch.nn import init
import json
import torch.nn as nn
import gc
import horovod.torch as hvd


############
from model import *
############

from focalloss import FocalLoss
from sklearn import metrics
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
import time
import torch.onnx
import os
import psutil
import sys
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from tools import *
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings('ignore')
os.chdir("")







class SparkIterableDataset(IterableDataset):
    def __init__(self, file_list, shuffle=False, buffer_size=0, negative_sampling_ratio=1.0, min_neighbour_num=0):
        self.file_list = file_list
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.buffer = []
        self.negative_sampling_ratio = negative_sampling_ratio
        self.min_neighbour_num=min_neighbour_num

    def process_json_file(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)
        for file_path in self.file_list:
            inst_list = []
            for line in open(file_path, 'r', encoding='utf-8'):
                inst = json.loads(line)
                if self.negative_sampling_ratio < 1:
                    if inst['label'] == 0:
                        rand_n = random.randint(0, 100)
                        if rand_n > self.negative_sampling_ratio * 100:
                            continue
                inst_list.append(inst)
            if self.shuffle:
                np.random.shuffle(inst_list)
            for inst in inst_list:
                if self.min_neighbour_num>0:
                    if sum(inst['x_in_vsg'])>self.min_neighbour_num and sum(inst['x_in_usg'])>self.min_neighbour_num:

                        yield inst
                else:
                    yield inst

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.file_list)
        return self.process_json_file()


def worker_init_fn(_):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    work_id = worker_info.id
    file_list = dataset.file_list
    split_size = math.ceil(len(file_list) / worker_info.num_workers)
    dataset.file_list = dataset.file_list[work_id * split_size: (work_id + 1) * split_size]


node_num_sum = 0

def load_feat_dict_v2(filenames):
    def piece_emb_read_and_decode(filenames, batch_size=50000):
        def _parse_fn(record):
            features = {
                #"node": tf.FixedLenFeature([], tf.string),
                "id": tf.FixedLenFeature([], tf.int64),
                #"node_type": tf.FixedLenFeature([], tf.string),
                "feature": tf.FixedLenFeature(shape=[args.feat_size], dtype=tf.float32),  # 250

            }
            # parsed = tf.parse_single_example(record, features)
            parsed = tf.parse_example(record, features)
            # return parsed
            return {#"node": parsed['node'],
                    "id": parsed['id'],
                    #"node_type": parsed['node_type'],
                    "feature": parsed['feature'],
                    }

        # Extract lines from input files using the Dataset API, can pass one filename or filename list
        dataset = tf.data.TFRecordDataset(
            filenames)  # .map(_parse_fn, num_parallel_calls=10).prefetch(1000)  # multi-thread pre-process then prefetch
        dataset = dataset.batch(batch_size).map(_parse_fn, num_parallel_calls=10).prefetch(100)  # Batch size to use
        iterator = dataset.make_one_shot_iterator()
        data_set = iterator.get_next()
        return data_set

    feat_dict = {}
    tf.reset_default_graph()
    node_iterator = piece_emb_read_and_decode(filenames, batch_size=500000)
    with tf.Session() as sess_0:
        while True:
            try:
                # get_memory_info(logger)
                node_data = sess_0.run([node_iterator])[0]
                # logger.info(node_data)
                ids = node_data['id']

                features = node_data['feature']
                # print(i)
                # i+=1
                for id, feature in zip(ids, features):
                    feat_dict[id] = feature

            except tf.errors.OutOfRangeError:
                break

    print(len(feat_dict))
    feat_matrix = np.zeros([len(feat_dict), args.feat_size]).astype(np.float32)  # 250
    for id, feature in feat_dict.items():
        feat_matrix[id] = feature

    return feat_matrix




def tm_collate_fn(data_list, feat_matrix, args):
    global cnt
    global big_cnt
    global node_num_sum

    batch_size=len(data_list)
    max_len=args.max_node_num
    feat_dim = feat_matrix.shape[1]
    batch_x = torch.zeros(batch_size, max_len, feat_dim)
    batch_adj = torch.zeros(batch_size, max_len,max_len)
    batch_x_mask = torch.zeros(batch_size,max_len)
    batch_x_type = torch.ones(batch_size, max_len)*args.node_type_num#padding的节点类型
    batch_x_in_vsg = torch.zeros(batch_size, max_len)
    batch_x_in_usg = torch.zeros(batch_size, max_len)
    batch_isout = torch.zeros(batch_size,max_len)
    batch_y = torch.zeros(batch_size)

    for i,inst in enumerate(data_list):
        x, y, edge_index, isout, x_type, x_in_vsg, x_in_usg= inst['x'], inst['label'], inst['edge_index'], inst[
            'isout'], inst['x_type'], inst['x_in_vsg'], inst['x_in_usg']
        x_feat = feat_matrix[x]
        x_num = len(x)

        if len(edge_index)<2:
            edge_adj = torch.zeros(max_len, max_len)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).reshape(2, -1)
            edge_adj = torch.zeros(max_len, max_len)
            edge_adj[edge_index[0], edge_index[1]] = 1

        assert sum(isout)==2

        if edge_adj.max()>1:
            print(edge_adj)

        cnt+=1
        node_num_sum+=len(x)

        if len(x)>args.max_node_num:
            big_cnt+=1
            if is_master:
                print(len(x),big_cnt,cnt,node_num_sum/cnt)
        if len(x)>max_len:
            print(len(x))
            raise  ValueError('Node Number exceed!')
        batch_x[i,:x_num,:]=torch.tensor(x_feat)
        batch_adj[i,:max_len,:max_len]=edge_adj
        batch_x_mask[i,:x_num]=1
        batch_x_type[i,:x_num]=torch.tensor(x_type)
        batch_x_in_usg[i,:x_num]=torch.tensor(x_in_usg)
        batch_x_in_vsg[i,:x_num]=torch.tensor(x_in_vsg)
        batch_isout[i,:x_num]=torch.tensor(isout)
        batch_y[i]=y


    return batch_x, batch_adj, batch_x_mask, batch_x_type, batch_x_in_vsg, batch_x_in_usg, batch_isout, batch_y.bool()





def get_model(args):

    if args.model_name =='transformer':
        model = Transformer_Model(args, trace_func=logger.info,rank=hvd_rank)
    else:
        raise ValueError('Model name Error')
    return model

def barrier(hvd):
    hvd.allreduce(torch.tensor(0), name='barrier')

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config.json', help='config path')
    parser.add_argument('--data_piece_num', type=int, default=8, help='Data Piece Number')
    parser.add_argument('--val', type=bool, default=True, help='whether validation')
    parser.add_argument('--test', type=bool, default=True, help='whether testing')
    parser.add_argument('--edge_mode', type=str, default='addsparse', help='use dataset with added edges')
    parser.add_argument('--flag', type=str, default='default', help='any description')
    parser.add_argument('--flag2', type=str, default='default', help='any description')
    parser.add_argument('--max_node_num', type=int, default=602, help='maximum node number')
    parser.add_argument('--train_dh_range', type=str, default='0_12', help='any description')
    parser.add_argument('--val_dh_range', type=str, default='12_24', help='any description')
    parser.add_argument('--test_dh_range', type=str, default='12_24', help='any description')
    parser.add_argument('--train_ds', type=str, default='offline/20210219dh', help='any description')
    parser.add_argument('--val_ds', type=str, default='offline/20210219dh', help='any description')
    parser.add_argument('--test_ds', type=str, default='offline/20210219dh', help='any description')
    parser.add_argument('--min_neb_num', type=int, default=0, help='the minimize node number of each subgraph')
    parser.add_argument('--feat_size', type=int, default=250, help='feature size')
    parser.add_argument('--att_score', type=bool, default=False, help='whether output attention score')
    # model parameters
    parser.add_argument('--model_name', type=str, default='RGCN_Rec', help='Model name')
    parser.add_argument('--hidden_size', type=int, default=100, help='hidden size')
    parser.add_argument('--gnn_num', type=int, default=3, help='gnn layer number')
    parser.add_argument('--gnn_type', type=str, default='rgcn', help='type of gnn layers')
    parser.add_argument('--cross_type', type=str, default='hgt', help='type of cross layers')
    parser.add_argument('--out_size', type=int, default=1, help='output size, 1 for sigmoid')
    parser.add_argument('--node_type_num', type=int, default=4, help='the number of node type')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--loss_func', type=str, default='BCE', help='loss function')
    parser.add_argument('--pool_type',type=str,default='split',help='pooling function')
    parser.add_argument('--feature_mode',type=str, default='dense+sparse',help='whether use dense or sparse features')
    parser.add_argument('--pos_encoding',type=int,default=1,help='whether use graph position encoding')
    parser.add_argument('--use_ffn',type=int,default=1,help='whether use pos_ffn layer')
    parser.add_argument('--head_masks',type=str,default='adj+sparse+full+full',help='multi-head mask types')
    # training parameters
    parser.add_argument('--optimizer',type=str, default='adam',help='name of the optimizer')
    parser.add_argument('--scheduler', type=str, default='none',help='name of the scheduler')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch Size')
    parser.add_argument('--epoch_num', type=int, default=10, help='Epoch Number')
    parser.add_argument('--negative_sampling_ratio', type=float, default=1.0,
                        help='sampling ratio of negative instances')
    parser.add_argument('--rand_seed', type=int, default=1025, help='ranndo')
    # parser.add_argument('--val_ratio',type=float, default=0.05, help='the ratio of validation set')
    args = parser.parse_args()
    config_path = '/apdcephfs/private_erxuemin/erxuemin/rec_project/wechat_project/config.json'
    if os.path.exists(config_path) and False:
        with open(config_path) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)

    model_path = '../model/checkpoint_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.pt' % \
                 (args.flag,
                  args.train_ds.replace('/',''),
                  args.train_dh_range,
                  args.val_ds.replace('/', ''),
                  args.val_dh_range,
                  args.max_node_num,
                  args.gnn_type,
                  args.gnn_num,
                  args.hidden_size,
                  args.head_masks,
                  args.pool_type,
                  args.loss_func,
                  args.min_neb_num
                  )
    general_model_path = '../model/checkpoint_%s_%s_%s_%s_%s_%s_%s.pt' % \
                 ('general',
                  args.max_node_num,
                  args.gnn_type,
                  args.gnn_num,
                  args.hidden_size,
                  args.head_masks,
                  args.pool_type
                  )
    setattr(args, 'model_path', model_path)
    setattr(args, 'general_model_path', general_model_path)
    return args

def get_logger(args):
    log_file_path = "/apdcephfs/private_erxuemin/erxuemin/rec_project/wechat_project/log/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.log.running" % (
        args.flag,
        time.strftime("%m-%d-%H-%M", time.localtime()),
        args.head_masks,
        args.model_name,
        args.feature_mode,
        args.max_node_num,
        args.gnn_type,
        args.pool_type,
        args.gnn_num,
        args.pos_encoding,
        args.use_ffn,
        args.hidden_size,
        args.negative_sampling_ratio,
        args.dropout_rate,
        args.loss_func)
    setattr(args, 'log_tmp_name',log_file_path)
    logging.basicConfig(filename=log_file_path, format='%(asctime)s,[:%(lineno)d] %(message)s', datefmt="%H:%M:%S")
    logging.getLogger().setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s,[:%(lineno)d] %(message)s',"%H:%M:%S"))
    logger = logging.getLogger()
    logger.addHandler(stream_handler)

    return logger


def get_loss_func(args):
    if args.loss_func == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
        args.out_size = 1
    elif args.loss_func == 'Focal':
        criterion = FocalLoss()
        args.out_size = 1
    elif args.loss_func == 'softmax':
        criterion = nn.CrossEntropyLoss()
        args.out_size = 2
    else:
        raise ValueError('loss function name error!')
    return criterion

def get_optimizer(optim_name, model):
    if optim_name == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=1e-3 * 2)
    elif optim_name == 'adamw':
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5*hvd_size*(args.batch_size/128))
    else:
        raise ValueError('optimizer name error')
    return optimizer

def get_scheduler(sche_name, optimizer, num_warmup_steps):
    if sche_name=='constant':
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
    else:
        raise ValueError('scheduler name error')
    return scheduler


def evaluation_v2(feat_path, data_path,dh_range, model, feat_matrix=None):
    if feat_matrix is None:
        feat_matrix = get_feat_matrix(feat_path, hvd_rank, False)

    #if is_master:
    #    get_memory_info(logger)
    if is_master:
        logger.info(data_path)
    barrier(hvd)
    data_files = gen_input_fileds_with_hours(data_path,dh_range)
    if hvd_size > 8:
        node_num = int(hvd_size / 8)
        file_per_rank = len(data_files) / node_num
        node_rank = int(hvd_rank / 8)
        data_files = data_files[int(node_rank*file_per_rank):int((node_rank+1)*file_per_rank)]


        #logger.info('Rank %s,Eval_%s, [%s,%s]' %(hvd_rank,hvd_local_rank,int(node_rank*file_per_rank),int((node_rank+1)*file_per_rank)))
    eval_batch_size = 2*args.batch_size
    barrier(hvd)
    data_dataset = SparkIterableDataset(data_files,min_neighbour_num=args.min_neb_num)
    data_loader = DataLoader(data_dataset, batch_size=eval_batch_size, num_workers=8,
                             worker_init_fn=worker_init_fn,
                             collate_fn=lambda x: tm_collate_fn(x, feat_matrix,args))
    barrier(hvd)

    labels = []
    probs = []
    total_loss = 0
    with torch.no_grad():

        batch_index = 0
        model.eval()
        for batch in data_loader:  # tqdm(test_loader):
            batch_index += 1
            batch = [item.to(device) for item in batch]
            batch_y = batch[-1]

            outputs = model(*batch[:-1])

            if outputs.shape[1] == 2:
                loss = criterion(outputs, batch_y)
                probs.extend(softmax(outputs)[:, 1].reshape(-1).tolist())
            else:
                outputs = outputs.reshape(-1)
                loss = criterion(outputs, batch_y.float())
                probs.extend(sigmoid(outputs).tolist())
            total_loss += loss.item()
            labels.extend(batch_y.tolist())
            if is_master:
                if batch_index % 200 == 1:
                    logger.info(
                        '{} instance used, testing loss: {:.4f},{:.4f}'.format(batch_index * eval_batch_size,
                                                                               total_loss / (batch_index),
                                                                               loss.item()))
    predicts = [int(p > 0.5) for p in probs]
    del (feat_matrix)
    gc.collect()

    barrier(hvd)

    if len(data_files) == 256:  # full test时候才输出每个
        all_labels = np.array(labels)
        all_probs = np.array(probs)
        fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs, pos_label=1)
        barrier(hvd)
        logger.info('rank %s: AUC %s' % (hvd_rank, metrics.auc(fpr, tpr)))
        barrier(hvd)

    # logger.info('Gathering...')
    predicts_list = hvd.allgather_object(predicts)
    all_predicts = sum(predicts_list, [])
    labels_list = hvd.allgather_object(labels)
    all_labels = sum(labels_list, [])
    probs_list = hvd.allgather_object(probs)
    all_probs = sum(probs_list, [])
    all_predicts = np.array(all_predicts)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs, pos_label=1)
    barrier(hvd)
    if is_master:
        logger.info('All Accuracy: %s' % ((all_predicts == all_labels).sum() / all_predicts.shape[0]))
        logger.info('All AUC %s' % metrics.auc(fpr, tpr))
        logger.info('All Postive samples %s, Negative samples %s, all %s' % (
            all_labels.sum(), len(all_labels) - all_labels.sum(), len(all_labels)))



    barrier(hvd)
    return metrics.auc(fpr, tpr), all_predicts, all_labels, all_probs



def train():
#if __name__ == '__main__':
    set_random_seeds(args.rand_seed + hvd_rank)
    if is_master:
        logger.info('Process Number: %s' % hvd_size)


    # =============================Model Preparation Start=======================
    early_stopping = EarlyStopping(patience=3, verbose=True, path=args.model_path, trace_func=logger.info,
                                   rank=hvd_rank)
    barrier(hvd)
    model = get_model(args)


    model = model.to(device)
    barrier(hvd)


    optimizer = get_optimizer(args.optimizer, model)

    # ++++++++++++++++++Horovod Distributed Module+++++++++++++++++++
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    if args.scheduler!='none':
        scheduler = get_scheduler(args.scheduler,optimizer,num_warmup_steps=500)

    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # ++++++++++++++++++Horovod Distributed Module+++++++++++++++++++
    # =============================Model Preparation End=======================
    root_path = "%s/%s" % (data_base_path, train_ds)
    piece_path_list = os.listdir(root_path)
    piece_num = len(piece_path_list)//8 * 8

    piece_id_list = [i for i in range(hvd_local_rank,piece_num,8)]
    #print(piece_id_list)
    local_piece_num = len(piece_id_list)

    barrier(hvd)



    train_cnt = 0
    for t in range(args.epoch_num):
        train_cnt += 1

        for k,piece_id in enumerate(piece_id_list):

            if is_master:
                logger.info('Epoch %s........%s/%s....................' % (t,k+1,local_piece_num))

            # =============================Training data loading start==============================
            # if is_master:
            barrier(hvd)

            if is_master:
                logger.info(root_path)
            #加载特征矩阵
            train_feat_matrix = get_feat_matrix(root_path + '/data_%s/node_feats.tfrecord/' % piece_id, hvd_rank, True)
            barrier(hvd)

            #加载数据
            train_path = root_path + '/data_%s/train.json/' % piece_id
            if is_master:
                logger.info(train_path)

            train_files = gen_input_fileds_with_hours(train_path,train_dh_range)  # 每个piece 256个文件，前234个做训练集，后16个做测试集
            if is_master:
                logger.info('total train files count: %s'%len(train_files) )
            if hvd_size>8:
                node_num =int(hvd_size/8)
                file_per_rank = len(train_files)/node_num
                node_rank = int(hvd_rank/8)
                train_files=train_files[int(node_rank*file_per_rank):int((node_rank+1)*file_per_rank)]

            #if is_master:
            #    get_memory_info(logger)

                #logger.info('rank %s, train_%s [%s,%s]' % (hvd_rank,hvd_local_rank,int(node_rank*file_per_rank),int((node_rank+1)*file_per_rank)))
            barrier(hvd)
            # 读取训练集
            train_dataset = SparkIterableDataset(train_files, shuffle=True,
                                                 negative_sampling_ratio=args.negative_sampling_ratio,min_neighbour_num=args.min_neb_num)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8,
                                      worker_init_fn=worker_init_fn,
                                      collate_fn=lambda x: tm_collate_fn(x, train_feat_matrix,args))
            barrier(hvd)

            model.train()
            total_loss = 0
            batch_index = 0
            labels = []

            flag=0.0
           # with torch.autograd.set_detect_anomaly(True):
            for i, batch in enumerate(train_loader):
                flag = hvd.allreduce(torch.tensor(flag), name='train_barrier').item()
                if flag>0:
                    #logger.info('rank %s break' %hvd_rank)
                    break
                optimizer.zero_grad()

                #batch2 = [item.clone() for item in batch]
                batch = [item.to(device) for item in batch]

                batch_y = batch[-1]

                try:
                    outputs = model(*batch[:-1])

                    if outputs.shape[1] == 2:#CE
                        loss = criterion(outputs, batch_y)
                    else:
                        outputs = outputs.reshape(-1)  # nx1 BCE
                        loss = criterion(outputs, batch_y.float())
                    #print(loss)
                    loss.backward()

                except:
                    #model.cpu()

                    #batch2 = [item.clone().cpu() for item in batch]
                    #print(batch2[1])
                    #f=open('errorbatch.pkl','wb')
                    #pickle.dump(batch2,f)
                    #f.close()
                    #outputs = model(*batch2[:-1])
                    #if outputs.shape[1] == 2:#CE
                    #    loss = criterion(outputs, batch_y.cpu())
                    #else:
                    #    outputs = outputs.reshape(-1)  # nx1 BCE
                    #    loss = criterion(outputs, batch_y.cpu().float())

                    raise ValueError('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')


                optimizer.step()
                if args.scheduler != 'none':
                    scheduler.step()

                total_loss += loss.item()
                batch_index += 1
                labels.extend(batch_y.tolist())
                if is_master:
                    if i % 400 == 0:
                        logger.info('{} instance used, training loss: {:.4f},{:.4f}'.format(batch_index * args.batch_size,
                                                                                            total_loss / (batch_index),
                                                                                            loss.item()))
                    #if i==10:#=========================
                    #    break#========================
            if flag==0:#first
                logger.info('rank %s first end' % hvd_rank)
                flag=hvd.allreduce(torch.tensor(1.0), name='train_barrier').item()

            #logger.info('iteration number: %s %s' %(i,flag))

            barrier(hvd)
            pos_num=sum(labels)
            neg_num = len(labels)-sum(labels)
            all_num = len(labels)
            pos_num_list=hvd.allgather_object(pos_num)
            neg_num_list=hvd.allgather_object(neg_num)
            all_num_list=hvd.allgather_object(all_num)

            if is_master:
                logger.info('Training Loss: %s' % (total_loss / (batch_index)))
                #for i in range(len(pos_num_list)):
                #    logger.info('rank %s, Postive samples %s, Negative samples %s, all %s' %(i, pos_num_list[i],neg_num_list[i],all_num_list[i]))
                logger.info('ALL, Postive samples %s, Negative samples %s, all %s' % (
                 sum(pos_num_list), sum(neg_num_list), sum(all_num_list)))

            del (train_feat_matrix)
            gc.collect()
            barrier(hvd)


        #================================================================================

        if val_ds:
            val_root_path = "%s/%s" % (data_base_path, val_ds)
            val_feat_path = val_root_path + '/node_feats.tfrecord/'  # % hvd_local_rank
            val_data_path = val_root_path + '/train.json/'  # % hvd_local_rank
            if hvd_rank == 0:
                logger.info('Validation.....................')
            metric = evaluation_v2(val_feat_path, val_data_path, val_dh_range, model)[0]

            early_stopping(metric, model)
            if early_stopping.early_stop:
                if hvd_rank == 0:
                    logger.info('Early stopping...........................')
                break




    barrier(hvd)


    if is_master and test_ds:

        for key, val in vars(args).items():
            logger.info(key + ':' + str(val))
        logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        score_list = early_stopping.score_list
        for cnt, score in score_list:
            logger.info('Epoch %s: AUC: %s' % (cnt, score))
        score_list = [p[1] for p in score_list]
        score_mean = np.mean(score_list)
        logger.info('Average val AUC: %s' % score_mean)
        logger.info('Best val AUC: %s, train_cnt: %s' % (early_stopping.best_score, early_stopping.best_cnt))
        logger.info("+++++++++++++++++++++++++++++++++++++++++++++++++++")
        os.system('mv %s %s' %(args.log_tmp_name, args.log_tmp_name.replace('.running','')))
        result_path = write_auc_results(early_stopping.best_score, score_list)


    if test_ds and False:
        test_score = test()

    if is_master and test_ds and False:
        f = open(result_path,'a')
        f.write('Test AUC score %s' %test_score)
        f.close()







def test():
    logger.info('Testing......')
    model = get_model(args)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    test_root_path = "%s/%s" % (data_base_path, test_ds)
    test_feat_path = test_root_path + '/data_%s/node_feats.tfrecord/' % hvd_local_rank
    test_data_path = test_root_path + '/data_%s/train.json/' % hvd_local_rank

    metric = evaluation_v2(test_feat_path, test_data_path, test_dh_range,model)[0]
    if is_master:
        logger.info('Test AUC: %s' %metric)
    return metric








def save_feat_matrix2hdf5(ds, replace=False):
    if hvd_size<16:
        return
    root_path = "%s/%s" % (data_base_path, ds)
    piece_path_list = os.listdir(root_path)
    data_piece_num = len(piece_path_list)//8 * 8
    times = hvd_size//data_piece_num

    if hvd_rank not in [times*i for i in range(data_piece_num)]:
        return


    root_path = "%s/%s" % (data_base_path, ds)

    path = root_path + '/data_%s/node_feats.tfrecord/' % (hvd_rank//times)
    hdf5_path = path.replace('node_feats.tfrecord/', 'node_feats.h5')
    if os.path.exists(hdf5_path):
        if replace:
            os.remove(hdf5_path)
        else:
            return
    logger.info('Saving... rank %s, piece id %s' % (hvd_rank, hvd_rank//times))
    node_feat_files = gen_input_fileds(path)
    feat_matrix = load_feat_dict_v2(node_feat_files)  # 返回的是一个字典
    if hvd_rank == 0:
        get_memory_info(logger)

    save_feat2hdf5(hdf5_path, feat_matrix)

    del (feat_matrix)
    gc.collect()



def write_auc_results(best_auc,auc_list):
    results_dir = ''
    result_path = '%s/%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s'%(results_dir,best_auc,args.flag,args.gnn_type,args.pool_type,args.hidden_size,
                                    train_ds.replace('/',''),train_dh_range[0],train_dh_range[1],
                                    val_ds.replace('/',''),val_dh_range[0],val_dh_range[1])
    f_res = open(result_path,'w')
    for i,auc in enumerate(auc_list):
        f_res.write('Epoch %s, AUC: %s\n'%(i,auc))
    f_res.close()
    return result_path





args = parse_args()
logger = get_logger(args)

hvd.init()
hvd_rank = hvd.rank()
hvd_size = hvd.size()
hvd_local_rank = hvd.local_rank()
device = torch.device('cuda', hvd_local_rank)
is_master = (hvd_rank == 0)
criterion = get_loss_func(args)
sigmoid = nn.Sigmoid()
softmax = nn.Softmax()
data_base_path = '/apdcephfs/private_erxuemin/erxuemin/dataset/graph'

train_ds = args.train_ds
val_ds=args.val_ds
test_ds=args.test_ds
train_dh_range =[int(i) for i in  args.train_dh_range.split('_')]
val_dh_range = [int(i) for i in  args.val_dh_range.split('_')]
test_dh_range =[int(i) for i in  args.test_dh_range.split('_')]



#输出参数
barrier(hvd)
if is_master:
    for key, val in vars(args).items():
        logger.info(key + ':' + str(val))
barrier(hvd)

if __name__ == '__main__':


    train()
















