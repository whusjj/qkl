# -*- coding: utf-8 -*-
import os
import copy
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
import sys
import math
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric import datasets
import torch_geometric.utils as utils
from sat.models import GraphTransformer
from sat.data import GraphDataset
from sat.utils import count_parameters
from sat.position_encoding import POSENCODINGS
from sat.gnn_layers import GNN_TYPES
from timeit import default_timer as timer
import ssl

def load_args():
    parser = argparse.ArgumentParser(
        description='Structure-Aware Transformer on ZINC',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    parser.add_argument('--dataset', type=str, default="ZINC",
                        help='name of dataset')
    parser.add_argument('--num-heads', type=int, default=8, help="number of heads")
    parser.add_argument('--num-layers', type=int, default=4, help="number of layers")
    parser.add_argument('--dim-hidden', type=int, default=32, help="hidden dimension of Transformer")
    parser.add_argument('--dropout', type=float, default=0.2, help="dropout")
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='batch size')
    parser.add_argument('--abs-pe', type=str, default=None, choices=POSENCODINGS.keys(),
                        help='which absolute PE to use?')
    parser.add_argument('--abs-pe-dim', type=int, default=20, help='dimension for absolute PE')
    parser.add_argument('--outdir', type=str, default='',
                        help='output path')
    parser.add_argument('--warmup', type=int, default=5000, help="number of iterations for warmup")
    parser.add_argument('--layer-norm', action='store_true', help='use layer norm instead of batch norm')
    parser.add_argument('--use-edge-attr', action='store_true', help='use edge features')
    parser.add_argument('--edge-dim', type=int, default=32, help='edge features hidden dim')
    parser.add_argument('--gnn-type', type=str, default='graphsage',
                        choices=GNN_TYPES,
                        help="GNN structure extractor type")
    parser.add_argument('--k-hop', type=int, default=2, 
        help="Number of hops to use when extracting subgraphs around each node")
    parser.add_argument('--global-pool', type=str, default='mean', choices=['mean', 'cls', 'add'],
                        help='global pooling method')
    parser.add_argument('--se', type=str, default="gnn", 
            help='Extractor type: khopgnn, or gnn')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    args.batch_norm = not args.layer_norm

    args.save_logs = False
    if args.outdir != '':
        args.save_logs = True
        outdir = args.outdir
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/{}'.format(args.dataset)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        outdir = outdir + '/seed{}'.format(args.seed)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        if args.use_edge_attr:
            outdir = outdir + '/edge_attr'
            if not os.path.exists(outdir):
                try:
                    os.makedirs(outdir)
                except Exception:
                    pass
        pedir = 'None' if args.abs_pe is None else '{}_{}'.format(args.abs_pe, args.abs_pe_dim)
        outdir = outdir + '/{}'.format(pedir)
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        bn = 'BN' if args.batch_norm else 'LN'
        if args.se == "khopgnn":
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.se, args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )

        else:
            outdir = outdir + '/{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.gnn_type, args.k_hop, args.dropout, args.lr, args.weight_decay,
                args.num_layers, args.num_heads, args.dim_hidden, bn,
            )
        if not os.path.exists(outdir):
            try:
                os.makedirs(outdir)
            except Exception:
                pass
        args.outdir = outdir
    return args


def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, graph_list, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        if epoch == 0:
            node_dict = {"node feature" : data.x,
                    "Edge indices" : data.edge_index,
                    "Edge attributes" : data.edge_attr,
                    "Node labels" : data.y}
            graph_list.append(node_dict)
        #print("Node features (if available):", data.x.shape)
        #print("Edge indices (source, target):", data.edge_index.shape)
        #print("Edge attributes (if available):", data.edge_attr.shape)
        #print("Node labels (if available):", data.y.shape)
        size = len(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)
        if args.abs_pe == 'lap':
            # sign flip as in Bresson et al. for laplacian PE
            sign_flip = torch.rand(data.abs_pe.shape[-1])
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.abs_pe = data.abs_pe * sign_flip.unsqueeze(0)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
        data = data.to(device)
        model=model.to(device)
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, epoch, graph_list, use_cuda=False, split='Val',conserve = False):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data in loader:
            if epoch == 1:
                node_dict = {"node feature" : data.x,
                        "Edge indices" : data.edge_index,
                        "Edge attributes" : data.edge_attr,
                        "Node labels" : data.y}
                graph_list.append(node_dict)
                print(split)
            size = len(data.y)
            if use_cuda:
                data = data.cuda()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       
            data = data.to(device)
            model=model.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            mse_loss += F.mse_loss(output, data.y).item() * size
            mae_loss += F.l1_loss(output, data.y).item() * size

            running_loss += loss.item() * size
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('{} loss: {:.4f} MSE loss: {:.4f} MAE loss: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_mse, epoch_mae, toc - tic))
    return epoch_mae, epoch_mse


def main():
    global args
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../datasets/ZINC'
    # number of node attributes for ZINC dataset
    n_tags = 28
    num_edge_features = 4

    train_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
        split='train'), degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)
    train_length = len(train_dset)
    input_size = n_tags
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
            shuffle=True)
    val_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
        split='val'), degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)
    val_length = len(val_dset)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)
    test_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
        split='test'), degree=True, k_hop=args.k_hop, se=args.se,
        use_subgraph_edge_attr=args.use_edge_attr)

    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)
    test_length = len(test_dset)
    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
        data in train_dset])

    model = GraphTransformer(in_size=input_size,
                             num_class=1,
                             d_model=args.dim_hidden,
                             dim_feedforward=2*args.dim_hidden,
                             dropout=args.dropout,
                             num_heads=args.num_heads,
                             num_layers=args.num_layers,
                             batch_norm=args.batch_norm,
                             abs_pe=args.abs_pe,
                             abs_pe_dim=args.abs_pe_dim,
                             gnn_type=args.gnn_type,
                             use_edge_attr=args.use_edge_attr,
                             num_edge_features=num_edge_features,
                             edge_dim=args.edge_dim,
                             k_hop=args.k_hop,
                             se=args.se,
                             deg=deg,
                             epoch=args.epochs,
                             layers=args.num_layers,
                             global_pool=args.global_pool,
                             test_length = test_length,
                             train_length=train_length,
                             val_length=val_length,
                             batch_size = args.batch_size) 

    if args.use_cuda:
        model.cuda()
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    
    #FIXME
    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_model = None
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    graph_list = []
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, graph_list, args.use_cuda, )
        val_loss,_ = eval_epoch(model, val_loader, criterion, args.use_cuda, graph_list, epoch, split='Val')
        test_loss,_ = eval_epoch(model, test_loader, criterion, args.use_cuda, graph_list, epoch, split='Test')

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs['train_mae'].append(train_loss)
        logs['val_mae'].append(val_loss)
        logs['test_mae'].append(test_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
    a = math.ceil(train_length / args.batch_size)
    b = math.ceil(train_length / args.batch_size) + math.ceil(val_length / args.batch_size)
    c = math.ceil(train_length / args.batch_size) +  math.ceil(val_length / args.batch_size)  +  math.ceil(test_length / args.batch_size)
    for i in range(len(graph_list)):
        j = i
        if i < a: 
            loaded_tensor = torch.load(f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/train/x_struct_{j}.pt")
        elif i <  b: 
            loaded_tensor = torch.load(f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/val/x_struct_{j}.pt")
        elif i <  c: 
            loaded_tensor = torch.load(f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/test/x_struct_{j}.pt")
        graph_list[i]["node feature"] = loaded_tensor
    torch.save(graph_list[:a - 1],f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/train_list.pt")
    torch.save(graph_list[a:b - 1],f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/val_list.pt")
    torch.save(graph_list[b:c - 1],f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/test_list.pt")
    for i in graph_list :
        i['Edge indices'] = i['Edge indices'].cpu().detach().numpy()
        i['Edge attributes'] = i['Edge attributes'].cpu().detach().numpy()
        i['Node labels'] = i['Node labels'].cpu().detach().numpy()
    print(graph_list[0]['node feature'].shape)
    print(graph_list[0]['Edge indices'].shape)
    print(graph_list[0]['Edge attributes'].shape)
    print(graph_list[0]['Node labels'].shape)
    np.save(f"/root/autodl-tmp/SAT-main/datasets/result/graph/raw/graph_list.npy",graph_list)
    split_dict = {
        'train' : np.arange(0,a - 1),
        'valid' : np.arange(a - 1,b - 2),
        'test-dev' : np.arange(b - 2,c - 3),
        'test-challenge' : np.arange(b - 2,c - 3)
    }
    print(split_dict)
    torch.save(split_dict,f"/root/autodl-tmp/SAT-main/datasets/result/graph/split_dict.pt")
    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)
    print("Testing...")
    test_loss, test_mse_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, graph_list, args.epochs+1, split='Test',conserve = True)

    print("test MAE loss {:.4f}".format(test_loss))
    print(args)

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_mae': test_loss,
            'test_mse': test_mse_loss,
            'val_mae': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            '/root/autodl-tmp/SAT-main/datasets/result/gnn_model.pth')


if __name__ == "__main__":
    ssl._create_default_https_context = ssl._create_unverified_context
    main()
