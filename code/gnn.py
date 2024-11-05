import time
import gc
from dgl.data.utils import Subset
import pandas as pd
from dgl.data.chem import csv_dataset, smiles_to_bigraph, MoleculeCSVDataset
from gnn_utils import AttentiveFPBondFeaturizer, AttentiveFPAtomFeaturizer, collate_molgraphs, \
    EarlyStopping, set_random_seed, Meter
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.utils.data import DataLoader
import torch
from dgl.model_zoo.chem import MPNNModel, GCNClassifier, GATClassifier, AttentiveFP
import numpy as np
from sklearn.model_selection import train_test_split
from dgl import backend as F
from hyperopt import fmin, tpe, hp, Trials
import sys
from task_dict import tasks_dic

start_time = time.time()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
set_random_seed(seed=43)

def run_a_train_epoch(model, data_loader, loss_func, optimizer, args):
    model.train()
    train_metric = Meter()  # for each epoch
    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        atom_feats = bg.ndata.pop('h')
        bond_feats = bg.edata.pop('e')

        # transfer the data to device(cpu or cuda)
        labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
            args['device']), bond_feats.to(args['device'])

        outputs = model(bg, atom_feats) if args['model'] in ['gcn', 'gat'] else model(bg, atom_feats,
                                                                                               bond_feats)
        loss = (loss_func(outputs, labels) * (masks != 0).float()).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs.cpu()
        labels.cpu()
        masks.cpu()
        atom_feats.cpu()
        bond_feats.cpu()
        loss.cpu()
        torch.cuda.empty_cache()

        train_metric.update(outputs, labels, masks)

    if args['metric'] == 'rmse':
        rmse_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(train_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(train_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(train_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(train_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def run_an_eval_epoch(model, data_loader, args):
    model.eval()
    eval_metric = Meter()
    with torch.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            atom_feats = bg.ndata.pop('h')
            bond_feats = bg.edata.pop('e')

            # transfer the data to device(cpu or cuda)
            labels, masks, atom_feats, bond_feats = labels.to(args['device']), masks.to(args['device']), atom_feats.to(
                args['device']), bond_feats.to(args['device'])
            outputs = model(bg, atom_feats) if args['model'] in ['gcn', 'gat'] else model(bg, atom_feats,
                                                                                                   bond_feats)
            outputs.cpu()
            labels.cpu()
            masks.cpu()
            atom_feats.cpu()
            bond_feats.cpu()
            # loss.cpu()
            torch.cuda.empty_cache()
            eval_metric.update(outputs, labels, masks)
    if args['metric'] == 'rmse':
        rmse_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        mae_score = np.mean(eval_metric.compute_metric('mae'))  # in case of multi-tasks
        r2_score = np.mean(eval_metric.compute_metric('r2'))  # in case of multi-tasks
        return {'rmse': rmse_score, 'mae': mae_score, 'r2': r2_score}
    else:
        roc_score = np.mean(eval_metric.compute_metric(args['metric']))  # in case of multi-tasks
        prc_score = np.mean(eval_metric.compute_metric('prc_auc'))  # in case of multi-tasks
        return {'roc_auc': roc_score, 'prc_auc': prc_score}


def get_pos_weight(my_dataset):
    num_pos = F.sum(my_dataset.labels, dim=0)
    num_indices = F.tensor(len(my_dataset.labels))
    return (num_indices - num_pos) / num_pos


def all_one_zeros(series):
    if (len(series.dropna().unique()) == 2):
        flag = False
    else:
        flag = True
    return flag


file_name = sys.argv[1]  # ./dataset/bace.csv
model_name = sys.argv[2]  # 'gcn' or 'mpnn' or 'gat' or 'attentivefp'
task_type = sys.argv[3]  # 'cla' or 'reg'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
# if device == 'cuda':
#    torch.cuda.set_device(eval(gpu_id))  # gpu device id
my_df = pd.read_csv(file_name)
AtomFeaturizer = AttentiveFPAtomFeaturizer
BondFeaturizer = AttentiveFPBondFeaturizer
epochs = 300
batch_size = 128*5
patience = 50
opt_iters = 50
repetitions = 50
num_workers = 0
args = {'device': device, 'task': file_name.split('/')[-1].split('.')[0],
        'metric': 'roc_auc' if task_type == 'cla' else 'rmse', 'model': model_name}
tasks = tasks_dic[args['task']]
Hspace = {'gcn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                      lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                      gcn_hidden_feats=hp.choice('gcn_hidden_feats',
                                                 [[128, 128], [256, 256], [128, 64], [256, 128]]),
                      classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
          'mpnn': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                       lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                       node_hidden_dim=hp.choice('node_hidden_dim', [64, 32, 16]),
                       edge_hidden_dim=hp.choice('edge_hidden_dim', [64, 32, 16]),
                       num_layer_set2set=hp.choice('num_layer_set2set', [2, 3, 4])),
          'gat': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                      lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                      gat_hidden_feats=hp.choice('gat_hidden_feats',
                                                 [[128, 128], [256, 256], [128, 64], [256, 128]]),
                      num_heads=hp.choice('num_heads', [[2, 2], [3, 3], [4, 4], [4, 3], [3, 2]]),
                      classifier_hidden_feats=hp.choice('classifier_hidden_feats', [128, 64, 256])),
          'attentivefp': dict(l2=hp.choice('l2', [0, 10 ** -8, 10 ** -6, 10 ** -4]),
                              lr=hp.choice('lr', [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]),
                              num_layers=hp.choice('num_layers', [2, 3, 4, 5, 6]),
                              num_timesteps=hp.choice('num_timesteps', [1, 2, 3, 4, 5]),
                              dropout=hp.choice('dropout', [0, 0.1, 0.3, 0.5]),
                              graph_feat_size=hp.choice('graph_feat_size', [50, 100, 200, 300]))}
hyper_space = Hspace[args['model']]

# get the df and generate graph, attention for graph generation for some bad smiles
# my_df.iloc[:, 0:-1], except with 'group'
my_dataset: MoleculeCSVDataset = csv_dataset.MoleculeCSVDataset(my_df.iloc[:, 0:-1], smiles_to_bigraph, AtomFeaturizer,
                                                                BondFeaturizer, 'cano_smiles',
                                                                file_name.replace('.csv', '.bin'))
if task_type == 'cla':
    pos_weight = get_pos_weight(my_dataset)
else:
    pos_weight = None

# get the training, validation, and test sets
tr_indx, val_indx, te_indx = my_df[my_df.group == 'train'].index, my_df[my_df.group == 'valid'].index, my_df[
    my_df.group == 'test'].index
# train_set, val_set, test_set = Subset(my_dataset, tr_indx), Subset(my_dataset, val_indx), Subset(my_dataset, te_indx)
train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                          collate_fn=collate_molgraphs, num_workers=num_workers)
val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=True,
                        collate_fn=collate_molgraphs, num_workers=num_workers)
test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=True,
                         collate_fn=collate_molgraphs, num_workers=num_workers)


def hyper_opt(hyper_paras):
    # get the model instance
    if model_name == 'gcn':
        my_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                 gcn_hidden_feats=hyper_paras['gcn_hidden_feats'],
                                 n_tasks=len(tasks), classifier_hidden_feats=hyper_paras['classifier_hidden_feats'])
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s.pth' % (args['model'], args['task'],
                                                                     hyper_paras['l2'], hyper_paras['lr'],
                                                                     hyper_paras['gcn_hidden_feats'],
                                                                     hyper_paras['classifier_hidden_feats'])
    elif model_name == 'mpnn':
        my_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'), edge_input_dim=BondFeaturizer.feat_size('e'),
                             output_dim=len(tasks), node_hidden_dim=hyper_paras['node_hidden_dim'],
                             edge_hidden_dim=hyper_paras['edge_hidden_dim'],
                             num_layer_set2set=hyper_paras['num_layer_set2set'])
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_%s.pth' % (args['model'], args['task'],
                                                                        hyper_paras['l2'], hyper_paras['lr'],
                                                                        hyper_paras['node_hidden_dim'],
                                                                        hyper_paras['edge_hidden_dim'],
                                                                        hyper_paras['num_layer_set2set'])
    elif model_name == 'attentivefp':
        my_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                               edge_feat_size=BondFeaturizer.feat_size('e'),
                               num_layers=hyper_paras['num_layers'], num_timesteps=hyper_paras['num_timesteps'],
                               graph_feat_size=hyper_paras['graph_feat_size'], output_size=len(tasks),
                               dropout=hyper_paras['dropout'])
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (args['model'], args['task'],
                                                                           hyper_paras['l2'], hyper_paras['lr'],
                                                                           hyper_paras['num_layers'],
                                                                           hyper_paras['num_timesteps'],
                                                                           hyper_paras['graph_feat_size'],
                                                                           hyper_paras['dropout'])
    else:
        my_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                 gat_hidden_feats=hyper_paras['gat_hidden_feats'],
                                 num_heads=hyper_paras['num_heads'], n_tasks=len(tasks),
                                 classifier_hidden_feats=hyper_paras['classifier_hidden_feats'])
        model_file_name = './saved_model/%s_%s_%s_%.6f_%s_%s_%s.pth' % (args['model'], args['task'],
                                                                        hyper_paras['l2'], hyper_paras['lr'],
                                                                        hyper_paras['gat_hidden_feats'],
                                                                        hyper_paras['num_heads'],
                                                                        hyper_paras['classifier_hidden_feats'])
    optimizer = torch.optim.Adam(my_model.parameters(), lr=hyper_paras['lr'], weight_decay=hyper_paras['l2'])

    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=model_file_name)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=model_file_name)
    my_model.to(device)

    for j in range(epochs):
        # training
        run_a_train_epoch(my_model, train_loader, loss_func, optimizer, args)

        # early stopping
        val_scores = run_an_eval_epoch(my_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], my_model)

        if early_stop:
            break
    stopper.load_checkpoint(my_model)
    tr_scores = run_an_eval_epoch(my_model, train_loader, args)
    val_scores = run_an_eval_epoch(my_model, val_loader, args)
    te_scores = run_an_eval_epoch(my_model, test_loader, args)
    print({'train': tr_scores, 'valid': val_scores, 'test': te_scores})
    feedback = val_scores[args['metric']] if task_type == 'reg' else (1 - val_scores[args['metric']])
    my_model.cpu()
    torch.cuda.empty_cache()
    gc.collect()
    return feedback


# start hyper-parameters optimization
print('******hyper-parameter optimization is starting now******')
trials = Trials()
opt_res = fmin(hyper_opt, hyper_space, algo=tpe.suggest, max_evals=opt_iters, trials=trials)

# hyper-parameters optimization is over
print('******hyper-parameter optimization is over******')
print('the best hyper-parameters settings for ' + args['task'] + ' ' + args['model'] + ' are:  ', opt_res)

# construct the model based on the optimal hyper-parameters
l2_ls = [0, 10 ** -8, 10 ** -6, 10 ** -4]
lr_ls = [10 ** -2.5, 10 ** -3.5, 10 ** -1.5]
hidden_feats_ls = [(128, 128), (256, 256), (128, 64), (256, 128)]
classifier_hidden_feats_ls = [128, 64, 256]
node_hidden_dim_ls = [64, 32, 16]
edge_hidden_dim_ls = [64, 32, 16]
num_layer_set2set_ls = [2, 3, 4]
num_heads_ls = [(2, 2), (3, 3), (4, 4), (4, 3), (3, 2)]
num_layers_ls = [2, 3, 4, 5, 6]
num_timesteps_ls = [1, 2, 3, 4, 5]
graph_feat_size_ls = [50, 100, 200, 300]
dropout_ls = [0, 0.1, 0.3, 0.5]
# if model_name == 'gcn':
#     best_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
#                                gcn_hidden_feats=hidden_feats_ls[opt_res['gcn_hidden_feats']],
#                                n_tasks=len(tasks),
#                                classifier_hidden_feats=classifier_hidden_feats_ls[opt_res['classifier_hidden_feats']])
#     best_model_file = './saved_model/%s_%s_%s_%.6f_%s_%s.pth' % (args['model'], args['task'],
#                                                                  l2_ls[opt_res['l2']], lr_ls[opt_res['lr']],
#                                                                  hidden_feats_ls[opt_res['gcn_hidden_feats']],
#                                                                  classifier_hidden_feats_ls[
#                                                                      opt_res['classifier_hidden_feats']])
# elif model_name == 'mpnn':
#     best_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'), edge_input_dim=BondFeaturizer.feat_size('e'),
#                            output_dim=len(tasks), node_hidden_dim=node_hidden_dim_ls[opt_res['node_hidden_dim']],
#                            edge_hidden_dim=edge_hidden_dim_ls[opt_res['edge_hidden_dim']],
#                            num_layer_set2set=num_layer_set2set_ls[opt_res['num_layer_set2set']])
#     best_model_file = './saved_model/%s_%s_%s_%.6f_%s_%s_%s.pth' % (args['model'], args['task'],
#                                                                     l2_ls[opt_res['l2']], lr_ls[opt_res['lr']],
#                                                                     node_hidden_dim_ls[opt_res['node_hidden_dim']],
#                                                                     edge_hidden_dim_ls[opt_res['edge_hidden_dim']],
#                                                                     num_layer_set2set_ls[opt_res['num_layer_set2set']])
# elif model_name == 'attentivefp':
#     best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'), edge_feat_size=BondFeaturizer.feat_size('e'),
#                            num_layers=num_layers_ls[opt_res['num_layers']],
#                            num_timesteps=num_timesteps_ls[opt_res['num_timesteps']],
#                            graph_feat_size=graph_feat_size_ls[opt_res['graph_feat_size']], output_size=len(tasks),
#                            dropout=dropout_ls[opt_res['dropout']])
#     best_model_file = './saved_model/%s_%s_%s_%.6f_%s_%s_%s_%s.pth' % (args['model'], args['task'],
#                                                                        l2_ls[opt_res['l2']], lr_ls[opt_res['lr']],
#                                                                        num_layers_ls[opt_res['num_layers']],
#                                                                        num_timesteps_ls[opt_res['num_timesteps']],
#                                                                        graph_feat_size_ls[opt_res['graph_feat_size']],
#                                                                        dropout_ls[opt_res['dropout']])
#
# else:
#     best_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
#                                gat_hidden_feats=hidden_feats_ls[opt_res['gat_hidden_feats']],
#                                num_heads=num_heads_ls[opt_res['num_heads']], n_tasks=len(tasks),
#                                classifier_hidden_feats=classifier_hidden_feats_ls[opt_res['classifier_hidden_feats']])
#     best_model_file = './saved_model/%s_%s_%s_%.6f_%s_%s_%s.pth' % (args['model'], args['task'],
#                                                                     l2_ls[opt_res['l2']], lr_ls[opt_res['lr']],
#                                                                     hidden_feats_ls[opt_res['gat_hidden_feats']],
#                                                                     num_heads_ls[opt_res['num_heads']],
#                                                                     classifier_hidden_feats_ls[
#                                                                     opt_res['classifier_hidden_feats']])
# print(best_model_file)
# best_model.load_state_dict(torch.load(best_model_file, map_location=device)['model_state_dict'])
# best_model.to(device)
# tr_scores = run_an_eval_epoch(best_model, train_loader, args)
# val_scores = run_an_eval_epoch(best_model, val_loader, args)
# te_scores = run_an_eval_epoch(best_model, test_loader, args)
# print('training set:', tr_scores)
# print('validation set:', val_scores)
# print('test set:', te_scores)

# 50 repetitions based on the best model
tr_res = []
val_res = []
te_res = []
# regenerate the graphs
if args['task'] == 'muv' or args['task'] == 'toxcast':
    file_name = './dataset/' + args['task'] + '_new.csv'
    my_df = pd.read_csv(file_name)
    my_dataset = csv_dataset.MoleculeCSVDataset(my_df, smiles_to_bigraph, AtomFeaturizer, BondFeaturizer,
                                                'cano_smiles', file_name.replace('.csv', '.bin'))
else:
    my_df.drop(columns=['group'], inplace=True)

for split in range(1, repetitions + 1):
    # splitting the data set for classification
    if args['metric'] == 'roc_auc':
        seed = split
        while True:
            training_data, data_te = train_test_split(my_df, test_size=0.1, random_state=seed)
            # the training set was further splitted into the training set and validation set
            data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=seed)
            if np.any(data_tr[tasks].apply(all_one_zeros)) or \
                    np.any(data_va[tasks].apply(all_one_zeros)) or \
                    np.any(data_te[tasks].apply(all_one_zeros)):
                print('\ninvalid random seed {} due to one class presented in the splitted {} sets...'.format(seed,
                                                                                                              args[
                                                                                                                  'task']))
                print('Changing to another random seed...\n')
                seed = np.random.randint(50, 999999)
            else:
                print('random seed used in repetition {} is {}'.format(split, seed))
                break
    else:
        training_data, data_te = train_test_split(my_df, test_size=0.1, random_state=split)
        # the training set was further splitted into the training set and validation set
        data_tr, data_va = train_test_split(training_data, test_size=0.1, random_state=split)
    tr_indx, val_indx, te_indx = data_tr.index, data_va.index, data_te.index
    train_loader = DataLoader(Subset(my_dataset, tr_indx), batch_size=batch_size, shuffle=True,
                              collate_fn=collate_molgraphs, num_workers=num_workers)
    val_loader = DataLoader(Subset(my_dataset, val_indx), batch_size=batch_size, shuffle=True,
                            collate_fn=collate_molgraphs, num_workers=num_workers)
    test_loader = DataLoader(Subset(my_dataset, te_indx), batch_size=batch_size, shuffle=True,
                             collate_fn=collate_molgraphs, num_workers=num_workers)
    best_model_file = './saved_model/%s_%s_bst_%s.pth' % (args['model'], args['task'], split)

    if model_name == 'gcn':
        best_model = GCNClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gcn_hidden_feats=hidden_feats_ls[opt_res['gcn_hidden_feats']],
                                   n_tasks=len(tasks),
                                   classifier_hidden_feats=classifier_hidden_feats_ls[
                                       opt_res['classifier_hidden_feats']])

    elif model_name == 'gat':
        best_model = GATClassifier(in_feats=AtomFeaturizer.feat_size('h'),
                                   gat_hidden_feats=hidden_feats_ls[opt_res['gat_hidden_feats']],
                                   num_heads=num_heads_ls[opt_res['num_heads']], n_tasks=len(tasks),
                                   classifier_hidden_feats=classifier_hidden_feats_ls[
                                       opt_res['classifier_hidden_feats']])
    elif model_name == 'attentivefp':
        best_model = AttentiveFP(node_feat_size=AtomFeaturizer.feat_size('h'),
                                 edge_feat_size=BondFeaturizer.feat_size('e'),
                                 num_layers=num_layers_ls[opt_res['num_layers']],
                                 num_timesteps=num_timesteps_ls[opt_res['num_timesteps']],
                                 graph_feat_size=graph_feat_size_ls[opt_res['graph_feat_size']], output_size=len(tasks),
                                 dropout=dropout_ls[opt_res['dropout']])
    else:
        best_model = MPNNModel(node_input_dim=AtomFeaturizer.feat_size('h'),
                               edge_input_dim=BondFeaturizer.feat_size('e'),
                               output_dim=len(tasks), node_hidden_dim=node_hidden_dim_ls[opt_res['node_hidden_dim']],
                               edge_hidden_dim=edge_hidden_dim_ls[opt_res['edge_hidden_dim']],
                               num_layer_set2set=num_layer_set2set_ls[opt_res['num_layer_set2set']])

    best_optimizer = torch.optim.Adam(best_model.parameters(), lr=lr_ls[opt_res['lr']],
                                      weight_decay=l2_ls[opt_res['l2']])
    if task_type == 'reg':
        loss_func = MSELoss(reduction='none')
        stopper = EarlyStopping(mode='lower', patience=patience, filename=best_model_file)
    else:
        loss_func = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight.to(args['device']))
        stopper = EarlyStopping(mode='higher', patience=patience, filename=best_model_file)
    best_model.to(device)

    for j in range(epochs):
        # training
        st = time.time()
        run_a_train_epoch(best_model, train_loader, loss_func, best_optimizer, args)
        end = time.time()
        # early stopping
        train_scores = run_an_eval_epoch(best_model, train_loader, args)
        val_scores = run_an_eval_epoch(best_model, val_loader, args)
        early_stop = stopper.step(val_scores[args['metric']], best_model)
        if early_stop:
            break
        print(
            'task:{} repetition {:d}/{:d} epoch {:d}/{:d}, training {} {:.3f}, validation {} {:.3f}, time:{:.3f}S'.format(
                args['task'], split, repetitions, j + 1, epochs, args['metric'], train_scores[args['metric']],
                args['metric'],
                val_scores[args['metric']], end - st))
    stopper.load_checkpoint(best_model)
    tr_scores = run_an_eval_epoch(best_model, train_loader, args)
    val_scores = run_an_eval_epoch(best_model, val_loader, args)
    te_scores = run_an_eval_epoch(best_model, test_loader, args)
    tr_res.append(tr_scores);
    val_res.append(val_scores);
    te_res.append(te_scores)
if task_type == 'reg':
    cols = ['rmse', 'mae', 'r2']
else:
    cols = ['roc_auc', 'prc_auc']
tr = [list(item.values()) for item in tr_res]
val = [list(item.values()) for item in val_res]
te = [list(item.values()) for item in te_res]
tr_pd = pd.DataFrame(tr, columns=cols)
tr_pd['split'] = range(1, repetitions + 1)
tr_pd['set'] = 'train'
val_pd = pd.DataFrame(val, columns=cols)
val_pd['split'] = range(1, repetitions + 1)
val_pd['set'] = 'validation'
te_pd = pd.DataFrame(te, columns=cols)
te_pd['split'] = range(1, repetitions + 1)
te_pd['set'] = 'test'
sta_pd = pd.concat([tr_pd, val_pd, te_pd], ignore_index=True)
sta_pd['model'] = args['model']
sta_pd['dataset'] = args['task']
sta_pd.to_csv('./stat_res/{}_{}_statistical_results_split50.csv'.format(args['task'], args['model']), index=False)

print('training mean:', np.mean(tr, axis=0), 'training std:', np.std(tr, axis=0))
print('validation mean:', np.mean(val, axis=0), 'validation std:', np.std(val, axis=0))
print('testing mean:', np.mean(te, axis=0), 'test std:', np.std(te, axis=0))
end_time = time.time()
print('the total elapsed time is', end_time - start_time, 'S')
