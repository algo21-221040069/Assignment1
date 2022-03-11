import collections
import copy
import datetime
import json
import os
import numpy as np
import pandas as pd
import qlib
import torch
import torch.optim as optim
# regiodatetimeG_CN, REG_US]
from qlib.config import REG_CN
from tqdm import tqdm

# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from model import HIST
from utils import metric_fn, mse
from dataloader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params' % i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


def loss_fn(pred, label):
    mask = ~torch.isnan(label)
    return mse(pred[mask], label[mask])


def pprint(*args):
    # print with UTC+8 time
    time = '[' + str(datetime.datetime.utcnow() +
                     datetime.timedelta(hours=8))[:19] + '] -'
    print(time, *args, flush=True)

    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1


def train_epoch(model, optimizer, train_loader, stock2concept_matrix=None):
    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value, stock_index, _ = train_loader.get(slc)
        pred = model(feature, stock2concept_matrix[stock_index], market_value)
        loss = loss_fn(pred, label)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)
        optimizer.step()


def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, prefix='Test'):
    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index = test_loader.get(slc)

        with torch.no_grad():
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            loss = loss_fn(pred, label)
            preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())
    # evaluate
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse

    writer.add_scalar(prefix + '/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix + '/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix + '/' + args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix + '/std(' + args.metric + ')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic


def inference(model, data_loader, stock2concept_matrix=None):
    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):

        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
            preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

    preds = pd.concat(preds, axis=0)
    return preds


def create_loaders(args):
    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')

    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler',
               'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time,
                          'fit_end_time': train_end_time, 'instruments': args.data_set,
                          'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature',
                                                                                        'clip_outlier': True}},
                                               {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}],
                          'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm',
                                                                          'kwargs': {'fields_group': 'label'}}],
                          'label': ['Ref($close, -1) / $close - 1']}}
    segments = {'train': (args.train_start_date, args.train_end_date),
                'valid': (args.valid_start_date, args.valid_end_date),
                'test': (args.test_start_date, args.test_end_date)}
    dataset = DatasetH(hanlder, segments)

    df_train, df_valid, df_test = dataset.prepare(["train", "valid", "test"], col_set=["feature", "label"],
                                                  data_key=DataHandlerLP.DK_L, )
    import pickle
    with open(args.market_value_path, "rb") as fh:
        df_market_value = pickle.load(fh)
    # df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value /= 1000000000
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'],
                              batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index,
                              device=device)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'],
                              start_index=start_index, device=device)

    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'],
                             start_index=start_index, device=device)

    return train_loader, valid_loader, test_loader


def main(args):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s" % (
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path + '/' + 'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)
    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args)

    stock2concept_matrix = np.load(args.stock2concept_matrix)
    stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)

    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(args.repeat):
        pprint('create model...')
        model = HIST(d_feat=args.d_feat, num_layers=args.num_layers, K=args.K)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)
        for epoch in range(args.n_epochs):
            pprint('Running', times, 'Epoch:', epoch)

            pprint('training...')
            train_epoch(model, optimizer, train_loader, stock2concept_matrix)
            torch.save(model.state_dict(), output_path + '/model.bin.e' + str(epoch))
            torch.save(optimizer.state_dict(), output_path + '/optimizer.bin.e' + str(epoch))

            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            model.load_state_dict(avg_params)

            pprint('evaluating...')
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(
                epoch, model, train_loader, writer, args, stock2concept_matrix, prefix='Train')
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader,
                                                                                             writer, args,
                                                                                             stock2concept_matrix,
                                                                                             prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model,
                                                                                                   test_loader, writer,
                                                                                                   args,
                                                                                                   stock2concept_matrix)

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f' % (train_loss, val_loss, test_loss))
            pprint('train_score %.6f, valid_score %.6f, test_score %.6f' % (train_score, val_score, test_score))
            # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
            # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f' % (train_ic, val_ic, test_ic))
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f' % (
                train_rank_ic, val_rank_ic, test_rank_ic))
            pprint('Train Precision: ', train_precision)
            pprint('Valid Precision: ', val_precision)
            pprint('Test Precision: ', test_precision)
            pprint('Train Recall: ', train_recall)
            pprint('Valid Recall: ', val_recall)
            pprint('Test Recall: ', test_recall)
            model.load_state_dict(params_ckpt)

            if val_score > best_score:
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path + '/model.bin')

        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:
            pred = inference(model, eval(name + '_loader'), stock2concept_matrix)
            pred.to_pickle(output_path + '/pred.pkl.' + name + str(times))

            precision, recall, ic, rank_ic = metric_fn(pred)

            pprint('%s: IC %.6f Rank IC %.6f' % (
                name, ic.mean(), rank_ic.mean()))
            pprint(name, ': Precision ', precision)
            pprint(name, ': Recall ', recall)
            res[name + '-IC'] = ic
            # res[name+'-ICIR'] = ic.mean() / ic.std()
            res[name + '-RankIC'] = rank_ic
            # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()

        all_precision.append(list(precision.values()))
        all_recall.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/' + key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path + '/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)' % (
        np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis=0)
    precision_std = np.array(all_precision).std(axis=0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        pprint('Precision@%d: %.4f (%.4f)' % (N[k], precision_mean[k], precision_std[k]))

    pprint('finished.')


class parse_args:
    def __init__(self, model_name='HIST',
                 d_feat=6,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.0,
                 K=1,

                 # training
                 n_epochs=200,
                 lr=2e-4,
                 early_stop=30,
                 smooth_steps=5,
                 metric='IC',
                 loss='mse',
                 repeat=10,

                 # data
                 data_set='csi100',  # 'csi300'
                 pin_memory=True,
                 batch_size=-1,  # -1 indicate daily batch
                 least_samples_num=1137.0,
                 label='',  # specify other labels
                 train_start_date='2007-01-01',
                 train_end_date='2014-12-31',
                 valid_start_date='2015-01-01',
                 valid_end_date='2016-12-31',
                 test_start_date='2017-01-01',
                 test_end_date='2020-12-31',

                 # other
                 seed=0,
                 annot='',
                 config='',
                 name='csi100_HIST',  # 'csi300_HIST'

                 # input for csi 300
                 market_value_path='./data/csi300_market_value_07to20.pkl',
                 stock2concept_matrix='./data/csi300_stock2concept.npy',
                 stock_index='./data/csi300_stock_index.npy',

                 outdir='./output/csi300_HIST',
                 overwrite=False):
        self.model_name = model_name
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.K = K

        # training
        self.n_epochs = n_epochs
        self.lr = lr
        self.early_stop = early_stop
        self.smooth_steps = smooth_steps
        self.metric = metric
        self.loss = loss
        self.repeat = repeat

        # data
        self.data_set = data_set
        self.pin_memory = pin_memory
        self.batch_size = batch_size  # -1 indicate daily batch
        self.least_samples_num = least_samples_num
        self.label = label,  # specify other labels
        self.train_start_date = train_start_date
        self.train_end_date = train_end_date
        self.valid_start_date = valid_start_date
        self.valid_end_date = valid_end_date
        self.test_start_date = test_start_date
        self.test_end_date = test_end_date

        # other
        self.seed = seed
        self.annot = annot
        self.config = config
        self.name = name

        # input for csi 300
        self.market_value_path = market_value_path
        self.stock2concept_matrix = stock2concept_matrix
        self.stock_index = stock_index

        self.outdir = outdir
        self.overwrite = overwrite


if __name__ == '__main__':
    args = parse_args()

    main(args)
