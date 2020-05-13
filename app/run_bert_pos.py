import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataloader import DataLoader
from transformers import BertModel
from transformers.optimization import AdamW

import metrics
from data_utils import OOSDataset, PosOOSDataset
from config import Config
from logger import Logger
from metrics import plot_confusion_matrix
from processor.oos_processor import OOSProcessor
from processor.smp_processor import SMPProcessor
from processor.pos_tagging_smp_processor import PosSMPProcessor
from model.dgan import Discriminator, Generator
from utils import check_manual_seed, save_gan_model, load_gan_model, save_model, load_model, output_cases, EarlyStopping
from utils import convert_to_int_by_threshold
from utils.visualization import scatter_plot, my_plot_roc
from utils.tool import ErrorRateAt95Recall, save_result


os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


if torch.cuda.is_available():
    device = 'cuda:0'
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = 'cpu'
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


def main(args):
    check_manual_seed(args.seed)
    logger.info('seed: {}'.format(args.seed))

    logger.info('Loading config...')
    bert_config = Config('config/bert.ini')
    bert_config = bert_config(args.bert_type)

    # for oos-eval dataset
    data_config = Config('config/data.ini')
    data_config = data_config(args.dataset)

    # Prepare data processor
    data_path = os.path.join(data_config['DataDir'], data_config[args.data_file])  # 把目录和文件名合成一个路径
    label_path = data_path.replace('.json', '.label')
    with open(data_path, 'r', encoding='utf-8')as fp:
        data = json.load(fp)
        for  type in data:
            logger.info('{} : {}'.format(type, len(data[type])))
    with open(label_path, 'r', encoding='utf-8')as fp:
        logger.info(json.load(fp))

    if args.dataset == 'oos-eval':
        processor = OOSProcessor(bert_config, maxlen=32)
        logger.info('OOSProcessor')
    elif args.dataset == 'smp':
        # processor = SMPProcessor(bert_config, maxlen=32)
        processor = PosSMPProcessor(bert_config, maxlen=32)
        logger.info('SMPProcessor')
    else:
        raise ValueError('The dataset {} is not supported.'.format(args.dataset))

    processor.load_label(label_path)  # Adding label_to_id and id_to_label ot processor.
    processor.load_pos('data/pos.json')
    logger.info("label_to_id: {}".format(processor.label_to_id))
    logger.info("id_to_label: {}".format(processor.id_to_label))

    n_class = len(processor.id_to_label)
    config = vars(args)  # 返回参数字典
    config['gan_save_path'] = os.path.join(args.output_dir, 'save', 'gan.pt')
    config['bert_save_path'] = os.path.join(args.output_dir, 'save', 'bert.pt')
    config['n_class'] = n_class

    logger.info('config:')
    logger.info(config)

    from model.pos import Pos
    from model.pos_emb import Pos_emb
    E = BertModel.from_pretrained(bert_config['PreTrainModelDir'])  # Bert encoder
    config['pos_dim'] = args.pos_dim
    config['batch_size'] = args.train_batch_size
    config['n_pos'] = len(processor.pos)
    config['device'] = device
    config['nhead'] = 2
    config['num_layers'] = 2
    config['maxlen'] = processor.maxlen
    print('config', config)
    print(processor.pos)
    pos = Pos_emb(config)

    if args.fine_tune:
        for param in E.parameters():
            param.requires_grad = True
    else:
        for param in E.parameters():
            param.requires_grad = False

    pos.to(device)
    E.to(device)

    # logger.info(('pos_dim: {}, feature_dim'.format(config['pos_dim'], config['feature_dim'])))

    global_step = 0

    def train(train_dataset, dev_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

        global best_dev
        nonlocal global_step

        n_sample = len(train_dataloader)
        early_stopping = EarlyStopping(args.patience, logger=logger)
        # Loss function
        adversarial_loss = torch.nn.BCELoss().to(device)

        # Optimizers
        optimizer_pos = torch.optim.Adam(pos.parameters(), lr=args.pos_lr)
        optimizer_E = AdamW(E.parameters(), args.bert_lr)

        valid_detection_loss = []
        valid_oos_ind_precision = []
        valid_oos_ind_recall = []
        valid_oos_ind_f_score = []

        for i in range(args.n_epoch):
            logger.info('***********************************')
            logger.info('epoch: {}'.format(i))

            # Initialize model state
            pos.train()
            E.train()

            total_loss = 0
            for sample in tqdm(train_dataloader):
                sample = (i.to(device) for i in sample)
                token, mask, type_ids, pos1, pos2, pos_mask, y = sample
                batch = len(token)

                optimizer_E.zero_grad()
                optimizer_pos.zero_grad()
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                # logger.info(('size pos1: {}, pos2: {}, real_feature: {}'.format(pos1.size(), pos2.size(), real_feature.size())))
                out = pos(pos1, pos2, real_feature)

                loss = adversarial_loss(out, y.float())
                loss.backward()
                total_loss += loss.detach()

                if args.fine_tune:
                    optimizer_E.step()

                optimizer_pos.step()

            logger.info('[Epoch {}] Train: loss: {}'.format(i, total_loss / n_sample))
            logger.info('---------------------------------------------------------------------------')

            if dev_dataset:
                logger.info('#################### eval result at step {} ####################'.format(global_step))
                eval_result = eval(dev_dataset)

                valid_detection_loss.append(eval_result['detection_loss'])
                valid_oos_ind_precision.append(eval_result['oos_ind_precision'])
                valid_oos_ind_recall.append(eval_result['oos_ind_recall'])
                valid_oos_ind_f_score.append(eval_result['oos_ind_f_score'])

                # 1 表示要保存模型
                # 0 表示不需要保存模型
                # -1 表示不需要模型，且超过了patience，需要early stop
                signal = early_stopping(-eval_result['eer'])
                if signal == -1:
                    break
                # elif signal == 0:
                #     pass
                # elif signal == 1:
                #     save_gan_model(D, G, config['gan_save_path'])
                #     if args.fine_tune:
                #         save_model(E, path=config['bert_save_path'], model_name='bert')

                logger.info(eval_result)
                logger.info('valid_eer: {}'.format(eval_result['eer']))
                logger.info('valid_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
                logger.info('valid_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
                logger.info('valid_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
                logger.info('valid_auc: {}'.format(eval_result['auc']))
                logger.info(
                    'valid_fpr95: {}'.format(ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])))

        best_dev = -early_stopping.best_score

    def eval(dataset):
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict()

        detection_loss = torch.nn.BCELoss().to(device)

        pos.eval()
        E.eval()

        all_detection_preds = []

        for sample in tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, pos1, pos2, pos_mask, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector
            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                out = pos(pos1, pos2, real_feature)
                all_detection_preds.append(out)

        all_y = LongTensor(dataset.dataset[:, -4].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]

        # 计算损失
        detection_loss = detection_loss(all_detection_preds, all_binary_y.float())
        result['detection_loss'] = detection_loss

        logger.info(
            metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(
            all_detection_binary_preds, all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

        result['eer'] = eer
        result['all_detection_binary_preds'] = all_detection_binary_preds
        result['detection_acc'] = detection_acc
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['y_score'] = y_score
        result['auc'] = roc_auc_score(all_binary_y, y_score)

        return result

    def test(dataset):
        # # load BERT and GAN
        # load_gan_model(D, G, config['gan_save_path'])
        # if args.fine_tune:
        #     load_model(E, path=config['bert_save_path'], model_name='bert')
        #
        test_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(test_dataloader)
        result = dict()

        # Loss function
        detection_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        pos.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []
        all_features = []

        for sample in tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, pos1, pos2, pos_mask, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                out = pos(pos1, pos2, real_feature)
                all_detection_preds.append(out)

        all_y = LongTensor(dataset.dataset[:, -4].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]

        # 计算损失
        detection_loss = detection_loss(all_detection_preds, all_binary_y.float())
        result['detection_loss'] = detection_loss

        logger.info(
            metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(
            all_detection_binary_preds, all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

        result['eer'] = eer
        result['all_detection_binary_preds'] = all_detection_binary_preds
        result['detection_acc'] = detection_acc
        result['all_binary_y'] = all_binary_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['y_score'] = y_score
        result['auc'] = roc_auc_score(all_binary_y, y_score)

        return result

    if args.do_train:
        if config['data_file'].startswith('binary'):
            text_train_set = processor.read_dataset(data_path, ['train'])
            text_dev_set = processor.read_dataset(data_path, ['val'])
        elif config['dataset'] == 'oos-eval':
            text_train_set = processor.read_dataset(data_path, ['train', 'oos_train'])
            text_dev_set = processor.read_dataset(data_path, ['val', 'oos_val'])
        elif config['dataset'] == 'smp':
            text_train_set = processor.read_dataset(data_path, ['train'])
            text_dev_set = processor.read_dataset(data_path, ['val'])

        train_features = processor.convert_to_ids(text_train_set)
        train_dataset = PosOOSDataset(train_features)
        dev_features = processor.convert_to_ids(text_dev_set)
        dev_dataset = PosOOSDataset(dev_features)

        train(train_dataset, dev_dataset)

    if args.do_eval:
        logger.info('#################### eval result at step {} ####################'.format(global_step))
        if config['data_file'].startswith('binary'):
            text_dev_set = processor.read_dataset(data_path, ['val'])
        elif config['dataset'] == 'oos-eval':
            text_dev_set = processor.read_dataset(data_path, ['val', 'oos_val'])
        elif config['dataset'] == 'smp':
            text_dev_set = processor.read_dataset(data_path, ['val'])

        dev_features = processor.convert_to_ids(text_dev_set)
        dev_dataset = PosOOSDataset(dev_features)
        eval_result = eval(dev_dataset)
        logger.info(eval_result)
        logger.info('eval_eer: {}'.format(eval_result['eer']))
        logger.info('eval_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
        logger.info('eval_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
        logger.info('eval_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
        logger.info('eval_auc: {}'.format(eval_result['auc']))
        logger.info(
            'eval_fpr95: {}'.format(ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])))

    if args.do_test:
        logger.info('#################### test result at step {} ####################'.format(global_step))
        if config['data_file'].startswith('binary'):
            text_test_set = processor.read_dataset(data_path, ['test'])
        elif config['dataset'] == 'oos-eval':
            text_test_set = processor.read_dataset(data_path, ['test', 'oos_test'])
        elif config['dataset'] == 'smp':
            text_test_set = processor.read_dataset(data_path, ['test'])

        test_features = processor.convert_to_ids(text_test_set)
        test_dataset = PosOOSDataset(test_features)
        test_result = test(test_dataset)
        logger.info(test_result)
        logger.info('test_eer: {}'.format(test_result['eer']))
        logger.info('test_ood_ind_precision: {}'.format(test_result['oos_ind_precision']))
        logger.info('test_ood_ind_recall: {}'.format(test_result['oos_ind_recall']))
        logger.info('test_ood_ind_f_score: {}'.format(test_result['oos_ind_f_score']))
        logger.info('test_auc: {}'.format(test_result['auc']))
        logger.info('test_fpr95: {}'.format(ErrorRateAt95Recall(test_result['all_binary_y'], test_result['y_score'])))
        my_plot_roc(test_result['all_binary_y'], test_result['y_score'],
                    os.path.join(args.output_dir, 'roc_curve.png'))
        save_result(test_result, os.path.join(args.output_dir, 'test_result'))


        # 输出错误cases
        if config['dataset'] == 'oos-eval':
            texts = [line[0] for line in text_test_set]
        elif config['dataset'] == 'smp':
            texts = [line['text'] for line in text_test_set]
        else:
            raise ValueError('The dataset {} is not supported.'.format(args.dataset))

        output_cases(texts, test_result['all_binary_y'], test_result['all_detection_binary_preds'],
                     os.path.join(args.output_dir, 'test_cases.csv'), processor)

        # confusion matrix
        plot_confusion_matrix(test_result['all_binary_y'], test_result['all_detection_binary_preds'],
                              args.output_dir)

        beta_log_path = 'beta_log.txt'
        if os.path.exists(beta_log_path):
            flag = True
        else:
            flag = False
        with open(beta_log_path, 'a', encoding='utf-8') as f:
            if flag == False:
                f.write('seed\tdataset\tdev_eer\ttest_eer\tdata_size\n')
            line = '\t'.join([str(config['seed']), str(config['data_file']), str(best_dev), str(test_result['eer']), '100'])
            f.write(line + '\n')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--data_file', type=str, required=True)

    parser.add_argument('--bert_type', type=str, required=True)

    parser.add_argument('--feature_dim', type=int, default=768)


    parser.add_argument('--do_train', action='store_true')

    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--n_epoch', type=int, default=50)

    parser.add_argument('--patience', default=10, type=int)

    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='Batch size for training.')

    parser.add_argument('--predict_batch_size', default=32, type=int,
                        help='Batch size for evaluating and testing.')

    parser.add_argument('--pos_lr', type=float, default=2e-5)
    parser.add_argument('--pos_dim', type=int)
    parser.add_argument('--bert_lr', type=float, default=5e-5, help="Learning rate for Generator.")
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine tune BERT during training.')
    parser.add_argument('--seed', type=int, default=123, help='seed')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)