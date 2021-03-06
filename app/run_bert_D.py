# coding: utf-8
# @author: Ross
# @file: run_gan.py
# @time: 2020/01/14
# @contact: devross@gmail.com
import argparse
import os
import pickle

import pandas as pd
import torch
import tqdm
from torch.utils.data.dataloader import DataLoader
from transformers.optimization import AdamW
from sklearn.metrics import roc_auc_score

import metrics
from config import Config, BertConfig
from data_utils import OOSDataset
from logger import Logger
from metrics import plot_confusion_matrix
from model.bert_D import BertClassifier
from processor.oos_processor import OOSProcessor
from processor.smp_processor import SMPProcessor
from utils import check_manual_seed, save_model, load_model, output_cases, EarlyStopping, convert_to_int_by_threshold
from utils.tool import ErrorRateAt95Recall, save_result

freeze_data = dict()
SEED = 123

if torch.cuda.is_available():
    device = 'cuda:0'
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
else:
    device = 'cpu'
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor


def check_args(args):
    """to check if the args is ok"""
    if not (args.do_train or args.do_eval or args.do_test):
        raise argparse.ArgumentError('You should pass at least one argument for --do_train or --do_eval or --do_test')
    if args.gradient_accumulation_steps < 1 or args.train_batch_size % args.gradient_accumulation_steps != 0:
        raise argparse.ArgumentError('Gradient_accumulation_steps should >=1 and train_batch_size%gradient_accumulation_steps == 0')


def main(args):
    logger.info('Checking...')
    check_manual_seed(args.seed)
    check_args(args)

    logger.info('Loading config...')
    bert_config = BertConfig('config/bert.ini')
    bert_config = bert_config(args.bert_type)

    # for oos-eval dataset
    data_config = Config('config/data.ini')
    data_config = data_config(args.dataset)

    # Prepare data processor
    data_path = os.path.join(data_config['DataDir'], data_config[args.data_file])  # 把目录和文件名合成一个路径
    label_path = data_path.replace('.json', '.label')

    if args.dataset == 'oos-eval':
        processor = OOSProcessor(bert_config, maxlen=32)
    elif args.dataset == 'smp':
        processor = SMPProcessor(bert_config, maxlen=32)
    else:
        raise ValueError('The dataset {} is not supported.'.format(args.dataset))

    processor.load_label(label_path)  # Adding label_to_id and id_to_label ot processor.

    n_class = len(processor.id_to_label)
    config = vars(args)  # 返回参数字典
    config['model_save_path'] = os.path.join(args.output_dir, 'save', 'bert.pt')
    config['n_class'] = n_class

    logger.info('config:')
    logger.info(config)

    model = BertClassifier(bert_config, config)  # Bert encoder
    if args.fine_tune:
        model.unfreeze_bert_encoder()
    else:
        model.freeze_bert_encoder()
    model.to(device)

    global_step = 0

    def train(train_dataset, dev_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size // args.gradient_accumulation_steps, shuffle=True,
                                      num_workers=2)

        nonlocal global_step
        n_sample = len(train_dataloader)
        early_stopping = EarlyStopping(args.patience, logger=logger)
        # Loss function
        classified_loss = torch.nn.CrossEntropyLoss().to(device)
        adversarial_loss = torch.nn.BCELoss().to(device)

        # Optimizers
        optimizer = AdamW(model.parameters(), args.lr)

        train_loss = []
        if dev_dataset:
            valid_loss = []
            valid_ind_class_acc = []
        iteration = 0
        for i in range(args.n_epoch):

            model.train()

            total_loss = 0
            for sample in tqdm.tqdm(train_dataloader):
                sample = (i.to(device) for i in sample)
                token, mask, type_ids, y = sample
                batch = len(token)

                f_vector, discriminator_output, classification_output = model(token, mask, type_ids, return_feature=True)
                discriminator_output = discriminator_output.squeeze()
                if args.BCE:
                    loss = adversarial_loss(discriminator_output, (y != 0.0).float())
                else:
                    loss = classified_loss(discriminator_output, y.long())
                total_loss += loss.item()
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                # bp and update parameters
                if (global_step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info('[Epoch {}] Train: train_loss: {}'.format(i, total_loss / n_sample))
            logger.info('-' * 30)

            train_loss.append(total_loss / n_sample)
            iteration += 1

            if dev_dataset:
                logger.info('#################### eval result at step {} ####################'.format(global_step))
                eval_result = eval(dev_dataset)

                valid_loss.append(eval_result['loss'])
                valid_ind_class_acc.append(eval_result['ind_class_acc'])

                # 1 表示要保存模型
                # 0 表示不需要保存模型
                # -1 表示不需要模型，且超过了patience，需要early stop
                signal = early_stopping(-eval_result['eer'])
                if signal == -1:
                    break
                elif signal == 0:
                    pass
                elif signal == 1:
                    save_model(model, path=config['model_save_path'], model_name='bert')

                logger.info(eval_result)
                logger.info('valid_eer: {}'.format(eval_result['eer']))
                logger.info('valid_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
                logger.info('valid_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
                logger.info('valid_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
                logger.info('valid_auc: {}'.format(eval_result['auc']))
                logger.info(
                    'valid_fpr95: {}'.format(ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])))

        from utils.visualization import draw_curve
        draw_curve(train_loss, iteration, 'train_loss', args.output_dir)
        if dev_dataset:
            draw_curve(valid_loss, iteration, 'valid_loss', args.output_dir)
            draw_curve(valid_ind_class_acc, iteration, 'valid_ind_class_accuracy', args.output_dir)

        if args.patience >= args.n_epoch:
            save_model(model, path=config['model_save_path'], model_name='bert')

        freeze_data['train_loss'] = train_loss
        freeze_data['valid_loss'] = valid_loss

    def eval(dataset):
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict()
        model.eval()

        # Loss function
        classified_loss = torch.nn.CrossEntropyLoss().to(device)
        detection_loss = torch.nn.BCELoss().to(device)
        all_detection_preds = []
        all_class_preds = []
        all_pred = []
        all_logit = []
        total_loss = 0
        for sample in tqdm.tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            with torch.no_grad():
                f_vector, discriminator_output, classification_output = model(token, mask, type_ids, return_feature=True)
                discriminator_output = discriminator_output.squeeze()
                all_detection_preds.append(discriminator_output)

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]
        # 计算损失
        detection_loss = detection_loss(all_detection_preds, all_binary_y.float())
        result['detection_loss'] = detection_loss

        logger.info(metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(all_detection_binary_preds,
                                                                                            all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

        ind_class_acc = metrics.ind_class_accuracy(all_detection_binary_preds, all_y)

        result['ind_class_acc'] = ind_class_acc
        result['loss'] = total_loss / n_sample

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
        load_model(model, path=config['model_save_path'], model_name='bert')
        test_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(test_dataloader)
        result = dict()
        model.eval()

        # Loss function
        classified_loss = torch.nn.CrossEntropyLoss().to(device)
        detection_loss = torch.nn.BCELoss().to(device)
        all_detection_preds = []
        all_features = []
        all_pred = []
        total_loss = 0
        all_logit = []
        for sample in tqdm.tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            with torch.no_grad():
                f_vector, discriminator_output, classification_output = model(token, mask, type_ids, return_feature=True)
                discriminator_output = discriminator_output.squeeze()
                all_detection_preds.append(discriminator_output)
                if args.do_vis:
                    all_features.append(f_vector)

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]

        # 计算损失
        detection_loss = detection_loss(all_detection_preds, all_binary_y.float())
        result['detection_loss'] = detection_loss

        logger.info(metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(all_detection_binary_preds,
                                                                                            all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features

        ind_class_acc = metrics.ind_class_accuracy(all_detection_binary_preds, all_y)

        result['ind_class_acc'] = ind_class_acc
        result['loss'] = total_loss / n_sample

        result['eer'] = eer
        result['all_detection_binary_preds'] = all_detection_binary_preds
        result['detection_acc'] = detection_acc
        result['all_binary_y'] = all_binary_y
        result['all_y'] = all_y
        result['oos_ind_precision'] = oos_ind_precision
        result['oos_ind_recall'] = oos_ind_recall
        result['oos_ind_f_score'] = oos_ind_fscore
        result['score'] = y_score
        result['y_score'] = y_score
        result['all_pred'] = all_detection_binary_preds
        result['auc'] = roc_auc_score(all_binary_y, y_score)


        freeze_data['test_all_y'] = all_y.tolist()
        freeze_data['test_all_pred'] = all_detection_binary_preds.tolist()
        freeze_data['test_score'] = y_score

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
        train_dataset = OOSDataset(train_features)
        dev_features = processor.convert_to_ids(text_dev_set)
        dev_dataset = OOSDataset(dev_features)

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
        dev_dataset = OOSDataset(dev_features)
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
        test_dataset = OOSDataset(test_features)
        test_result = test(test_dataset)
        logger.info(test_result)
        logger.info('test_eer: {}'.format(test_result['eer']))
        logger.info('test_ood_ind_precision: {}'.format(test_result['oos_ind_precision']))
        logger.info('test_ood_ind_recall: {}'.format(test_result['oos_ind_recall']))
        logger.info('test_ood_ind_f_score: {}'.format(test_result['oos_ind_f_score']))
        logger.info('test_auc: {}'.format(test_result['auc']))
        logger.info('test_fpr95: {}'.format(ErrorRateAt95Recall(test_result['all_binary_y'], test_result['y_score'])))

        # 输出错误cases
        if config['dataset'] == 'oos-eval':
            texts = [line[0] for line in text_test_set]
        elif config['dataset'] == 'smp':
            texts = [line['text'] for line in text_test_set]
        else:
            raise ValueError('The dataset {} is not supported.'.format(args.dataset))

        # output_cases(texts, test_result['all_y'], test_result['all_pred'],
        #              os.path.join(args.output_dir, 'test_cases.csv'), processor, test_result['test_logit'])

        # confusion matrix
        plot_confusion_matrix(test_result['all_y'], test_result['all_pred'],
                              args.output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ------------------------data------------------------ #
    parser.add_argument('--dataset',
                        choices={'oos-eval', 'smp'}, required=True,
                        help='Which dataset will be used.')

    parser.add_argument('--data_file', required=False, type=str,
                        help="""Which type of dataset to be used, 
                        i.e. binary_undersample.json, binary_wiki_aug.json. Detail in config/data.ini""")

    # ------------------------bert------------------------ #
    parser.add_argument('--bert_type',
                        choices={'bert-base-uncased', 'bert-large-uncased', 'bert-base-chinese', 'chinese-bert-wwm'}, required=True,
                        help='Type of the pre-trained BERT to be used.')

    # ------------------------action------------------------ #
    parser.add_argument('--do_train', action='store_true',
                        help='Do training step')

    parser.add_argument('--do_eval', action='store_true',
                        help='Do evaluation on devset step')

    parser.add_argument('--do_test', action='store_true',
                        help='Do validation on testset step')

    parser.add_argument('--do_vis', action='store_true',
                        help='Do visualization.')

    parser.add_argument('--output_dir', required=True,
                        help='The output directory saving model and logging file.')

    parser.add_argument('--n_epoch', default=500, type=int,
                        help='Number of epoch for training.')

    parser.add_argument('--patience', default=10, type=int,
                        help='Number of epoch of early stopping patience.')

    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='Batch size for training.')

    parser.add_argument('--predict_batch_size', default=16, type=int,
                        help='Batch size for evaluating and testing.')

    parser.add_argument('--gradient_accumulation_steps', default=1, type=int,
                        help='Number of updates steps to accumulate before performing a backward/update pass')

    parser.add_argument('--lr', type=float, default=4e-5,
                        help="Learning rate for Discriminator.")

    parser.add_argument('--feature_dim', default=768, type=int,
                        help='The Dimension of feature vector for Generator output and Discriminator input.')

    parser.add_argument('--D_Wf_dim', default=512, type=int,
                        help='The Dimension of Wf for Discriminator.')

    parser.add_argument('--seed', default=123, type=int)

    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine tune BERT during training.')
    parser.add_argument('--BCE', action='store_true', default=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)
