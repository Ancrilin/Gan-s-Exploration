# coding: utf-8
# @author: Ross
# @file: run_gan.py
# @time: 2020/01/14
# @contact: devross@gmail.com
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader
from transformers import BertModel
from transformers.optimization import AdamW
from importlib import import_module

import metrics
from config import Config
from data_utils import OOSDataset
from logger import Logger
from metrics import plot_confusion_matrix
from processor.oos_processor import OOSProcessor
from processor.smp_processor import SMPProcessor
from utils import check_manual_seed, save_gan_model, load_gan_model, save_model, load_model, output_cases, EarlyStopping
from utils import convert_to_int_by_threshold
from utils.visualization import scatter_plot, my_plot_roc
from utils.tool import ErrorRateAt95Recall, save_result, save_feature

SEED = 123
freeze_data = dict()
best_dev = -1

if torch.cuda.is_available():
    device = 'cuda'
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


def main(args):
    logger.info('Checking...')
    SEED = args.seed
    logger.info('seed: {}'.format(SEED))
    logger.info('model: {}'.format(args.model))
    check_manual_seed(SEED)
    check_args(args)

    logger.info('Loading config...')
    bert_config = Config('config/bert.ini')
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
    config['gan_save_path'] = os.path.join(args.output_dir, 'save', 'gan.pt')
    config['bert_save_path'] = os.path.join(args.output_dir, 'save', 'bert.pt')
    config['n_class'] = n_class

    logger.info('config:')
    logger.info(config)

    model = import_module('model.' + args.model)

    D = model.Discriminator(config)
    G = model.Generator(config)
    E = BertModel.from_pretrained(bert_config['PreTrainModelDir'])  # Bert encoder

    logger.info('Discriminator: {}'.format(D))
    logger.info('Generator: {}'.format(G))

    if args.fine_tune:
        for param in E.parameters():
            param.requires_grad = True
    else:
        for param in E.parameters():
            param.requires_grad = False

    D.to(device)
    G.to(device)
    E.to(device)

    global_step = 0

    def train(train_dataset, dev_dataset):
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)

        global best_dev
        nonlocal global_step
        n_sample = len(train_dataloader)
        early_stopping = EarlyStopping(args.patience, logger=logger)
        # Loss function
        adversarial_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss().to(device)

        # Optimizers
        optimizer_G = torch.optim.Adam(G.parameters(), lr=args.G_lr)  # optimizer for generator
        optimizer_D = torch.optim.Adam(D.parameters(), lr=args.D_lr)  # optimizer for discriminator
        optimizer_E = AdamW(E.parameters(), args.bert_lr)

        G_total_train_loss = []
        D_total_fake_loss = []
        D_total_real_loss = []
        FM_total_train_loss = []
        D_total_class_loss = []
        valid_detection_loss = []
        valid_oos_ind_precision = []
        valid_oos_ind_recall = []
        valid_oos_ind_f_score = []

        all_features = []
        result = dict()

        for i in range(args.n_epoch):

            # Initialize model state
            G.train()
            D.train()
            E.train()

            G_train_loss = 0
            D_fake_loss = 0
            D_real_loss = 0
            FM_train_loss = 0
            D_class_loss = 0

            for sample in tqdm.tqdm(train_dataloader):
                sample = (i.to(device) for i in sample)
                token, mask, type_ids, y = sample
                batch = len(token)

                ood_sample = (y==0.0)
                # weight = torch.ones(len(ood_sample)).to(device) - ood_sample * args.beta
                # real_loss_func = torch.nn.BCELoss(weight=weight).to(device)

                # the label used to train generator and discriminator.
                valid_label = FloatTensor(batch, 1).fill_(1.0).detach()
                fake_label = FloatTensor(batch, 1).fill_(0.0).detach()

                optimizer_E.zero_grad()
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                # train D on real
                optimizer_D.zero_grad()
                real_f_vector, discriminator_output, classification_output = D(real_feature, return_feature=True)
                discriminator_output = discriminator_output.squeeze()
                real_loss = adversarial_loss(discriminator_output, (y != 0.0).float())
                # real_loss = real_loss_func(discriminator_output, (y != 0.0).float())
                if n_class > 2:  # 大于2表示除了训练判别器还要训练分类器
                    class_loss = classified_loss(classification_output, y.long())
                    real_loss += class_loss
                    D_class_loss += class_loss.detach()
                real_loss.backward()

                if args.do_vis:
                    all_features.append(real_f_vector.detach())

                # 除去 G
                # # # train D on fake
                # if args.model == 'lstm_gan' or args.model == 'cnn_gan':
                #     z = FloatTensor(np.random.normal(0, 1, (batch, 32, args.G_z_dim))).to(device)
                # else:
                #     z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                # fake_feature = G(z).detach()
                # fake_discriminator_output = D.detect_only(fake_feature)
                # # fake_loss = args.beta * adversarial_loss(fake_discriminator_output, fake_label)
                # fake_loss = adversarial_loss(fake_discriminator_output, fake_label)
                # fake_loss.backward()
                optimizer_D.step()

                if args.fine_tune:
                    optimizer_E.step()

                # 除去 G
                # # train G
                # optimizer_G.zero_grad()
                # if args.model == 'lstm_gan' or args.model == 'cnn_gan':
                #     z = FloatTensor(np.random.normal(0, 1, (batch, 32, args.G_z_dim))).to(device)
                # else:
                #     z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                # fake_f_vector, D_decision = D.detect_only(G(z), return_feature=True)
                # gd_loss = adversarial_loss(D_decision, valid_label)
                # fm_loss = torch.abs(torch.mean(real_f_vector.detach(), 0) - torch.mean(fake_f_vector, 0)).mean()
                # g_loss = gd_loss + 0 * fm_loss
                # g_loss.backward()
                # optimizer_G.step()

                global_step += 1

                # D_fake_loss += fake_loss.detach()
                D_real_loss += real_loss.detach()
                # G_train_loss += g_loss.detach() + fm_loss.detach()
                # FM_train_loss += fm_loss.detach()

            logger.info('[Epoch {}] Train: D_fake_loss: {}'.format(i, D_fake_loss / n_sample))
            logger.info('[Epoch {}] Train: D_real_loss: {}'.format(i, D_real_loss / n_sample))
            logger.info('[Epoch {}] Train: D_class_loss: {}'.format(i, D_class_loss / n_sample))
            logger.info('[Epoch {}] Train: G_train_loss: {}'.format(i, G_train_loss / n_sample))
            logger.info('[Epoch {}] Train: FM_train_loss: {}'.format(i, FM_train_loss / n_sample))
            logger.info('---------------------------------------------------------------------------')

            D_total_fake_loss.append(D_fake_loss / n_sample)
            D_total_real_loss.append(D_real_loss / n_sample)
            D_total_class_loss.append(D_class_loss / n_sample)
            G_total_train_loss.append(G_train_loss / n_sample)
            FM_total_train_loss.append(FM_train_loss / n_sample)

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
                elif signal == 0:
                    pass
                elif signal == 1:
                    save_gan_model(D, G, config['gan_save_path'])
                    if args.fine_tune:
                        save_model(E, path=config['bert_save_path'], model_name='bert')

                logger.info(eval_result)
                logger.info('valid_eer: {}'.format(eval_result['eer']))
                logger.info('valid_oos_ind_precision: {}'.format(eval_result['oos_ind_precision']))
                logger.info('valid_oos_ind_recall: {}'.format(eval_result['oos_ind_recall']))
                logger.info('valid_oos_ind_f_score: {}'.format(eval_result['oos_ind_f_score']))
                logger.info(
                    'valid_fpr95: {}'.format(ErrorRateAt95Recall(eval_result['all_binary_y'], eval_result['y_score'])))

        if args.patience >= args.n_epoch:
            save_gan_model(D, G, config['gan_save_path'])
            if args.fine_tune:
                save_model(E, path=config['bert_save_path'], model_name='bert')

        freeze_data['D_total_fake_loss'] = D_total_fake_loss
        freeze_data['D_total_real_loss'] = D_total_real_loss
        freeze_data['D_total_class_loss'] = D_total_class_loss
        freeze_data['G_total_train_loss'] = G_total_train_loss
        freeze_data['FM_total_train_loss'] = FM_total_train_loss
        freeze_data['valid_real_loss'] = valid_detection_loss
        freeze_data['valid_oos_ind_precision'] = valid_oos_ind_precision
        freeze_data['valid_oos_ind_recall'] = valid_oos_ind_recall
        freeze_data['valid_oos_ind_f_score'] = valid_oos_ind_f_score

        best_dev = -early_stopping.best_score

        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features
        return result

    def eval(dataset):
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict()

        # Loss function
        detection_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        G.eval()
        D.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []

        for sample in tqdm.tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                # 大于2表示除了训练判别器还要训练分类器
                if n_class > 2:
                    f_vector, discriminator_output, classification_output = D(real_feature, return_feature=True)
                    all_detection_preds.append(discriminator_output)
                    all_class_preds.append(classification_output)

                # 只预测判别器
                else:
                    f_vector, discriminator_output = D.detect_only(real_feature, return_feature=True)
                    all_detection_preds.append(discriminator_output)

        all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
        all_binary_y = (all_y != 0).long()  # [length, 1] label 0 is oos
        all_detection_preds = torch.cat(all_detection_preds, 0).cpu()  # [length, 1]
        all_detection_binary_preds = convert_to_int_by_threshold(all_detection_preds.squeeze())  # [length, 1]

        # 计算损失
        detection_loss = detection_loss(all_detection_preds, all_binary_y.float())
        result['detection_loss'] = detection_loss

        if n_class > 2:
            class_one_hot_preds = torch.cat(all_class_preds, 0).detach().cpu()  # one hot label
            class_loss = classified_loss(class_one_hot_preds, all_y)  # compute loss
            all_class_preds = torch.argmax(class_one_hot_preds, 1)  # label
            class_acc = metrics.ind_class_accuracy(all_class_preds, all_y, oos_index=0)  # accuracy for ind class
            logger.info(metrics.classification_report(all_y, all_class_preds, target_names=processor.id_to_label))

        logger.info(metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(all_detection_binary_preds, all_binary_y)
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
        if n_class > 2:
            result['class_loss'] = class_loss
            result['class_acc'] = class_acc

        freeze_data['valid_all_y'] = all_y
        freeze_data['vaild_all_pred'] = all_detection_binary_preds
        freeze_data['valid_score'] = y_score

        return result

    def test(dataset):
        # load BERT and GAN
        load_gan_model(D, G, config['gan_save_path'])
        if args.fine_tune:
            load_model(E, path=config['bert_save_path'], model_name='bert')

        test_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(test_dataloader)
        result = dict()

        # Loss function
        detection_loss = torch.nn.BCELoss().to(device)
        classified_loss = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)

        G.eval()
        D.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []
        all_features = []

        for sample in tqdm.tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                # 大于2表示除了训练判别器还要训练分类器
                if n_class > 2:
                    f_vector, discriminator_output, classification_output = D(real_feature, return_feature=True)
                    all_detection_preds.append(discriminator_output)
                    all_class_preds.append(classification_output)

                # 只预测判别器
                else:
                    f_vector, discriminator_output = D.detect_only(real_feature, return_feature=True)
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

        if n_class > 2:
            class_one_hot_preds = torch.cat(all_class_preds, 0).detach().cpu()  # one hot label
            class_loss = classified_loss(class_one_hot_preds, all_y)  # compute loss
            all_class_preds = torch.argmax(class_one_hot_preds, 1)  # label
            class_acc = metrics.ind_class_accuracy(all_class_preds, all_y, oos_index=0)  # accuracy for ind class
            logger.info(metrics.classification_report(all_y, all_class_preds, target_names=processor.id_to_label))

        logger.info(metrics.classification_report(all_binary_y, all_detection_binary_preds, target_names=['oos', 'in']))

        # report
        oos_ind_precision, oos_ind_recall, oos_ind_fscore, _ = metrics.binary_recall_fscore(all_detection_binary_preds, all_binary_y)
        detection_acc = metrics.accuracy(all_detection_binary_preds, all_binary_y)

        y_score = all_detection_preds.squeeze().tolist()
        eer = metrics.cal_eer(all_binary_y, y_score)

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
        if n_class > 2:
            result['class_loss'] = class_loss
            result['class_acc'] = class_acc
        if args.do_vis:
            all_features = torch.cat(all_features, 0).cpu().numpy()
            result['all_features'] = all_features

        freeze_data['test_all_y'] = all_y.tolist()
        freeze_data['test_all_pred'] = all_detection_binary_preds.tolist()
        freeze_data['test_score'] = y_score

        return result

    def get_fake_feature(num_output):
        """
        生成一定数量的假特征
        """
        G.eval()
        fake_features = []
        start = 0
        batch = args.predict_batch_size
        with torch.no_grad():
            while start < num_output:
                end = min(num_output, start + batch)
                if args.model == 'lstm_gan' or args.model == 'cnn_gan':
                    z = FloatTensor(np.random.normal(0, 1, size=(end - start, 32, args.G_z_dim)))
                else:
                    z = FloatTensor(np.random.normal(0, 1, size=(end - start, args.G_z_dim)))
                fake_feature = G(z)
                f_vector, _ = D.detect_only(fake_feature, return_feature=True)
                fake_features.append(f_vector)
                start += batch
            return torch.cat(fake_features, 0).cpu().numpy()

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

        train_result = train(train_dataset, dev_dataset)
        save_feature(train_result['all_features'], os.path.join(args.output_dir, 'train_feature'))


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
        logger.info('test_fpr95: {}'.format(ErrorRateAt95Recall(test_result['all_binary_y'], test_result['y_score'])))
        my_plot_roc(test_result['all_binary_y'], test_result['y_score'],
                    os.path.join(args.output_dir, 'roc_curve.png'))
        save_result(test_result, os.path.join(args.output_dir, 'test_result'))
        save_feature(test_result['all_features'], os.path.join(args.output_dir, 'test_feature'))

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

        if args.do_vis:
            # [2 * length, feature_fim]
            features = np.concatenate([test_result['all_features'], get_fake_feature(len(test_dataset) // 2)], axis=0)
            features = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features)  # [2 * length, 2]
            # [2 * length, １]
            if n_class > 2:
                labels = np.concatenate([test_result['all_y'], np.array([-1] * (len(test_dataset) // 2))], 0).reshape((-1, 1))
            else:
                labels = np.concatenate([test_result['all_binary_y'], np.array([-1] * (len(test_dataset) // 2))], 0).reshape((-1, 1))
            # [2 * length, 3]
            data = np.concatenate([features, labels], 1)
            fig = scatter_plot(data, processor)
            fig.savefig(os.path.join(args.output_dir, 'plot.png'))
            fig.show()
            freeze_data['feature_label'] = data

    with open(os.path.join(config['output_dir'], 'freeze_data.pkl'), 'wb') as f:
        pickle.dump(freeze_data, f)
    df = pd.DataFrame(data={'valid_y': freeze_data['valid_all_y'],
                            'valid_score': freeze_data['valid_score'],
                            })
    df.to_csv(os.path.join(config['output_dir'], 'valid_score.csv'))

    df = pd.DataFrame(data={'test_y': freeze_data['test_all_y'],
                            'test_score': freeze_data['test_score']
                            })
    df.to_csv(os.path.join(config['output_dir'], 'test_score.csv'))


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
                        choices={'bert-base-uncased', 'bert-large-uncased', 'bert-base-chinese'}, required=True,
                        help='Type of the pre-trained BERT to be used.')

    # ------------------------Discriminator------------------------ #
    parser.add_argument('--D_Wf_dim', default=512, type=int,
                        help='The Dimension of Wf for Discriminator.')

    # ------------------------Generator------------------------ #
    parser.add_argument('--G_z_dim', default=512, type=int,
                        help='The Dimension of z (noise) for Generator.')

    parser.add_argument('--feature_dim', default=768, type=int,
                        help='The Dimension of feature vector for Generator output and Discriminator input.')

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

    parser.add_argument('--train_batch_size', default=64, type=int,
                        help='Batch size for training.')

    parser.add_argument('--predict_batch_size', default=32, type=int,
                        help='Batch size for evaluating and testing.')

    parser.add_argument('--D_lr', type=float, default=1e-5, help="Learning rate for Discriminator.")
    parser.add_argument('--G_lr', type=float, default=1e-5, help="Learning rate for Generator.")
    parser.add_argument('--beta', type=float, default=0.1, help="Weight of fake sample loss for Discriminator.")

    parser.add_argument('--bert_lr', type=float, default=2e-4, help="Learning rate for Generator.")
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine tune BERT during training.')
    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--model', type=str, required=True,
                        choices={'gan', 'dgan', 'lstm_gan', 'cnn_gan'},
                        help='choose gan model')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)
