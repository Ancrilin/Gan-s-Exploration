import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.manifold import TSNE
from torch.utils.data.dataloader import DataLoader
from transformers import BertModel
from transformers.optimization import AdamW

import metrics
from data_utils import OOSDataset
from config import Config
from logger import Logger
from metrics import plot_confusion_matrix
from processor.oos_processor import OOSProcessor
from processor.smp_processor import SMPProcessor
from model.dgan import Discriminator, Generator
from utils import check_manual_seed, save_gan_model, load_gan_model, save_model, load_model, output_cases, EarlyStopping
from utils import convert_to_int_by_threshold
from utils.visualization import scatter_plot


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

    D_detect = Discriminator(config)
    D_g = Discriminator(config)
    G = Generator(config)
    E = BertModel.from_pretrained(bert_config['PreTrainModelDir'])  # Bert encoder

    if args.fine_tune:
        for param in E.parameters():
            param.requires_grad = True
    else:
        for param in E.parameters():
            param.requires_grad = False

    D_detect.to(device)
    D_g.to(device)
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
        optimizer_D_detect = torch.optim.Adam(D_detect.parameters(), lr=args.D_detect_lr)  # optimizer for discriminator
        optimizer_D_g = torch.optim.Adam(D_g.parameters(), lr=args.D_g_lr)
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

        for i in range(args.n_epoch):

            # Initialize model state
            G.train()
            D_detect.train()
            D_g.train()
            E.train()

            D_g_real_loss = 0
            D_g_fake_loss = 0
            D_detect_real_loss = 0
            D_detect_fake_loss = 0
            G_loss = 0


            for sample in tqdm(train_dataloader):
                sample = (i.to(device) for i in sample)
                token, mask, type_ids, y = sample
                batch = len(token)

                # the label used to train generator and discriminator.
                valid_label = FloatTensor(batch, 1).fill_(1.0).detach()
                fake_label = FloatTensor(batch, 1).fill_(0.0).detach()

                optimizer_E.zero_grad()
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                #------------------------- train D_g -------------------------#
                # train on D_g real
                optimizer_D_g.zero_grad()
                D_gen_real_discriminator_output, f_vector = D_g(real_feature)
                D_gen_real_loss = adversarial_loss(D_gen_real_discriminator_output, valid_label) # 判别器对真实样本的损失

                # train on D_g fake
                z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                fake_feature = G(z).detach()
                D_gen_fake_discriminator_output, f_vector = D_g(fake_feature)
                D_gen_fake_loss = adversarial_loss(D_gen_fake_discriminator_output, fake_label) # 判别器对假样本的损失

                D_gen_loss = D_gen_real_loss + D_gen_fake_loss
                D_gen_loss.backward(retain_graph=True)# 保存计算图，生成器还要使用
                optimizer_D_g.step()

                # ------------------------- train D_ood -------------------------#
                # train on real(detect real sample)
                optimizer_D_detect.zero_grad()
                ood_real_detect_discriminator_output, f_vector = D_detect(real_feature)
                ood_real_detect_loss = adversarial_loss(ood_real_detect_discriminator_output, (y != 0.0).float()) # ood_判别器的损失

                # train on fake(detect fake sample)
                z = FloatTensor(np.random.normal(0, 1, (batch, args.G_z_dim))).to(device)
                fake_feature = G(z).detach()
                ood_fake_detect_discriminator_output, f_vector = D_detect(fake_feature)
                ood_fake_detect_loss = adversarial_loss(ood_fake_detect_discriminator_output, fake_label)# 假样本趋向ood

                D_ood_loss = args.beta * ood_real_detect_loss + (1 - args.beta) * ood_fake_detect_loss# 真实样本与假样本比例
                D_ood_loss.backward()
                optimizer_D_detect.step()

                if args.fine_tune:
                    optimizer_E.step()

                # ------------------------- train G -------------------------#
                optimizer_G.zero_grad()
                g_D_g_loss = adversarial_loss(D_gen_fake_discriminator_output, valid_label)# 生成器趋向真实样本
                g_D_g_loss.backward()
                optimizer_G.step()

                global_step += 1

                D_g_real_loss += D_gen_real_loss.detach()
                D_g_fake_loss += D_gen_fake_loss.detach()
                D_detect_real_loss += ood_real_detect_loss.detach()
                D_detect_fake_loss +=  ood_fake_detect_loss.detach()
                G_loss += g_D_g_loss.detach()

            logger.info('[Epoch {}] Train: D_g_real_loss: {}'.format(i, D_g_real_loss / n_sample))
            logger.info('[Epoch {}] Train: D_g_fake_loss: {}'.format(i, D_g_fake_loss / n_sample))
            logger.info('[Epoch {}] Train: D_detect_real_loss: {}'.format(i, D_detect_real_loss / n_sample))
            logger.info('[Epoch {}] Train: D_detect_fake_loss: {}'.format(i, D_detect_fake_loss / n_sample))
            logger.info('[Epoch {}] Train: G_loss: {}'.format(i, G_loss / n_sample))
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

        best_dev = -early_stopping.best_score

    def eval(dataset):
        dev_dataloader = DataLoader(dataset, batch_size=args.predict_batch_size, shuffle=False, num_workers=2)
        n_sample = len(dev_dataloader)
        result = dict()

        detection_loss = torch.nn.BCELoss().to(device)

        D_detect.eval()
        E.eval()

        all_detection_preds = []

        for sample in tqdm(dev_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector
            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                discriminator_output, f_vector = D_detect(real_feature)
                all_detection_preds.append(discriminator_output)

            all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
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

        D_detect.eval()
        E.eval()

        all_detection_preds = []
        all_class_preds = []
        all_features = []

        for sample in tqdm(test_dataloader):
            sample = (i.to(device) for i in sample)
            token, mask, type_ids, y = sample
            batch = len(token)

            # -------------------------evaluate D------------------------- #
            # BERT encode sentence to feature vector

            with torch.no_grad():
                sequence_output, pooled_output = E(token, mask, type_ids)
                real_feature = pooled_output

                discriminator_output, f_vector = D_detect(real_feature)
                all_detection_preds.append(discriminator_output)

            all_y = LongTensor(dataset.dataset[:, -1].astype(int)).cpu()  # [length, n_class]
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
            if args.do_vis:
                all_features = torch.cat(all_features, 0).cpu().numpy()
                result['all_features'] = all_features

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
                z = FloatTensor(np.random.normal(0, 1, size=(end - start, args.G_z_dim)))
                fake_feature = G(z)
                discriminator_output, f_vector = D_detect(fake_feature)
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
                f.write('seed\tbeta\tdataset\tdev_eer\ttest_eer\tdata_size\n')
            line = '\t'.join([str(config['seed']), str(config['beta']), str(config['data_file']), str(best_dev), str(test_result['eer']), '100'])
            f.write(line + '\n')

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




if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, required=True)

    parser.add_argument('--data_file', type=str, required=True)

    parser.add_argument('--bert_type', type=str, required=True)


    parser.add_argument('--D_Wf_dim', type=int, default=512)

    parser.add_argument('--G_z_dim', type=int, default=512)

    parser.add_argument('--feature_dim', type=int, default=768)


    parser.add_argument('--do_train', action='store_true')

    parser.add_argument('--do_eval', action='store_true')

    parser.add_argument('--do_test', action='store_true')

    parser.add_argument('--do_vis', action='store_true')

    parser.add_argument('--output_dir', type=str, required=True)

    parser.add_argument('--n_epoch', type=int, default=50)

    parser.add_argument('--patience', default=10, type=int)

    parser.add_argument('--train_batch_size', default=32, type=int,
                        help='Batch size for training.')

    parser.add_argument('--predict_batch_size', default=32, type=int,
                        help='Batch size for evaluating and testing.')

    parser.add_argument('--D_detect_lr', type=float, default=1e-5, help="Learning rate for Discriminator.")
    parser.add_argument('--D_g_lr', type=float, default=1e-5, help="Learning rate for Discriminator.")
    parser.add_argument('--G_lr', type=float, default=1e-5, help="Learning rate for Generator.")
    parser.add_argument('--beta', type=float, default=0.1, help="Weight of fake sample loss for Discriminator.")

    parser.add_argument('--bert_lr', type=float, default=2e-4, help="Learning rate for Generator.")
    parser.add_argument('--fine_tune', action='store_true',
                        help='Whether to fine tune BERT during training.')
    parser.add_argument('--seed', type=int, default=123, help='seed')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    logger = Logger(os.path.join(args.output_dir, 'train.log'))
    main(args)