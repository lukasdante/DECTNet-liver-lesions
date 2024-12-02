# -*-ing:utf-8-*-
import os
import random

import torch
import shutil
import logging
import numpy as np
import torch.nn as nn
import ml_collections

from CaTNet import CaT_Net_with_Decoder_DeepSup
from Dataset import mms_train_dataset, mms_test_dataset

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from Metrics import calculate_metric_cardiac

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # temp_prob是True or False
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), \
			'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def del_file(filepath):
    """
	delete everything in the specified directory.
	:param filepath: specified directory
	:return: None
	"""
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def multi_classify_decoder_deep_supervision_trainer(save_path, model, hyperparameter, train_dataset, test_dataset, logger_name):
    base_lr = hyperparameter.base_lr
    weight_decay = hyperparameter.weight_decay
    momentum = hyperparameter.momentum
    num_classes = hyperparameter.num_classes
    metrics_num = hyperparameter.metrics_num
    batch_size = hyperparameter.batch_size
    epochs = hyperparameter.epochs

    # loss function
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)

    # data_loader
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=2,
                                   pin_memory=True,
                                   drop_last=True)

    test_loader = data.DataLoader(dataset=test_dataset,
                                  batch_size=batch_size,
                                  num_workers=2,
                                  pin_memory=True,
                                  shuffle=False)

    # network
    net = model.cuda()

    # optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=base_lr,
                          momentum=momentum,
                          weight_decay=weight_decay)

    # save_file_path
    save_path = os.path.join(save_path, net.name)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    else:
        del_file(save_path)

    # save_logging
    logging.getLogger(name=logger_name["train_MYO_metrics"])
    logging.getLogger(name=logger_name["train_MYO_metrics"]).setLevel(logging.DEBUG)
    train_MYO_metrics_handle = logging.StreamHandler(open(os.path.join(save_path, "train_MYO_metric.txt"), "a"))
    train_MYO_metrics_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["train_MYO_metrics"]).addHandler(train_MYO_metrics_handle)

    logging.getLogger(name=logger_name["train_RV_metrics"])
    logging.getLogger(name=logger_name["train_RV_metrics"]).setLevel(logging.DEBUG)
    train_RV_metrics_handle = logging.StreamHandler(open(os.path.join(save_path, "train_RV_metric.txt"), "a"))
    train_RV_metrics_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["train_RV_metrics"]).addHandler(train_RV_metrics_handle)

    logging.getLogger(name=logger_name["train_LV_metrics"])
    logging.getLogger(name=logger_name["train_LV_metrics"]).setLevel(logging.DEBUG)
    train_LV_metrics_handle = logging.StreamHandler(open(os.path.join(save_path, "train_LV_metric.txt"), "a"))
    train_LV_metrics_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["train_LV_metrics"]).addHandler(train_LV_metrics_handle)

    # save_logging
    logging.getLogger(name=logger_name["test_MYO_metrics"])
    logging.getLogger(name=logger_name["test_MYO_metrics"]).setLevel(logging.DEBUG)
    test_MYO_metrics_handle = logging.StreamHandler(open(os.path.join(save_path, "test_MYO_metric.txt"), "a"))
    test_MYO_metrics_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test_MYO_metrics"]).addHandler(test_MYO_metrics_handle)

    logging.getLogger(name=logger_name["test_RV_metrics"])
    logging.getLogger(name=logger_name["test_RV_metrics"]).setLevel(logging.DEBUG)
    test_RV_metrics_handle = logging.StreamHandler(open(os.path.join(save_path, "test_RV_metric.txt"), "a"))
    test_RV_metrics_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test_RV_metrics"]).addHandler(test_RV_metrics_handle)

    logging.getLogger(name=logger_name["test_LV_metrics"])
    logging.getLogger(name=logger_name["test_LV_metrics"]).setLevel(logging.DEBUG)
    test_LV_metrics_handle = logging.StreamHandler(open(os.path.join(save_path, "test_LV_metric.txt"), "a"))
    test_LV_metrics_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test_LV_metrics"]).addHandler(test_LV_metrics_handle)

    logging.getLogger(name=logger_name["train_loss"])
    logging.getLogger(name=logger_name["train_loss"]).setLevel(logging.DEBUG)
    train_loss_handle = logging.StreamHandler(open(os.path.join(save_path, "train_loss.txt"), "a"))
    train_loss_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["train_loss"]).addHandler(train_loss_handle)

    logging.getLogger(name=logger_name["test_loss"])
    logging.getLogger(name=logger_name["test_loss"]).setLevel(logging.DEBUG)
    test_loss_handle = logging.StreamHandler(open(os.path.join(save_path, "test_loss.txt"), "a"))
    test_loss_handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test_loss"]).addHandler(test_loss_handle)

    BestPerformanceFile = open(os.path.join(save_path, r"BsetPerformance.txt"), "w")
    best_performance = np.zeros((num_classes - 1, metrics_num))
    iter_num = 0
    max_iterations = epochs * len(train_loader)
    print("***********Dataset start training**************")
    for epoch in range(epochs):
        net.train()
        losses = []
        final_losses = []
        train_metric = np.zeros((num_classes - 1, metrics_num))
        for _, sample in enumerate(train_loader):
            img, gt = sample["image"], sample["label"]
            img, gt = img.cuda(), gt.cuda()
            train_pred, decoder_train_pred = net(img)
            gt = gt.squeeze(dim=1).long()
            tr_final_loss_ce = ce_loss(train_pred, gt)
            tr_final_loss_dice = dice_loss(train_pred, gt, softmax=True)
            tr_final_loss = 0.5 * tr_final_loss_ce + 0.5 * tr_final_loss_dice
            tr_decoder_loss_ce_1 = ce_loss(decoder_train_pred[0], gt)
            tr_decoder_loss_ce_2 = ce_loss(decoder_train_pred[1], gt)
            tr_decoder_loss_ce_3 = ce_loss(decoder_train_pred[2], gt)


            tr_decoder_loss_dice_1 = dice_loss(decoder_train_pred[0], gt, softmax=True)
            tr_decoder_loss_dice_2 = dice_loss(decoder_train_pred[1], gt, softmax=True)
            tr_decoder_loss_dice_3 = dice_loss(decoder_train_pred[2], gt, softmax=True)

            tr_decoder_loss_1 = 0.5 * tr_decoder_loss_ce_1 + 0.5 * tr_decoder_loss_dice_1
            tr_decoder_loss_2 = 0.5 * tr_decoder_loss_ce_2 + 0.5 * tr_decoder_loss_dice_2
            tr_decoder_loss_3 = 0.5 * tr_decoder_loss_ce_3 + 0.5 * tr_decoder_loss_dice_3

            tr_deep_supervised_loss = tr_decoder_loss_1 + tr_decoder_loss_2 + tr_decoder_loss_3

            tr_loss = 0.9 * tr_final_loss + 0.1 * (tr_deep_supervised_loss)

            final_losses.append(tr_final_loss.detach().cpu().numpy())
            losses.append(tr_loss.detach().cpu().numpy())
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

            _, pred = torch.max(train_pred, 1)  # 返回pred在dim=1的最大值的索引，最大值被引成_。
            pred = pred.detach().cpu().numpy()
            gt = gt.detach().cpu().numpy()

            for i in range(num_classes - 1):
                train_metric[i] += np.array(calculate_metric_cardiac(pred == i + 1, gt == i + 1))
        train_metric = train_metric / len(train_loader)

        logging.getLogger(name=logger_name["train_RV_metrics"]).info(
            "dice:{}, jaccard:{}, accuracy:{}, hd95:{}, asd:{}".
                format(train_metric[0][0],
                       train_metric[0][1],
                       train_metric[0][2],
                       train_metric[0][3],
                       train_metric[0][4]))
        logging.getLogger(name=logger_name["train_MYO_metrics"]).info(
            "dice:{}, jaccard:{}, accuracy:{}, hd95:{}, asd:{}".
                format(train_metric[1][0],
                       train_metric[1][1],
                       train_metric[1][2],
                       train_metric[1][3],
                       train_metric[1][4]))
        logging.getLogger(name=logger_name["train_LV_metrics"]).info(
            "dice:{}, jaccard:{}, accuracy:{}, hd95:{}, asd:{}".
                format(train_metric[2][0],
                       train_metric[2][1],
                       train_metric[2][2],
                       train_metric[2][3],
                       train_metric[2][4]))

        logging.getLogger(name=logger_name["train_loss"]).info("final_loss:{}, total_loss:{}".
                                                               format(str(sum(final_losses) / len(final_losses)),
                                                                      str(sum(losses) / len(losses))))
        print("第" + str(epoch + 1) + "轮训练集平均损失为：", str(sum(losses) / len(losses)))
        print("RV Dice cofficient:", str(train_metric[0][0]))
        print("MYO Dice cofficient:", str(train_metric[1][0]))
        print("LV Dice cofficient:", str(train_metric[2][0]))

        with torch.no_grad():
            net.eval()
            test_loss = []
            test_final_loss = []
            metric = np.zeros((num_classes - 1, metrics_num))
            for _, sample in enumerate(test_loader):
                img, gt = sample["image"], sample["label"]
                img, gt = img.cuda(), gt.cuda()
                te_pred, decoder_test_perd, = net(img)
                gt = gt.squeeze(dim=1).long()
                te_final_loss_ce = ce_loss(te_pred, gt)
                te_final_loss_dice = dice_loss(te_pred, gt, softmax=True)
                te_final_loss = 0.5 * te_final_loss_ce + 0.5 * te_final_loss_dice

                te_decoder_loss_ce_1 = ce_loss(decoder_test_perd[0], gt)
                te_decoder_loss_ce_2 = ce_loss(decoder_test_perd[1], gt)
                te_decoder_loss_ce_3 = ce_loss(decoder_test_perd[2], gt)


                te_decoder_loss_dice_1 = dice_loss(decoder_test_perd[0], gt, softmax=True)
                te_decoder_loss_dice_2 = dice_loss(decoder_test_perd[1], gt, softmax=True)
                te_decoder_loss_dice_3 = dice_loss(decoder_test_perd[2], gt, softmax=True)

                te_decoder_loss_1 = 0.5 * te_decoder_loss_ce_1 + 0.5 * te_decoder_loss_dice_1
                te_decoder_loss_2 = 0.5 * te_decoder_loss_ce_2 + 0.5 * te_decoder_loss_dice_2
                te_decoder_loss_3 = 0.5 * te_decoder_loss_ce_3 + 0.5 * te_decoder_loss_dice_3


                te_deep_supervised_loss = te_decoder_loss_1 + te_decoder_loss_2 + te_decoder_loss_3

                te_loss = 0.9 * te_final_loss + 0.1 * (te_deep_supervised_loss)

                test_final_loss.append(te_final_loss.detach().cpu().numpy())
                test_loss.append(te_loss.detach().cpu().numpy())

                _, t_pred = torch.max(te_pred, 1)  # 返回pred在dim=1的最大值的索引，最大值被引成_。
                t_pred = t_pred.detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()

                for i in range(num_classes - 1):
                    metric[i] += np.array(calculate_metric_cardiac(t_pred == i + 1, gt == i + 1))

            metric = metric / len(test_loader)
            for i in range(len(metric)):
                for j in range(metrics_num):
                    if metric[i][j] > best_performance[i][j]:
                        best_performance[i][j] = metric[i][j]
            logging.getLogger(name=logger_name["test_loss"]).info("final_loss:{}, total_loss:{}".
                                                                  format(
                str(sum(test_final_loss) / len(test_final_loss)),
                str(sum(test_loss) / len(test_loss))))

            logging.getLogger(name=logger_name["test_RV_metrics"]).info(
                "dice:{}, jaccard:{}, accuracy:{}, hd95:{}, asd:{}".
                    format(metric[0][0],
                           metric[0][1],
                           metric[0][2],
                           metric[0][3],
                           metric[0][4]))
            logging.getLogger(name=logger_name["test_MYO_metrics"]).info(
                "dice:{}, jaccard:{}, accuracy:{}, hd95:{}, asd:{}".
                    format(metric[1][0],
                           metric[1][1],
                           metric[1][2],
                           metric[1][3],
                           metric[1][4]))
            logging.getLogger(name=logger_name["test_LV_metrics"]).info(
                "dice:{}, jaccard:{}, accuracy:{}, hd95:{}, asd:{}".
                    format(metric[2][0],
                           metric[2][1],
                           metric[2][2],
                           metric[2][3],
                           metric[2][4]))
            print("第" + str(epoch + 1) + "轮测试集平均损失为：", str(sum(test_loss) / len(test_loss)))
            print("RV Dice cofficient:", str(metric[0][0]))
            print("MYO Dice cofficient:", str(metric[1][0]))
            print("LV Dice cofficient:", str(metric[2][0]))
            # save model parameters
            ParaPath = os.path.join(save_path, net.name + r"_Parameters")
            if not os.path.isdir(ParaPath):
                os.mkdir(ParaPath)
            # only save best dice model
            if (epoch + 1) > 5 and metric[0][0] >= best_performance[0][0]:
                # 把原来存储的删除，这样也能省很多空间
                if os.listdir(ParaPath) != []:
                    os.remove(os.path.join(ParaPath, os.listdir(ParaPath)[0]))
                Para_save_file = os.path.join(ParaPath, r"Dice_{:.5}".format(metric[0][0]) + r'.pt')
                torch.save(net.state_dict(), Para_save_file)

    BestPerformanceFile.write(str(best_performance))
    BestPerformanceFile.close()
    print("*****************Dataset Training Finished**************************")

if __name__ == "__main__":
    # avoid randomness
    cudnn.benchmark = False
    cudnn.deterministic = True

    seed = 1500
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    multi_classify_hyperparameters = ml_collections.ConfigDict()
    multi_classify_hyperparameters.base_lr = 0.03
    multi_classify_hyperparameters.momentum = 0.9
    multi_classify_hyperparameters.weight_decay = 0.0001
    multi_classify_hyperparameters.num_classes = 4
    multi_classify_hyperparameters.metrics_num = 5
    multi_classify_hyperparameters.batch_size = 2
    multi_classify_hyperparameters.epochs = 100

    root_path = r"Weights/"
    if not os.path.isdir(root_path):
        os.mkdir(root_path)


    save_path_mms = os.path.join(root_path, "mms_final")
    if not os.path.isdir(save_path_mms):
        os.mkdir(save_path_mms)

    print("CaTNet-mms")
    model_mms_CaTNet = CaT_Net_with_Decoder_DeepSup(num_classes=4,
                                                    cnn_channels=32,
                                                    swin_trans_channels=24,
                                                    num_layers=[4, 4, 4, 4, 4])

    CaTNet_mms_logger_name = {}
    CaTNet_mms_logger_name["train_loss"] = "CaTNet_mms_train_loss"
    CaTNet_mms_logger_name["test_loss"] = "CaTNet_mms_test_loss"
    CaTNet_mms_logger_name["train_RV_metrics"] = "CaTNet_mms_train_RV_metrics"
    CaTNet_mms_logger_name["train_LV_metrics"] = "CaTNet_mms_train_LV_metrics"
    CaTNet_mms_logger_name["train_MYO_metrics"] = "CaTNet_mms_train_MYO_metrics"
    CaTNet_mms_logger_name["test_RV_metrics"] = "CaTNet_mms_test_RV_metrics"
    CaTNet_mms_logger_name["test_LV_metrics"] = "CaTNet_mms_test_LV_metrics"
    CaTNet_mms_logger_name["test_MYO_metrics"] = "CaTNetmms_test_MYO_metrics"
    multi_classify_decoder_deep_supervision_trainer(save_path=save_path_mms,
                           model=model_mms_CaTNet,
                           hyperparameter=multi_classify_hyperparameters,
                           train_dataset=mms_train_dataset,
                           test_dataset=mms_test_dataset,
                           logger_name=CaTNet_mms_logger_name)