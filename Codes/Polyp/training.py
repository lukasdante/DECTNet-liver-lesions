# -*-ing:utf-8-*-
# -*-ing:utf-8-*-
import os
import random

import torch
import shutil
import logging
import numpy as np
import torch.nn as nn
import ml_collections
import torch.optim as optim
import torch.backends.cudnn as cudnn

from CaT_Net import CaT_Net_with_Decoder_DeepSup

import torch.utils.data as data
from Metrics import calculate_metric_other

from Polyp.PolypDataset import polyp_train_dataset, polyp_kvasir_dataset, polyp_cvc_300_dataset
from Polyp.PolypDataset import polyp_etis_larib_dataset, polyp_cvc_clinic_dataset,polyp_cvc_colon_dataset

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
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def Polyp_trainer_catnet(save_path, model, hyperparameter, train_dataset, test_dataset, logger_name):
    base_lr = hyperparameter.base_lr
    weight_decay = hyperparameter.weight_decay
    momentum = hyperparameter.momentum
    num_classes = hyperparameter.num_classes
    metrics_num = hyperparameter.metrics_num
    batch_size = hyperparameter.batch_size
    epochs = hyperparameter.epochs

    # loss_function
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(n_classes=num_classes)

    # dataloader
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=0,
                                   pin_memory=True,
                                   drop_last=True)


    test_dataloader_name_list = ["CVC_300", "CVC_Clinic", "CVC_Colon", "ETIS_Larib", "Kvasir"]

    test_loader = {"CVC_300": data.DataLoader(dataset=test_dataset["CVC_300"], batch_size=batch_size, shuffle=False),
                   "CVC_Clinic": data.DataLoader(dataset=test_dataset["CVC_Clinic"], batch_size=batch_size, shuffle=False),
                   "CVC_Colon": data.DataLoader(dataset=test_dataset["CVC_Colon"], batch_size=batch_size, shuffle=False),
                   "ETIS_Larib": data.DataLoader(dataset=test_dataset["ETIS_Larib"], batch_size=batch_size, shuffle=False),
                   "Kvasir": data.DataLoader(dataset=test_dataset["Kvasir"], batch_size=batch_size, shuffle=False)}


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

    # save logging
    # train logging
    logging.getLogger(name=logger_name["train"]['train_loss'])
    logging.getLogger(name=logger_name["train"]['train_loss']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "train_loss.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["train"]['train_loss']).addHandler(handle)

    logging.getLogger(name=logger_name["train"]['train_metrics'])
    logging.getLogger(name=logger_name["train"]['train_metrics']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "train_metrics.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["train"]['train_metrics']).addHandler(handle)

    # logging test dataset:CVC_300
    logging.getLogger(name=logger_name["test"]["CVC_300"]['test_loss'])
    logging.getLogger(name=logger_name["test"]["CVC_300"]['test_loss']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "CVC_300_loss.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["CVC_300"]['test_loss']).addHandler(handle)

    logging.getLogger(name=logger_name["test"]["CVC_300"]['test_metrics'])
    logging.getLogger(name=logger_name["test"]["CVC_300"]['test_metrics']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "CVC_300_metrics.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["CVC_300"]['test_metrics']).addHandler(handle)

    # logging test dataset:CVC_Clinic
    logging.getLogger(name=logger_name["test"]["CVC_Clinic"]["test_loss"])
    logging.getLogger(name=logger_name["test"]["CVC_Clinic"]['test_loss']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "CVC_Clinic_loss.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["CVC_Clinic"]['test_loss']).addHandler(handle)

    logging.getLogger(name=logger_name["test"]["CVC_Clinic"]['test_metrics'])
    logging.getLogger(name=logger_name["test"]["CVC_Clinic"]['test_metrics']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "CVC_Clinic_metrics.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["CVC_Clinic"]['test_metrics']).addHandler(handle)

    # logging test dataset:CVC_Colon
    logging.getLogger(name=logger_name["test"]["CVC_Colon"]['test_loss'])
    logging.getLogger(name=logger_name["test"]["CVC_Colon"]['test_loss']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "CVC_Colon_loss.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["CVC_Colon"]['test_loss']).addHandler(handle)

    logging.getLogger(name=logger_name["test"]["CVC_Colon"]['test_metrics'])
    logging.getLogger(name=logger_name["test"]["CVC_Colon"]['test_metrics']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "CVC_Colon_metrics.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["CVC_Colon"]['test_metrics']).addHandler(handle)

    # logging test dataset:ETIS_Larib
    logging.getLogger(name=logger_name["test"]["ETIS_Larib"]['test_loss'])
    logging.getLogger(name=logger_name["test"]["ETIS_Larib"]['test_loss']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "ETIS_Larib_loss.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["ETIS_Larib"]['test_loss']).addHandler(handle)

    logging.getLogger(name=logger_name["test"]["ETIS_Larib"]['test_metrics'])
    logging.getLogger(name=logger_name["test"]["ETIS_Larib"]['test_metrics']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "ETIS_Larib_metrics.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["ETIS_Larib"]['test_metrics']).addHandler(handle)

    # logging test dataset:Kvasir
    logging.getLogger(name=logger_name["test"]["Kvasir"]['test_loss'])
    logging.getLogger(name=logger_name["test"]["Kvasir"]['test_loss']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "Kvasir_loss.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["Kvasir"]['test_loss']).addHandler(handle)

    logging.getLogger(name=logger_name["test"]["Kvasir"]['test_metrics'])
    logging.getLogger(name=logger_name["test"]["Kvasir"]['test_metrics']).setLevel(logging.DEBUG)
    handle = logging.StreamHandler(open(os.path.join(save_path, "Kvasir_metrics.txt"), "a"))
    handle.setLevel(logging.DEBUG)
    logging.getLogger(name=logger_name["test"]["Kvasir"]['test_metrics']).addHandler(handle)

    BestPerformanceFile ={"CVC_300": open(os.path.join(save_path, r"CVC_300_BsetPerformance.txt"), "w"),
                          "CVC_Clinic": open(os.path.join(save_path, r"CVC_Clinic_BsetPerformance.txt"), "w"),
                          "CVC_Colon": open(os.path.join(save_path, r"CVC_Colon_BsetPerformance.txt"), "w"),
                          "ETIS_Larib": open(os.path.join(save_path, r"ETIS_Larib_BsetPerformance.txt"), "w"),
                          "Kvasir": open(os.path.join(save_path, r"Kvasir_BsetPerformance.txt"), "w")}

    best_performance ={"CVC_300": np.zeros((num_classes - 1, metrics_num)),
                       "CVC_Clinic": np.zeros((num_classes - 1, metrics_num)),
                       "CVC_Colon": np.zeros((num_classes - 1, metrics_num)),
                       "ETIS_Larib": np.zeros((num_classes - 1, metrics_num)),
                       "Kvasir": np.zeros((num_classes - 1, metrics_num))}
    iter_num = 0
    max_iterations = epochs * len(train_loader)
    print("***********dataset start training**************")
    # train process
    for epoch in range(epochs):
        with torch.enable_grad():
            net.train()
            losses = []
            final_losses = []
            train_metric = np.zeros((num_classes - 1, metrics_num))
            for _, sample in enumerate(train_loader):
                img, gt = sample["image"], sample["label"]
                img, gt = img.cuda(), gt.cuda()
                train_pred, decoder_train_pred, = net(img)
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
                losses.append(tr_loss.detach().cpu().numpy())
                optimizer.zero_grad()
                tr_loss.backward()
                optimizer.step()


                # calculate training data metrics
                _, train_pred_index = torch.max(train_pred, 1)  # 返回pred在dim=1的最大值的索引，最大值被引成_。
                train_pred_index = train_pred_index.detach().cpu().numpy()
                gt = gt.detach().cpu().numpy()
                for i in range(num_classes - 1):
                    train_metric[i] += np.array(calculate_metric_other(train_pred_index == i + 1, gt == i + 1))

            train_metric = train_metric / len(train_loader)
            logging.getLogger(name=logger_name["train"]['train_metrics']).info("dice:{}, jaccard:{}, accuracy:{}, "
                                                                 "sensitivity:{}, precision:{}, specificity:{}".
                                                                 format(train_metric[0][0],
                                                                        train_metric[0][1],
                                                                        train_metric[0][2],
                                                                        train_metric[0][3],
                                                                        train_metric[0][4],
                                                                        train_metric[0][5]))
            logging.getLogger(name=logger_name["train"]['train_loss']).info("total_loss:{}".format(str(sum(losses) / len(losses))))

            # adjust learning rate
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            iter_num = iter_num + 1

            print("The " + str(epoch + 1) + " epoch training process finished")

        with torch.no_grad():
            net.eval()
            for test_dataloader_name in test_dataloader_name_list:
                test_loss = []
                test_final_loss = []
                metric = np.zeros((num_classes-1, metrics_num))
                for _, sample in enumerate(test_loader[test_dataloader_name]):
                    img, gt = sample["image"], sample["label"]
                    img, gt = img.cuda(), gt.cuda()
                    te_pred, decoder_test_perd = net(img)
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
                        metric[i] += np.array(calculate_metric_other(t_pred == i + 1, gt == i + 1))

                metric = metric / len(test_loader[test_dataloader_name])
                for i in range(len(metric)):
                    for j in range(metrics_num):
                        if metric[i][j] > best_performance[test_dataloader_name][i][j]:
                            best_performance[test_dataloader_name][i][j] = metric[i][j]

                logging.getLogger(name=logger_name["test"][test_dataloader_name]['test_loss']).info(
                            "total_loss:{}".format(str(sum(test_loss) / len(test_loss))))

                logging.getLogger(name=logger_name["test"][test_dataloader_name]['test_metrics']).info("dice:{}, jaccard:{}, accuracy:{}, "
                                                                                 "sensitivity:{}, precision:{}, specificity:{}".
                                                                                 format(metric[0][0],
                                                                                        metric[0][1],
                                                                                        metric[0][2],
                                                                                        metric[0][3],
                                                                                        metric[0][4],
                                                                                        metric[0][5]))
                print(test_dataloader_name + ": The " + str(epoch + 1) + " epoch test process finished")

                # save model parameters
                # only save best Dice model

                ParaPath = os.path.join(save_path, net.name + r"_Parameters")
                if not os.path.isdir(ParaPath):
                    os.mkdir(ParaPath)
                # only save best dice model
                if (epoch + 1) > 1 and metric[0][0] >= best_performance[test_dataloader_name][0][0]:
                    # 把原来存储的删除，这样也能省很多空间
                    para_list = os.listdir(ParaPath)
                    for para_name in para_list:
                        if test_dataloader_name in para_name:
                            os.remove(os.path.join(ParaPath, para_name))
                    # for para_name in range(len(para_list)):
                    #     if test_dataloader_name in para_list[para_name]:
                    #         os.remove(os.path.join(ParaPath, para_list[para_name]))
                    Para_save_file = os.path.join(ParaPath, test_dataloader_name + r"_Dice_{:.5}".format(metric[0][0]) + r'.pt')
                    torch.save(net.state_dict(), Para_save_file)

    for test_dataloader_name in test_dataloader_name_list:
        BestPerformanceFile[test_dataloader_name].write(str(best_performance[test_dataloader_name]))
        BestPerformanceFile[test_dataloader_name].close()
    print("*****************Dataset Training Finished**************************")



if __name__ == "__main__":
    root_dir = r"/home/lbl/Experiments/2022_9_11"
    if not os.path.isdir(root_dir):
        os.mkdir(root_dir)

    cudnn.benchmark = False
    cudnn.deterministic = True

    seed = 1500
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    polyp_test_dataset = {"CVC_300": polyp_cvc_300_dataset,
                          "CVC_Clinic": polyp_cvc_clinic_dataset,
                          "CVC_Colon": polyp_cvc_colon_dataset,
                          "ETIS_Larib": polyp_etis_larib_dataset,
                          "Kvasir": polyp_kvasir_dataset}

    polyp_hyperparameters = ml_collections.ConfigDict()
    polyp_hyperparameters.base_lr = 0.04
    polyp_hyperparameters.weight_decay = 0.0001
    polyp_hyperparameters.momentum = 0.9
    polyp_hyperparameters.num_classes = 2
    polyp_hyperparameters.metrics_num = 6
    polyp_hyperparameters.batch_size = 16
    polyp_hyperparameters.epochs = 100

    polyp_root_dir = os.path.join(root_dir, "Polyp_lr4")
    if not os.path.isdir(polyp_root_dir):
        os.mkdir(polyp_root_dir)

    print("-" * 30 + "CAT-Net-polyp" + "-" * 30)
    model_polyp_catnet = CaT_Net_with_Decoder_DeepSup(num_classes=2,
                                                      cnn_channels=32,
                                                      swin_trans_channels=24,
                                                      num_layers=[4, 4, 4, 4, 4])

    CATNet_polyp_train_logger_name = {}
    CATNet_polyp_test_logger_name = {}
    CATNet_polyp_logger_name = {"train": CATNet_polyp_train_logger_name,
                               "test": CATNet_polyp_test_logger_name}

    CATNet_polyp_train_logger_name["train_loss"] = "CATNet_polyp_train_loss"
    CATNet_polyp_train_logger_name["train_metrics"] = "CATNet_polyp_train_metrics"

    CATNet_polyp_cvc_300_logger_name = {"test_loss": "CATNet_polyp_CVC_300_loss",
                                      "test_metrics": "CATNet_polyp_CVC_300_metrics"}
    CATNet_polyp_cvc_clinic_logger_name = {"test_loss": "CATNet_polyp_CVC_Clinic_loss",
                                         "test_metrics": "CATNet_polyp_CVC_Clinic_metrics"}
    CATNet_polyp_cvc_colon_logger_name = {"test_loss": "CATNet_polyp_CVC_Colon_loss",
                                        "test_metrics": "CATNet_polyp_CVC_Colon_metrics"}
    CATNet_polyp_etis_larib_logger_name = {"test_loss": "CATNet_polyp_ETIS_Larib_loss",
                                         "test_metrics": "CATNet_polyp_ETIS_Larib_metrics"}
    CATNet_polyp_kvasir_logger_name = {"test_loss": "CATNet_polyp_Kvasir_loss",
                                     "test_metrics": "CATNet_polyp_Kvasir_metrics"}

    CATNet_polyp_test_logger_name["CVC_300"] = CATNet_polyp_cvc_300_logger_name
    CATNet_polyp_test_logger_name["CVC_Clinic"] = CATNet_polyp_cvc_clinic_logger_name
    CATNet_polyp_test_logger_name["CVC_Colon"] = CATNet_polyp_cvc_colon_logger_name
    CATNet_polyp_test_logger_name["ETIS_Larib"] = CATNet_polyp_etis_larib_logger_name
    CATNet_polyp_test_logger_name["Kvasir"] = CATNet_polyp_kvasir_logger_name

    Polyp_trainer_catnet(polyp_root_dir, model_polyp_catnet, polyp_hyperparameters, polyp_train_dataset, polyp_test_dataset, CATNet_polyp_logger_name)






