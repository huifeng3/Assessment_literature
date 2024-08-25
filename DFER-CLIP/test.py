import argparse
import os
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from models.Generate_Model import GenerateModel
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import itertools
import datetime
from dataloader.video_dataloader import train_data_loader, test_data_loader
from sklearn.metrics import confusion_matrix
import tqdm
from models.clip import clip
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models.Text import *
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="DFEW")

parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=48)

parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr-image-encoder', type=float, default=1e-5)
parser.add_argument('--lr-prompt-learner', type=float, default=1e-3)

parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int, default=30)

parser.add_argument('--contexts-number', type=int, default=8)
parser.add_argument('--class-token-position', type=str, default="end")
parser.add_argument('--class-specific-contexts', type=str, default='True')
parser.add_argument('--load-and-tune-prompt-learner', type=str, default='False')

parser.add_argument('--text-type', type=str, default="class_descriptor")
parser.add_argument('--exper-name', type=str, default="test")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--temporal-layers', type=int, default=1)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

now = datetime.datetime.now()
time_str = now.strftime("%y%m%d%H%M")
time_str = time_str + "test"

print('************************')
for k, v in vars(args).items():
    print(k, '=', v)
print('************************')

if args.dataset == "FERV39K" or args.dataset == "DFEW":
    number_class = 7
    class_names = class_names_7
    class_names_with_context = class_names_with_context_7
    class_descriptor = class_descriptor_7

elif args.dataset == "MAFW":
    number_class = 11
    class_names = class_names_11
    class_names_with_context = class_names_with_context_11
    class_descriptor = class_descriptor_11


def main(set):
    data_set = set + 1

    if args.dataset == "FERV39K":
        print("*********** FERV39K Dataset ***********")
        log_txt_path = './log/' + 'FER39K-' + time_str + '-log.txt'
        log_curve_path = './log/' + 'FER39K-' + time_str + '-log.png'
        log_confusion_matrix_path = './log/' + 'FER39K-' + time_str + '-cn.png'
        checkpoint_path = '/checkpoint/' + 'FER39K-' + time_str + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'FER39K-' + time_str + '-model_best.pth'
        train_annotation_file_path = "./annotation/FERV39K_train.txt"
        test_annotation_file_path = "./annotation/FERV39K_test.txt"

    elif args.dataset == "DFEW":
        print("*********** DFEW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log.txt'
        log_curve_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-log.png'
        log_confusion_matrix_path = './log/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-cn.png'
        checkpoint_path = './checkpoint/' + 'DFEW-' + time_str + '-set' + str(data_set) + '-model.pth'
        best_checkpoint_path = 'DFEW-set1-model.pth'
        train_annotation_file_path = "./annotation/DFEW_set_" + str(data_set) + "_train.txt"
        test_annotation_file_path = "./annotation/DFEW_set_" + str(data_set) + "_test.txt"

    elif args.dataset == "MAFW":
        print("*********** MAFW Dataset Fold  " + str(data_set) + " ***********")
        log_txt_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log.txt'
        log_curve_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-log.png'
        log_confusion_matrix_path = './log/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-cn.png'
        checkpoint_path = './checkpoint/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-model.pth'
        best_checkpoint_path = './checkpoint/' + 'MAFW-' + time_str + '-set' + str(data_set) + '-model_best.pth'
        train_annotation_file_path = "./annotation/MAFW_set_" + str(data_set) + "_train.txt"
        test_annotation_file_path = "./annotation/MAFW_set_" + str(data_set) + "_test.txt"

    CLIP_model, _ = clip.load("ViT-B/32", device='cpu')

    if args.text_type == "class_names":
        input_text = class_names
    elif args.text_type == "class_names_with_context":
        input_text = class_names_with_context
    elif args.text_type == "class_descriptor":
        input_text = class_descriptor

    cudnn.benchmark = True

    model = GenerateModel(input_text=input_text, clip_model=CLIP_model, args=args)
    state_dict = torch.load("DFEW-set1-model.pth", map_location='cpu')
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    # Data loading code
    test_data = test_data_loader(list_file=test_annotation_file_path,
                                 num_segments=16,
                                 duration=1,
                                 image_size=224)

    val_loader = torch.utils.data.DataLoader(test_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    uar, war = computer_uar_war(val_loader, model, log_confusion_matrix_path, log_txt_path,
                                data_set)

    return uar, war


def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=12,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=18)
    plt.xlabel('Predicted label', fontsize=18)
    plt.tight_layout()

def computer_uar_war(val_loader, model, log_confusion_matrix_path, log_txt_path, data_set):
    # pre_trained_dict = torch.load(best_checkpoint_path)['state_dict']
    # model.load_state_dict(pre_trained_dict)

    # state_dict = torch.load("DFEW-set1-model.pth", map_location='cpu')
    # new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # model.load_state_dict(new_state_dict)
    #
    # model.eval()

    correct = 0
    with torch.no_grad():
        for i, (images, target) in enumerate(tqdm.tqdm(val_loader)):

            images = images.cuda()
            target = target.cuda()

            output = model(images)

            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.view_as(predicted)).sum().item()

            if i == 0:
                all_predicted = predicted
                all_targets = target
            else:
                all_predicted = torch.cat((all_predicted, predicted), 0)
                all_targets = torch.cat((all_targets, target), 0)

    war = 100. * correct / len(val_loader.dataset)

    # Compute confusion matrix
    _confusion_matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
    np.set_printoptions(precision=4)
    normalized_cm = _confusion_matrix.astype('float') / _confusion_matrix.sum(axis=1)[:, np.newaxis]
    normalized_cm = normalized_cm * 100
    list_diag = np.diag(normalized_cm)
    uar = list_diag.mean()

    print("Confusion Matrix Diag:", list_diag)
    print("UAR: %0.2f" % uar)
    print("WAR: %0.2f" % war)

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))

    if args.dataset == "FERV39K":
        title_ = "Confusion Matrix on FERV39k"
    elif args.dataset == "DFEW":
        title_ = "Confusion Matrix on DFEW fold " + str(data_set)
    elif args.dataset == "MAFW":
        title_ = "Confusion Matrix on MAFW fold " + str(data_set)

    plot_confusion_matrix(normalized_cm, classes=class_names, normalize=True, title=title_)
    plt.savefig(os.path.join(log_confusion_matrix_path))
    plt.close()

    with open(log_txt_path, 'a') as f:
        f.write('************************' + '\n')
        f.write("Confusion Matrix Diag:" + '\n')
        f.write(str(list_diag.tolist()) + '\n')
        f.write('UAR: {:.2f}'.format(uar) + '\n')
        f.write('WAR: {:.2f}'.format(war) + '\n')
        f.write('************************' + '\n')

    return uar, war


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            # print('Curve was saved')
        plt.close(fig)


if __name__ == '__main__':
    main(0)
