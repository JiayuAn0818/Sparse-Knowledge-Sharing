import argparse
import numpy as np
import random
import os
import sys
import copy
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from utils import get_data
from Conformer import Conformer, channel_selection, MultiHeadAttention, FeedForwardBlock, PatchEmbedding
from logger import Logger

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from method.moe import SupConLoss

from scipy.spatial.distance import cdist


def evaluate_validation(model, loader, args, prototype):
    model.eval()
    pred_y = []
    true_y = []
    for x, y in loader:
        x = x.cuda()
        y = y.cuda()
        with torch.no_grad():
            _, fea = model.forward(x)
            fea = F.normalize(fea, dim=-1)
            distances = cosine_distance_matrix(fea, prototype.cuda())
            pred_y.append(torch.argmin(distances, dim=1)%2)
            true_y.append(y)
    pred_y = torch.cat(pred_y)
    true_y = torch.cat(true_y)
    acc = accuracy_score(true_y.cpu(), pred_y.cpu())
    bca = balanced_accuracy_score(true_y.cpu(), pred_y.cpu())
    f1 = f1_score(true_y.cpu(), pred_y.cpu(), average='weighted')
    return acc, bca, f1


def cal_metrics(acc_matrix, task_id):
    ## calculate evaluate metrics: ACC, BWT
    acc = acc_matrix[task_id].sum()/(task_id+1)
    bwt = 0
    for i in range(task_id):
        bwt += acc_matrix[task_id, i] - acc_matrix[i, i]
    bwt /= (task_id+1)
    return acc, bwt

def sparse_selection(model):
    s1 = args.s1
    s2 = args.s2
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            m.select1.indexes.grad.data.add_(s1*torch.sign(m.select1.indexes.data))  # L1
        elif isinstance(m, FeedForwardBlock):
            m.select2.indexes.grad.data.add_(s2*torch.sign(m.select2.indexes.data))  # L1

class Bn_Controller:
    def __init__(self):
        """
        freeze_bn and unfreeze_bn must appear in pairs
        """
        self.backup = {}

    def freeze_bn(self, model):
        assert self.backup == {}
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                self.backup[name + '.running_mean'] = m.running_mean.data.clone()
                self.backup[name + '.running_var'] = m.running_var.data.clone()
                self.backup[name + '.num_batches_tracked'] = m.num_batches_tracked.data.clone()
            # elif isinstance(m, nn.LayerNorm):
            #     self.backup[name + '.weight'] = m.weight.data.clone()
            #     self.backup[name + '.bias'] = m.bias.data.clone()

    def unfreeze_bn(self, model):
        for name, m in model.named_modules():
            if isinstance(m, nn.SyncBatchNorm) or isinstance(m, nn.BatchNorm2d):
                m.running_mean.data = self.backup[name + '.running_mean']
                m.running_var.data = self.backup[name + '.running_var']
                m.num_batches_tracked.data = self.backup[name + '.num_batches_tracked']
            # elif isinstance(m, nn.LayerNorm):
            #     m.weight.data = self.backup[name + '.weight'] 
            #     m.bias.data = self.backup[name + '.bias'] 
        self.backup = {}

def constrain_gradient(model):
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            mask = m.select1.masks.data.float()
            ## pre
            m.queries.weight.grad.data *= (1-mask).view(-1, 1).expand(400, 400).cuda()
            m.keys.weight.grad.data *= (1-mask).view(-1, 1).expand(400, 400).cuda()
            m.values.weight.grad.data *= (1-mask).view(-1, 1).expand(400, 400).cuda()
            m.queries.bias.grad.data *= (1-mask).view(-1).cuda()
            m.keys.bias.grad.data *= (1-mask).view(-1).cuda()
            m.values.bias.grad.data *= (1-mask).view(-1).cuda()
            ## post
            m.projection.weight.grad.data *= (1-mask).view(1, -1).expand(400, 400).cuda()
        if isinstance(m, FeedForwardBlock):
            mask = m.select2.masks.data.float()
            ## pre
            m.net1[0].weight.grad.data *= (1-mask).view(-1, 1).expand_as(m.net1[0].weight).cuda()
            m.net1[0].bias.grad.data *= (1-mask).view(-1).cuda()
            ## post
            m.net2.weight.grad.data *= (1-mask).view(1, -1).expand_as(m.net2.weight).cuda()

def train(model, train_loader, val_loader, args, task_id):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch*len(train_loader), eta_min=1e-7)
    loss_func = nn.CrossEntropyLoss().cuda()

    ## training parameters
    for e in range(args.epoch):
        model.train()

        fea_all = []
        true_y = []
        for _, (b_x, b_y) in enumerate(train_loader):
            b_x, b_y = b_x.cuda(), b_y.cuda()
            
            output, fea = model.forward(b_x)
            fea_norm = F.normalize(fea, dim=-1)
            fea_all.append(fea_norm.detach().cpu().numpy()) 
            true_y.append(b_y.detach().cpu().numpy())
            loss = loss_func(output, b_y.long())

            loss += SupConLoss()(fea_norm.unsqueeze(1), labels=b_y) ## 一致性损失
            
            optimizer.zero_grad()
            loss.backward()       

            sparse_selection(model)   
            if task_id>0:
                constrain_gradient(model)

            optimizer.step()
            scheduler.step()

        fea_all = np.concatenate(fea_all)
        true_y = np.concatenate(true_y)
        index0 = np.where(true_y==0)
        fea0 = fea_all[index0].mean(axis=0)
        index1 = np.where(true_y==1)
        fea1 = fea_all[index1].mean(axis=0)
        prototype = torch.cat((torch.tensor(fea0).view(1, -1), torch.tensor(fea1).view(1, -1))).cuda()

             
        if (e+1)%10 == 0:
            val_acc, val_bca, val_f1 = evaluate_validation(model, val_loader, args, prototype)
            print('Epoch[{}/{}], val acc = {:.4f}, val bca = {:.4f}, val f1 Score = {:.4f}.'.format(
                e, args.epoch, val_acc, val_bca, val_f1))              

        # print(a)
        # print(b)

    torch.save(model.state_dict(), args.OutputPathModels+'/task{}_orig.npy'.format(task_id+1))

def prune(model, task_id):
    cur_mask = []
    cur_index = []
    for m in model.modules():
        if isinstance(m, channel_selection):
            weight_copy = m.indexes.data
            _, topk_indices = torch.topk(weight_copy, len(weight_copy)//10)
            mask_for_cur = torch.zeros_like(weight_copy).cuda()
            mask_for_cur[topk_indices] = 1.0
            m.indexes.data.mul_(mask_for_cur)
            cur_index.append(m.indexes.data.cpu())

            if task_id>0:
                mask_for_old = m.masks.data.bool()
                mask_for_cur = mask_for_cur.bool()
                mask_for_cur = (~mask_for_old) & mask_for_cur
                mask_all = mask_for_cur | mask_for_old
                m.masks.data = mask_all.float()
            else:
                mask_all = mask_for_cur
                m.masks.data = mask_all
            cur_mask.append(mask_for_cur.cpu())
            print(torch.sum(mask_for_cur), torch.sum(mask_all))
    torch.save(cur_mask, args.OutputPathModels+'/mask{}.npy'.format(task_id+1))
    torch.save(cur_index, args.OutputPathModels+'/index{}.npy'.format(task_id+1))

def constrain_gradient_retrain(model, cur_mask):
    k=0
    for m in model.modules():
        if isinstance(m, MultiHeadAttention):
            ## pre
            m.queries.weight.grad.data *= cur_mask[k].view(-1, 1).expand(400, 400).cuda()
            m.keys.weight.grad.data *= cur_mask[k].view(-1, 1).expand(400, 400).cuda()
            m.values.weight.grad.data *= cur_mask[k].view(-1, 1).expand(400, 400).cuda()
            m.queries.bias.grad.data *= cur_mask[k].view(-1).cuda()
            m.keys.bias.grad.data *= cur_mask[k].view(-1).cuda()
            m.values.bias.grad.data *= cur_mask[k].view(-1).cuda()
            ## post
            m.projection.weight.grad.data *= cur_mask[k].view(1, -1).expand(400, 400).cuda()
            k += 1
        if isinstance(m, FeedForwardBlock):
            ## pre
            m.net1[0].weight.grad.data *= cur_mask[k].view(-1, 1).expand_as(m.net1[0].weight).cuda()
            m.net1[0].bias.grad.data *= cur_mask[k].view(-1).cuda()
            ## post
            m.net2.weight.grad.data *= cur_mask[k].view(1, -1).expand_as(m.net2.weight).cuda()
            k += 1

def retrain(model, train_loader, val_loader, args, task_id):
    ## 对模型剪枝
    prune(model, task_id)
    cur_mask = torch.load(args.OutputPathModels+'/mask{}.npy'.format(task_id+1))
    ## 对mask冻结
    for m in model.modules():
        if isinstance(m, channel_selection):
            for p in m.parameters():
                p.requires_grad = False

    bn_con = Bn_Controller() ## 控制bn不发生改变

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/5)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
    loss_func = nn.CrossEntropyLoss().cuda()

    ## training parameters
    for e in range(10):
        model.train()

        fea_all = []
        true_y = []
        for _, (b_x, b_y) in enumerate(train_loader):
            bn_con.freeze_bn(model)
            b_x, b_y = b_x.cuda(), b_y.cuda()
            
            output, fea = model.forward(b_x)
            fea_norm = F.normalize(fea, dim=-1)
            fea_all.append(fea_norm.detach().cpu().numpy()) 
            true_y.append(b_y.detach().cpu().numpy())
            loss = loss_func(output, b_y.long())
            loss += SupConLoss()(fea_norm.unsqueeze(1), labels=b_y) ## 一致性损失
            
            optimizer.zero_grad()
            loss.backward()       

            constrain_gradient_retrain(model, cur_mask)     

            optimizer.step()
            bn_con.unfreeze_bn(model)

        scheduler.step()

        fea_all = np.concatenate(fea_all)
        true_y = np.concatenate(true_y)
        index0 = np.where(true_y==0)
        fea0 = fea_all[index0].mean(axis=0)
        index1 = np.where(true_y==1)
        fea1 = fea_all[index1].mean(axis=0)
        prototype = torch.cat((torch.tensor(fea0).view(1, -1), torch.tensor(fea1).view(1, -1))).cuda()
            
        if (e+1)%5 == 0:
            val_acc, val_bca, val_f1 = evaluate_validation(model, val_loader, args, prototype)
            print('Epoch[{}/{}], val acc = {:.4f}, val bca = {:.4f}, val f1 Score = {:.4f}.'.format(
                e, 10, val_acc, val_bca, val_f1))

    # --------------------
    # 重新计算prototype并保存
    # --------------------
    model.eval()
    fea_all = []
    true_y = []
    for _, (b_x, b_y) in enumerate(train_loader):
        b_x, b_y = b_x.cuda(), b_y.cuda()
        
        _, fea = model.forward(b_x)
        fea_all.append(F.normalize(fea, dim=-1).detach().cpu().numpy()) 
        true_y.append(b_y.detach().cpu().numpy())
        
    fea_all = np.concatenate(fea_all)
    true_y = np.concatenate(true_y)
    index0 = np.where(true_y==0)
    fea0 = fea_all[index0].mean(axis=0)
    index1 = np.where(true_y==1)
    fea1 = fea_all[index1].mean(axis=0)
    prototype = torch.cat((torch.tensor(fea0).view(1, -1), torch.tensor(fea1).view(1, -1))).cuda()

    torch.save(model.state_dict(), args.OutputPathModels+'/task{}.npy'.format(task_id+1))
    torch.save(prototype, args.OutputPathModels+'/prototype{}.npy'.format(task_id+1))


def cosine_distance_matrix(a, b):
    return 1 - torch.mm(a, b.t())

def exp_setting(model, task_id, args):
    if task_id>0:
        for m in model.modules():
            if isinstance(m, channel_selection):
                m.indexes.data.fill_(1.0)
        for p in model.parameters():
            p.requires_grad = True
        for m in model.modules():
            if isinstance(m, PatchEmbedding):
                for p in m.parameters():
                    p.requires_grad = False

def main():
    global method

    print("START")
    args.OutputPathModels = os.path.join(OutputPath, 'model')
    if not os.path.exists(args.OutputPathModels):
        os.makedirs(args.OutputPathModels)

    model = Conformer(num_classes=2, input_ch=args.input_ch).cuda()

    all_testloader = []
    args.all_prototype = []
    acc_matrix = np.zeros((len(patients), len(patients)))
    bca_matrix = np.zeros((len(patients), len(patients))) 
    f1_matrix = np.zeros((len(patients), len(patients)))

    for i, person_id in enumerate(patients):
        exp_setting(model, i, args)
        train_loader, test_loader = get_data(person_id, args)

        orig_model_path = args.OutputPathModels+'/task{}_orig.npy'.format(i+1)
        if os.path.exists(orig_model_path) == False:
            train(model, train_loader, test_loader, args, i)
        model.load_state_dict(torch.load(orig_model_path))

        retrain_model_path = args.OutputPathModels+'/task{}.npy'.format(i+1)
        if os.path.exists(retrain_model_path) == False:
            retrain(model, train_loader, test_loader, args, i)
        model.load_state_dict(torch.load(retrain_model_path))

        if args.log_name == '1020' or args.log_name=='1021':
            args.all_prototype.append(torch.load(args.OutputPathModels+'/prototype{}.npy'.format(i+1)))

        all_testloader.append(test_loader)
        print('\n Testing until Task {}'.format(i+1))
        with torch.no_grad():
            for j in range(i+1):
                index_cur = torch.load(args.OutputPathModels+'/index{}.npy'.format(j+1))
                k=0
                for m in model.modules():
                    if isinstance(m, channel_selection):
                        m.indexes.data = index_cur[k].cuda()
                        k += 1

                prototype_cur = torch.load(args.OutputPathModels+'/prototype{}.npy'.format(j+1))
                test_acc, test_bca, test_f1 = evaluate_validation(model, all_testloader[j], args, prototype_cur)
                print('Test for patient {}: ACC = {:.4f}, BCA = {:.4f}, F1_score = {:.4f}'.format(patients[j], test_acc, test_bca, test_f1))
                acc_matrix[i, j] = test_acc
                bca_matrix[i, j] = test_bca
                f1_matrix[i, j] = test_f1
            avg_acc, _ = cal_metrics(acc_matrix, i)
            avg_bca, bwt = cal_metrics(bca_matrix, i)
            avg_f1, _ = cal_metrics(f1_matrix, i)
            print('Avg ACC = {:.4f}, Avg BCA = {:.4f}, Avg F1 Score = {:.4f}, BWT = {:.4f} \n'.format(avg_acc, avg_bca, avg_f1, bwt))

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', default='chb-mit', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--fold', default=1, type=int)
    parser.add_argument('--log_name', default='1016', type=str)
    parser.add_argument('--net', default='conformer', type=str)
    parser.add_argument('--gpu', default='0', type=str)

    parser.add_argument('--batchsize', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--ea', action="store_true")

    parser.add_argument('--method', default='none', type=str, choices=['ewc', 'mas', 'lwf'])
    parser.add_argument('--temp', default=0.1, type=float, help=' ')
    parser.add_argument('--s1', default=1e-4, type=float, help=' ')
    parser.add_argument('--s2', default=1e-5, type=float, help=' ')

    args = parser.parse_args()

    device = torch.device('cuda:'+args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # fix the seed for reproducibility
    seed = args.seed
    seed_torch(seed)

    OutputPath = os.path.join('./results_{}/fold{}'.format(args.dataset, args.fold), args.log_name)
    if not os.path.exists(OutputPath):
        os.makedirs(OutputPath)
    sys.stdout = Logger(output_dir=OutputPath, stream=sys.stdout)  # record log
    
    if args.dataset == 'chb-mit':
        patients = ["01", "03", "05", "08", "10", "13", "14", "15", "18", "20", "23"]
        args.input_ch = 18
    else:
        patients = ["12", "15", "17", "18", "21", "22", "23"]
        args.input_ch = 18

    main()

