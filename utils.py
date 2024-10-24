import numpy as np
import os
from scipy.linalg import fractional_matrix_power
import scipy.io

import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split
from scipy.stats import zscore


fs = 256
patients = ["01", "03", "05", "08", "10", "13", "14", "15", "18", "20", "23"]

def EA(x): # x(bs, channel, point)
    bs, channel, point = x.shape
    # 计算协方差矩阵
    x_mean = x - np.mean(x, axis=2, keepdims=True)
    cov = np.einsum('bij,bkj->bik', x_mean, x_mean) / (point - 1)  # (bs, channel, channel)
    # 计算平均协方差矩阵
    refEA = np.mean(cov, axis=0)  # (channel, channel)
    sqrtRefEA = fractional_matrix_power(refEA, -0.5) + (1e-8) * np.eye(channel)
    # 计算XEA
    XEA = np.einsum('ij,bjk->bik', sqrtRefEA, x)  # (bs, channel, point)
    
    return XEA, sqrtRefEA

def torch_transform(TORCH_X, TORCH_Y):
    TORCH_X = np.expand_dims(TORCH_X, 1)
    X_train = torch.from_numpy(TORCH_X).to(torch.float32)
    y_train = torch.from_numpy(TORCH_Y)
    dataset = Data.TensorDataset(X_train, y_train)
    return dataset

def get_data_nicu(person_id, args=None):
    DataPath = '/mnt/data4/prepro/'
    P_DataPath = os.path.join(DataPath, 'prepro_subject_{}'.format(person_id))
    mat_data = scipy.io.loadmat(P_DataPath)
    X = mat_data['X'].transpose(0, 2, 1)
    y = mat_data['y'].reshape(-1)
    print(X.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = torch_transform(X_train, y_train)
    test_dataset = torch_transform(X_val, y_val)
    trainloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)
    testloader = DataLoader(test_dataset, batch_size=args.batchsize*4, shuffle=True, num_workers=4, drop_last=False)
    return trainloader, testloader



def get_data(person_id, args=None):
    if args.dataset == 'chb-mit':
        DataPath = '/mnt/data4/jyan/data/CHB_MIT/detection_processed/'
        P_DataPath = os.path.join(DataPath, 'paz'+person_id)
    
        iterictalData = np.load(os.path.join(P_DataPath, 'iterictalData.npy'), allow_pickle=True)
        ictalData = np.load(os.path.join(P_DataPath, 'ictalData.npy'), allow_pickle=True)
        seizure_num = len(ictalData)
        seizure_list = list(range(seizure_num))
        remove = args.fold-1
        seizure_list.remove(remove)

        iter_train_x = np.concatenate(iterictalData[seizure_list], axis=0)
        ictal_train_x = np.concatenate(ictalData[seizure_list], axis=0)
        iter_test_x = iterictalData[remove] 
        ictal_test_x = ictalData[remove] 

        train_x = np.concatenate((iter_train_x, ictal_train_x), axis=0)
        train_y = np.concatenate((np.zeros(iter_train_x.shape[0], dtype=int), np.ones(ictal_train_x.shape[0], dtype=int)))
        test_x = np.concatenate((iter_test_x, ictal_test_x), axis=0)
        test_y = np.concatenate((np.zeros(iter_test_x.shape[0], dtype=int), np.ones(ictal_test_x.shape[0], dtype=int)))


        if args.ea:
            train_x, _ = EA(train_x)
            test_x, _ = EA(test_x)

        train_dataset = torch_transform(train_x, train_y)
        test_dataset = torch_transform(test_x, test_y)

        class_sample_count = np.array([iter_train_x.shape[0], ictal_train_x.shape[0]])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_y])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        trainloader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=sampler, num_workers=4, drop_last=True)
        testloader = DataLoader(test_dataset, batch_size=args.batchsize*4, shuffle=True, num_workers=4, drop_last=False)
        print('Data Loaded')
        return trainloader, testloader
    elif args.dataset == 'chsz':
        DataPath = '/mnt/data4/jyan/data/CHSZ/detection_processed_new/'
        P_DataPath = os.path.join(DataPath, 'paz'+person_id)
    
        iterictalData = np.load(os.path.join(P_DataPath, 'iterictalData.npy'), allow_pickle=True)
        ictalData = np.load(os.path.join(P_DataPath, 'ictalData.npy'), allow_pickle=True)
        
        seizure_num = len(ictalData)
        seizure_list = list(range(seizure_num))
        remove = args.fold-1
        seizure_list.remove(remove)

        iter_train_x = np.concatenate(iterictalData[seizure_list], axis=0)
        ictal_train_x = np.concatenate(ictalData[seizure_list], axis=0)
        iter_test_x = iterictalData[remove] 
        ictal_test_x = ictalData[remove] 

        train_x = np.concatenate((iter_train_x, ictal_train_x), axis=0)
        train_y = np.concatenate((np.zeros(iter_train_x.shape[0], dtype=int), np.ones(ictal_train_x.shape[0], dtype=int)))
        test_x = np.concatenate((iter_test_x, ictal_test_x), axis=0)
        test_y = np.concatenate((np.zeros(iter_test_x.shape[0], dtype=int), np.ones(ictal_test_x.shape[0], dtype=int)))

        # train_x = zscore(train_x, axis=(1,2), ddof=0)
        # test_x = zscore(test_x, axis=(1,2), ddof=0)

        if args.ea:
            train_x, _ = EA(train_x)
            test_x, _ = EA(test_x)

        train_dataset = torch_transform(train_x, train_y)
        test_dataset = torch_transform(test_x, test_y)

        class_sample_count = np.array([iter_train_x.shape[0], ictal_train_x.shape[0]])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_y])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        trainloader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=sampler, num_workers=4, drop_last=True)
        testloader = DataLoader(test_dataset, batch_size=args.batchsize*4, shuffle=True, num_workers=4, drop_last=False)
        print('Data Loaded')
        return trainloader, testloader
    elif args.dataset == 'nicu':
        trainloader, testloader = get_data_nicu(person_id, args)
        return trainloader, testloader
    

def get_data_all(args=None):
    if args.dataset == 'chb-mit':
        DataPath = '/mnt/data4/jyan/data/CHB_MIT/detection_processed/'
        patients = ["01", "03", "05", "08", "10", "13", "14", "15", "18", "20", "23"]
    else:
        DataPath = '/mnt/data4/jyan/data/CHSZ/detection_processed_new/'
        patients = ["14", "15", "17", "18", "21", "22"]
    for i, person_id in enumerate(patients):
        P_DataPath = os.path.join(DataPath, 'paz'+person_id)
        
        iterictalData = np.load(os.path.join(P_DataPath, 'iterictalData.npy'), allow_pickle=True)
        ictalData = np.load(os.path.join(P_DataPath, 'ictalData.npy'), allow_pickle=True)
        seizure_num = len(ictalData)
        seizure_list = list(range(seizure_num))
        remove = args.fold-1
        seizure_list.remove(remove)

        iter_train_x = np.concatenate(iterictalData[seizure_list], axis=0)
        ictal_train_x = np.concatenate(ictalData[seizure_list], axis=0)

        train_x = np.concatenate((iter_train_x, ictal_train_x), axis=0)
        train_y = np.concatenate((np.zeros(iter_train_x.shape[0], dtype=int), np.ones(ictal_train_x.shape[0], dtype=int)))

        if i==0:
            dataset_all_x = train_x
            dataset_all_y = train_y
        else:
            dataset_all_x = np.concatenate((dataset_all_x, train_x))
            dataset_all_y = np.concatenate((dataset_all_y, train_y))

    dataset = torch_transform(dataset_all_x, dataset_all_y)
    trainloader = DataLoader(dataset, batch_size=args.batchsize, shuffle=True, num_workers=4, drop_last=True)
    print('Data Loaded')
    return trainloader, None