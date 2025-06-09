
import os
import open_clip
import math
import copy
import random
import torch
import json
import numpy as np
from torchvision.datasets import CIFAR10, CIFAR100
from decouple import config
from torchvision.datasets import CIFAR10
from utilities import Paths, calculate_metrics, brier_score, calculate_auroc_multiclass
from preprocessor import load_data_cifar, load_data_casting, TrainDataset
from bayes_wrap import validate, load_ensemble



from methods import tent, eata, sam, sar, deyo, btta
from utils.utils import get_logger
from utils.cli_utils import *
import time
import wandb

import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

torch.cuda.set_device(int(config('device')))

seed = int(config('seed'))
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


''' -----------------------   Set path ------------------------------'''
paths = Paths(config)
paths.create_path()


''' -----------------------   loading CLIP ViT ------------------------------'''

device =f"cuda:{int(config('device'))}" if torch.cuda.is_available() else "cpu"

# mdl, preprocess = clip.load('ViT-B/32', device)
mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')


if config('dataset_name').upper() == "CIFAR10":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("Data/" + "cifar-10-batches-py")
    train = CIFAR10(root, download=True, train=True)
    test = CIFAR10(root, download=True, train=False, transform=preprocess)

    print('cifar10 loaded')

elif config('dataset_name').upper() == "CIFAR100":

    ''' -----------------------   Loading the Data   ----------------------- '''
    root = os.path.expanduser("Data/" + "cifar-100-batches-py")
    train = CIFAR100(root, download=True, train=True)
    test = CIFAR100(root, download=True, train=False, transform=preprocess)

    print('cifar100 loaded')

elif config('dataset_name').upper() == "IMAGENET":
    ''' -----------------------   Loading the Data   ----------------------- '''
    test_set = TrainDataset(data_folder=f'/media/tower2/DATA4/Afshar/datasets/imgnt-c/{config("corruption")}/5', transform=preprocess)

    test_loader = load_data_imagenet(test_set, device)

    print('ImageNet loaded')

elif config('dataset_name').upper() == "CASTING":
    ''' -----------------------   Loading the Data   ----------------------- '''
    # test_set = TrainDataset(data_folder=f'/home/tower2/Documents/Rejisa/adapt/casting_data/casting_data/casting_data/test', transform=preprocess)
    # test_set = TrainDataset(data_folder=f'/media/tower2/DATA4/Afshar/datasets/casting_xray/test/', transform=preprocess)
    test_set = TrainDataset(data_folder=f'/media/tower2/DATA4/Afshar/datasets/casting-c/5/{config("corruption")}/', transform=preprocess)
    # test_loader = load_data_casting(test_set, device)

    print('Casting loaded')




############################################################################################
##################################################################################################################

if config('dataset_name').upper() == "CIFAR10":
    path_address = 'Model/max_min_cifar10/'

elif config('dataset_name').upper() == "CIFAR100":
    path_address = 'Model/max_min_cifar100/'

elif(config('dataset_name').upper() == "CASTING"):
    path_address = 'Model/max_min_casting/'


mdl, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
ens_addr = [f for f in os.listdir(path_address) if f[-4:]=='1.pt']

# corruptions = ['brightness', 'contrast', 'elastic_transform', 'gaussian_blur', 'gaussian_noise', 'impulse_noise', 'jpeg_compression', 'pixelate', 'saturate', 'shot_noise', 'spatter', 'speckle_noise']
corruptions = ['gaussian_noise']


performance = []
for corruption in corruptions:
    if config('dataset_name').upper() in ['CIFAR10', 'CIFAR100']:
        corrupted_testset = np.load(f"Data/{corruption}.npy")
        lbls = np.load("Data/labels.npy")
        test.data = corrupted_testset
        test.targets = lbls
        test.transform = preprocess

        trainloaders, validation_loader, test_loader = load_data_cifar(preprocess, train, test, device)

    elif(config('dataset_name').upper() == "CASTING"):   

        test_set = TrainDataset(data_folder=f'/media/tower2/DATA4/Afshar/datasets/casting-c/{int(config("severity"))}/{corruption}/', transform=preprocess)
        # test_set = TrainDataset(data_folder=f'/media/tower2/DATA4/Afshar/datasets/casting_xray/test/', transform=preprocess)
        test_loader = load_data_casting(test_set, device)


    # if  config('method').lower()!='btta':
    ensemble = load_ensemble(ens_addr, path_address)

    e_margin = float(config('e_margin'))
    sar_margin_e0 = float(config('sar_margin_e0'))
    deyo_margin = float(config('deyo_margin'))
    deyo_margin_e0 = float(config('deyo_margin_e0'))

    e_margin *= math.log(float(config('num_class')))
    sar_margin_e0 *= math.log(float(config('num_class')))
    deyo_margin *= math.log(float(config('num_class'))) # for thresholding
    deyo_margin_e0 *= math.log(float(config('num_class'))) # for reweighting tuning

    if config('method').lower()=='tent':
        print(f"method: {config('method')}, corruption: {corruption}")
        net = tent.configure_model(ensemble[0].to(device))
        params, param_names = tent.collect_params(net)
        # print(param_names)

        optimizer = torch.optim.SGD(params, float(config('learning_rate')), momentum=0.9) 
        tented_model = tent.Tent(net, optimizer)

        acc1, acc5, ECE, MCE, brier, AUROC = validate(test_loader, tented_model, device, mode='eval')


    elif config('method').lower() == "no_adapt":
        print(f"method: {config('method')}, corruption: {corruption}")
        tented_model = ensemble[0].to(device)
        acc1, acc5 , ECE, MCE, brier, AUROC = validate(test_loader, tented_model, device, mode='eval')


    elif config('method').lower() == "eata":
        print(f"method: {config('method')}, corruption: {corruption}")
        if config("eata_fishers"):
            print('EATA!')

            net = eata.configure_model(ensemble[0].to(device))
            params, param_names = eata.collect_params(net)
            # print(param_names)

            ewc_optimizer = torch.optim.SGD(params, 0.001)
            fishers = {}
            train_loss_fn = torch.nn.CrossEntropyLoss().cuda()
            st_time = time.time()
            for iter_, data in enumerate(test_loader, start=1):

                images, targets = data[0], data[1]
                images, targets = images.to(device), targets.to(device)

                outputs = net(images)
                _, targets = outputs.max(1)
                loss = train_loss_fn(outputs, targets)
                loss.backward()
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        if iter_ > 1:
                            fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                        else:
                            fisher = param.grad.data.clone().detach() ** 2
                        if iter_ == len(test_loader):
                            fisher = fisher / iter_
                        fishers.update({name: [fisher, param.data.clone().detach()]})
                ewc_optimizer.zero_grad()
                print(f'\r fisher iter {iter_}/{len(test_loader)}', end='')
            print("\ncompute fisher matrices finished")
            del ewc_optimizer
        else:
            net = eata.configure_model(ensemble[0].to(device))
            params, param_names = eata.collect_params(net)
            print('ETA!')
            fishers = None
        
        end_time = time.time()

        print(f'time1: {end_time - st_time}')
        optimizer = torch.optim.SGD(params, float(config('learning_rate')), momentum=0.9)
        adapt_model = eata.EATA( net, optimizer, fishers, int(config('fisher_alpha')), e_margin = e_margin, d_margin=float(config('d_margin')))
        acc1, acc5, ECE, MCE, brier, AUROC = validate(test_loader, adapt_model, device, mode='eval')


    elif config('method').lower() == "sar":

        print(f"method: {config('method')}, corruption: {corruption}")
        biased = False
        wandb_log = False

        st_time = time.time()

        net = sar.configure_model(ensemble[0].to(device))
        params, param_names = sar.collect_params(net)
        # print(param_names)

        base_optimizer = torch.optim.SGD
        optimizer = sam.SAM(params, base_optimizer, lr=float(config('learning_rate')), momentum=0.9)
        adapt_model = sar.SAR(net, optimizer, margin_e0= sar_margin_e0)

        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        if biased:
            LL_AM = AverageMeter('LL Acc', ':6.2f')
            LS_AM = AverageMeter('LS Acc', ':6.2f')
            SL_AM = AverageMeter('SL Acc', ':6.2f')
            SS_AM = AverageMeter('SS Acc', ':6.2f')
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
                prefix='Test: ')
        else:
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
        
        end = time.time()
        correct_count = [0,0,0,0]
        total_count = [1e-6,1e-6,1e-6,1e-6]
        logits_test, targets_test= [], []
        for i, (images, target) in enumerate(test_loader):
            # images, target = dl[0], dl[1]
            images, target = images.to(device), target.to(device)

            if biased:
                if config('dataset_name')=='Waterbirds':
                    place = dl[2]['place'].cuda()
                else:
                    place = dl[2].cuda()
                group = 2*target + place
            output = adapt_model(images)
            if biased:
                TFtensor = (output.argmax(dim=1)==target)
                
                for group_idx in range(4):
                    correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                    total_count[group_idx] += len(TFtensor[group==group_idx])
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 1))

            logits_test.append(output.cpu().detach().numpy())
            targets_test.append(target.cpu().detach().numpy())           

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if (i+1) % int(config('wandb_interval')) == 0:
                if biased:
                    LL = correct_count[0]/total_count[0]*100
                    LS = correct_count[1]/total_count[1]*100
                    SL = correct_count[2]/total_count[2]*100
                    SS = correct_count[3]/total_count[3]*100
                    LL_AM.update(LL, images.size(0))
                    LS_AM.update(LS, images.size(0))
                    SL_AM.update(SL, images.size(0))
                    SS_AM.update(SS, images.size(0))
                    if wandb_log:
                        wandb.log({f'{config("corruption")}/LL': LL,
                                    f'{config("corruption")}/LS': LS,
                                    f'{config("corruption")}/SL': SL,
                                    f'{config("corruption")}/SS': SS,
                                    })
                if wandb_log:
                    wandb.log({f'{config("corruption")}/top1': top1.avg,
                                f'{config("corruption")}/top5': top5.avg
                                })

            if (i+1) % float(config('wandb_interval')) == 0:
                progress.display(i)

        acc1 = top1.avg
        acc5 = top5.avg
        
        if biased:
            print(f"- Detailed result under {corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
            if wandb_log:
                wandb.log({'final_avg/LL': LL,
                            'final_avg/LS': LS,
                            'final_avg/SL': SL,
                            'final_avg/SS': SS,
                            'final_avg/AVG': (LL+LS+SL+SS)/4,
                            'final_avg/WORST': min(LL,LS,SL,SS),
                            })

            avg = (LL+LS+SL+SS)/4
            print(f"Result under {corruption}. The adaptation accuracy of SAR is  average: {avg:.5f}")

        else:

            en_time = time.time()
            print(f'time: {en_time - st_time}')
            print(f"acc1s are {top1.avg.item()}")
            print(f"acc5s are {top5.avg.item()}")

            logits_test = np.concatenate(logits_test, axis=0)
            targets_test = np.concatenate(targets_test, axis=0)

            logits = torch.tensor(logits_test, dtype=torch.float32)
            targets = torch.tensor(targets_test, dtype=torch.long)
            preds = torch.argmax(logits, dim=1)
            correct = (preds == targets).sum().item()
            ac = correct / targets.size(0)


            num_classes = int(config('num_class'))
            ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
            brier = brier_score(logits_test, targets_test, num_classes)
            AUROC = calculate_auroc_multiclass(logits_test, targets_test, num_classes)


        print(
            '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f, MCE = %.4f, Brier = %.5f, AUROC = %.4f' %
            (ac, ECE, MCE, brier, AUROC)
        )     

    elif config('method').lower() == "deyo":

        print(f"method: {config('method')}, aug_type: {config('aug_type')}, corruption: {corruption}")
        biased = False
        wandb_log = False       

        st_time = time.time()

        net = deyo.configure_model(ensemble[0].to(device))
        params, param_names = deyo.collect_params(net)
        # print(param_names)

        optimizer = torch.optim.SGD(params, float(config('learning_rate')), momentum=0.9)
        adapt_model = deyo.DeYO(net, optimizer, deyo_margin= deyo_margin, margin_e0= deyo_margin_e0)

        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        if biased:
            LL_AM = AverageMeter('LL Acc', ':6.2f')
            LS_AM = AverageMeter('LS Acc', ':6.2f')
            SL_AM = AverageMeter('SL Acc', ':6.2f')
            SS_AM = AverageMeter('SS Acc', ':6.2f')
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
                prefix='Test: ')
        else:
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
        end = time.time()
        count_backward = 1e-6
        final_count_backward =1e-6
        count_corr_pl_1 = 0
        count_corr_pl_2 = 0
        total_count_backward = 1e-6
        total_final_count_backward =1e-6
        total_count_corr_pl_1 = 0
        total_count_corr_pl_2 = 0
        correct_count = [0,0,0,0]
        total_count = [1e-6,1e-6,1e-6,1e-6]
        logits_test, targets_test= [], []
        for i, dl in enumerate(test_loader):
            images, target = dl[0], dl[1]
            images, target = images.to(device), target.to(device)

            if biased:
                if config('dataset_name')=='Waterbirds':
                    place = dl[2]['place'].cuda()
                else:
                    place = dl[2].cuda()
                group = 2*target + place
            else:
                group=None

            output, backward, final_backward, corr_pl_1, corr_pl_2 = adapt_model(images, i, target, group=group)
            if biased:
                TFtensor = (output.argmax(dim=1)==target)
                
                for group_idx in range(4):
                    correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                    total_count[group_idx] += len(TFtensor[group==group_idx])
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 1))

            logits_test.append(output.cpu().detach().numpy())
            targets_test.append(target.cpu().detach().numpy())  
                
            count_backward += backward
            final_count_backward += final_backward
            total_count_backward += backward
            total_final_count_backward += final_backward
            
            count_corr_pl_1 += corr_pl_1
            count_corr_pl_2 += corr_pl_2
            total_count_corr_pl_1 += corr_pl_1
            total_count_corr_pl_2 += corr_pl_2

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            if (i+1) % int(config('wandb_interval')) == 0:
                if biased:
                    LL = correct_count[0]/total_count[0]*100
                    LS = correct_count[1]/total_count[1]*100
                    SL = correct_count[2]/total_count[2]*100
                    SS = correct_count[3]/total_count[3]*100
                    LL_AM.update(LL, images.size(0))
                    LS_AM.update(LS, images.size(0))
                    SL_AM.update(SL, images.size(0))
                    SS_AM.update(SS, images.size(0))
                    if wandb_log:
                        wandb.log({f"{config('corruption')}/LL": LL,
                                    f"{config('corruption')}/LS": LS,
                                    f"{config('corruption')}/SL": SL,
                                    f"{config('corruption')}/SS": SS,
                                    })

                if wandb_log:
                    wandb.log({f'{config("corruption")}/top1': top1.avg,
                                f'{config("corruption")}/top5': top5.avg,
                                f'acc_pl_1': count_corr_pl_1/count_backward,
                                f'acc_pl_2': count_corr_pl_2/final_count_backward,
                                f'count_backward': count_backward,
                                f'final_count_backward': final_count_backward})
                
                count_backward = 1e-6
                final_count_backward =1e-6
                count_corr_pl_1 = 0
                count_corr_pl_2 = 0

            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % int(config('wandb_interval')) == 0:
                progress.display(i)

        acc1 = top1.avg
        acc5 = top5.avg
        
        if biased:
            print(f"- Detailed result under {corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
            if wandb_log:
                wandb.log({'final_avg/LL': LL,
                            'final_avg/LS': LS,
                            'final_avg/SL': SL,
                            'final_avg/SS': SS,
                            'final_avg/AVG': (LL+LS+SL+SS)/4,
                            'final_avg/WORST': min(LL,LS,SL,SS),
                            })
            
        if wandb_log:
            wandb.log({f'{config("corruption")}/top1': acc1,
                        f'{config("corruption")}/top5': acc5,
                        f'total_acc_pl_1': total_count_corr_pl_1/total_count_backward,
                        f'total_acc_pl_2': total_count_corr_pl_2/total_final_count_backward,
                        f'total_count_backward': total_count_backward,
                        f'total_final_count_backward': total_final_count_backward})

        if biased:
            avg = (LL+LS+SL+SS)/4
            print(f"Result under {corruption}. The adaptation accuracy of DeYO is  average: {avg:.5f}")

        else:
            en_time = time.time()

            print(f'time: {en_time - st_time}')
            print(f"acc1s are {top1.avg.item()}")
            print(f"acc5s are {top5.avg.item()}")

        logits_test = np.concatenate(logits_test, axis=0)
        targets_test = np.concatenate(targets_test, axis=0)

        logits = torch.tensor(logits_test, dtype=torch.float32)
        targets = torch.tensor(targets_test, dtype=torch.long)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        ac = correct / targets.size(0)


        num_classes = int(config('num_class'))
        ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
        brier = brier_score(logits_test, targets_test, num_classes)
        AUROC = calculate_auroc_multiclass(logits_test, targets_test, num_classes)


        print(
            '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f, MCE = %.4f, Brier = %.5f, AUROC = %.4f' %
            (ac, ECE, MCE, brier, AUROC)
        )  
        acc1 = top1.avg
        acc5 = top5.avg



    elif config('method').lower() == "btta":

        print(f"method: {config('method')}, div_type: {config('div_type')}, corruption: {corruption}")
        biased = False
        wandb_log = False       

        st_time = time.time()

        lr = float(config('learning_rate'))

        particles, optimizers = [], []
        # lr_rates = [0.05, 0.06, 0.08, 0.05, 0.06]
        # momentum_list=[0.9, 0.9, 0.85, 0.88, 0.88]

        lr_rates = [0.06, 0.06, 0.06, 0.06, 0.06]
        momentum_list=[0.91, 0.9, 0.89, 0.87, 0.88]


        for i in range(int(config('opt'))):
            NET = copy.deepcopy(ensemble[0].to(device))
            NET = btta.configure_model(NET)
            params, param_names = btta.collect_params(NET)

            optimizer = torch.optim.SGD(params, lr_rates[i], momentum = momentum_list[i])
            particles.append(NET)
            optimizers.append(optimizer)


        adapt_model = btta.BTTA(particles, optimizers, deyo_margin= deyo_margin, margin_e0= deyo_margin_e0)



        batch_time = AverageMeter('Time', ':6.3f')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        if biased:
            LL_AM = AverageMeter('LL Acc', ':6.2f')
            LS_AM = AverageMeter('LS Acc', ':6.2f')
            SL_AM = AverageMeter('SL Acc', ':6.2f')
            SS_AM = AverageMeter('SS Acc', ':6.2f')
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, top1, top5, LL_AM, LS_AM, SL_AM, SS_AM],
                prefix='Test: ')
        else:
            progress = ProgressMeter(
                len(test_loader),
                [batch_time, top1, top5],
                prefix='Test: ')
        end = time.time()
        count_backward = 1e-6
        final_count_backward =1e-6
        count_corr_pl_1 = 0
        count_corr_pl_2 = 0
        total_count_backward = 1e-6
        total_final_count_backward =1e-6
        total_count_corr_pl_1 = 0
        total_count_corr_pl_2 = 0
        correct_count = [0,0,0,0]
        total_count = [1e-6,1e-6,1e-6,1e-6]
        logits_test, targets_test= [], []
        for i, dl in enumerate(test_loader):
            images, target = dl[0], dl[1]
            images, target = images.to(device), target.to(device)

            if biased:
                if config('dataset_name')=='Waterbirds':
                    place = dl[2]['place'].cuda()
                else:
                    place = dl[2].cuda()
                group = 2*target + place
            else:
                group=None

            output = adapt_model(images, i, target, group=group)
            if biased:
                TFtensor = (output.argmax(dim=1)==target)
                
                for group_idx in range(4):
                    correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                    total_count[group_idx] += len(TFtensor[group==group_idx])
                acc1, acc5 = accuracy(output, target, topk=(1, 1))
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 1))

            logits_test.append(output.cpu().detach().numpy())
            targets_test.append(target.cpu().detach().numpy())  
                
            # count_backward += backward
            # final_count_backward += final_backward
            # total_count_backward += backward
            # total_final_count_backward += final_backward
            
            # count_corr_pl_1 += corr_pl_1
            # count_corr_pl_2 += corr_pl_2
            # total_count_corr_pl_1 += corr_pl_1
            # total_count_corr_pl_2 += corr_pl_2

            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            
            if (i+1) % int(config('wandb_interval')) == 0:
                if biased:
                    LL = correct_count[0]/total_count[0]*100
                    LS = correct_count[1]/total_count[1]*100
                    SL = correct_count[2]/total_count[2]*100
                    SS = correct_count[3]/total_count[3]*100
                    LL_AM.update(LL, images.size(0))
                    LS_AM.update(LS, images.size(0))
                    SL_AM.update(SL, images.size(0))
                    SS_AM.update(SS, images.size(0))
                    if wandb_log:
                        wandb.log({f"{config('corruption')}/LL": LL,
                                    f"{config('corruption')}/LS": LS,
                                    f"{config('corruption')}/SL": SL,
                                    f"{config('corruption')}/SS": SS,
                                    })

                if wandb_log:
                    wandb.log({f'{config("corruption")}/top1': top1.avg,
                                f'{config("corruption")}/top5': top5.avg,
                                f'acc_pl_1': count_corr_pl_1/count_backward,
                                f'acc_pl_2': count_corr_pl_2/final_count_backward,
                                f'count_backward': count_backward,
                                f'final_count_backward': final_count_backward})
                
                count_backward = 1e-6
                final_count_backward =1e-6
                count_corr_pl_1 = 0
                count_corr_pl_2 = 0

            batch_time.update(time.time() - end)
            end = time.time()

            if (i+1) % int(config('wandb_interval')) == 0:
                progress.display(i)

        acc1 = top1.avg
        acc5 = top5.avg
        
        if biased:
            print(f"- Detailed result under {corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
            if wandb_log:
                wandb.log({'final_avg/LL': LL,
                            'final_avg/LS': LS,
                            'final_avg/SL': SL,
                            'final_avg/SS': SS,
                            'final_avg/AVG': (LL+LS+SL+SS)/4,
                            'final_avg/WORST': min(LL,LS,SL,SS),
                            })
            
        if wandb_log:
            wandb.log({f'{config("corruption")}/top1': acc1,
                        f'{config("corruption")}/top5': acc5,
                        f'total_acc_pl_1': total_count_corr_pl_1/total_count_backward,
                        f'total_acc_pl_2': total_count_corr_pl_2/total_final_count_backward,
                        f'total_count_backward': total_count_backward,
                        f'total_final_count_backward': total_final_count_backward})

        if biased:
            avg = (LL+LS+SL+SS)/4
            print(f"Result under {corruption}. The adaptation accuracy of DeYO is  average: {avg:.5f}")

        else:
            en_time = time.time()

            print(f'time: {en_time - st_time}')
            print(f"acc1s are {top1.avg.item()}")
            print(f"acc5s are {top5.avg.item()}")

        logits_test = np.concatenate(logits_test, axis=0)
        targets_test = np.concatenate(targets_test, axis=0)

        logits = torch.tensor(logits_test, dtype=torch.float32)
        targets = torch.tensor(targets_test, dtype=torch.long)
        preds = torch.argmax(logits, dim=1)
        correct = (preds == targets).sum().item()
        ac = correct / targets.size(0)


        num_classes = int(config('num_class'))
        ECE, MCE = calculate_metrics(logits_test, targets_test, num_classes, n_bins=15)
        brier = brier_score(logits_test, targets_test, num_classes)
        AUROC = calculate_auroc_multiclass(logits_test, targets_test, num_classes)


        print(
            '[Calibration - Default T=1] ACC = %.4f, ECE = %.4f, MCE = %.4f, Brier = %.5f, AUROC = %.4f' %
            (ac, ECE, MCE, brier, AUROC)
        )  
        acc1 = top1.avg
        acc5 = top5.avg





    perf = {  
                    "Corruption": corruption,
                    "acc1": acc1.item(),
                    "acc5": acc5.item(),   
                    "ECE":  ECE,
                    "MCE":  MCE,
                    "brier":  brier,
                    "AUROC": AUROC
                                                }
    
    performance.append(perf)
if config('method').lower() == "btta":
    labels_info_path = f"Results/{config('dataset_name')}_{config('method')}_{config('div_type')}_{config('seed')}_{config('batch_size')}.json"
else:
    labels_info_path = f"Results/{config('dataset_name')}_{config('method')}_{config('seed')}_{config('batch_size')}.json"

with open(labels_info_path, 'w') as fp:
    json.dump(performance, fp, indent=2)
