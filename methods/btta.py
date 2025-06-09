"""
Copyright to btta Authors
"""

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
import torchvision
import math
import numpy as np
from decouple import config
import matplotlib.pyplot as plt
from einops import rearrange

import torch.nn.functional as F

class BTTA(nn.Module):
    """btta online adapts a model by entropy minimization with entropy and PLPD filtering & reweighting during testing.
    """
    def __init__(self, particles, optimizers, steps=1, episodic=False, deyo_margin=0.5*math.log(1000), margin_e0=0.4*math.log(1000)):
        super().__init__()
        wandb_log = False
        self.particles = particles
        self.optimizers = optimizers
        if wandb_log:
            import wandb
        self.steps = steps
        self.episodic = episodic
        counts = [1e-6,1e-6,1e-6,1e-6]
        correct_counts = [0,0,0,0]

        self.deyo_margin = deyo_margin
        self.margin_e0 = margin_e0

    def forward(self, x, iter_, targets=None, flag=True, group=None):
        if self.episodic:
            self.reset()
        
        if targets is None:
            for _ in range(self.steps):
                if flag:
                    outputs, backward, final_backward = forward_and_adapt_btta(x, iter_, self.particles,
                                                                              self.optimizers, self.deyo_margin,
                                                                              self.margin_e0, targets, flag, group)
                else:
                    outputs = forward_and_adapt_btta(x, iter_, self.particles,
                                                    self.optimizers, self.deyo_margin,
                                                    self.margin_e0, targets, flag, group)
        else:
            for _ in range(self.steps):
                if flag:
                    outputs = forward_and_adapt_btta(x, iter_, 
                                                        self.particles, 
                                                        self.optimizers, 
                                                        self.deyo_margin,
                                                        self.margin_e0,
                                                        targets, flag, group)
                else:
                    outputs = forward_and_adapt_btta(x, iter_, 
                                                    self.particles, 
                                                    self.optimizers, 
                                                    self.deyo_margin,
                                                    self.margin_e0,
                                                    targets, flag, group, self)
        if targets is None:
            if flag:
                return outputs
            else:
                return outputs
        else:
            if flag:
                return outputs
            else:
                return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.ema = None


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    #temprature = 1.1 #0.9 #1.2
    #x = x ** temprature #torch.unsqueeze(temprature, dim=-1)
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def kl_divergence(p, q):

    return F.kl_div(F.log_softmax(p, dim=1), F.softmax(q, dim=1), reduction='batchmean') + \
           F.kl_div(F.log_softmax(q, dim=1), F.softmax(p, dim=1), reduction='batchmean')  

def compute_input_gradients(model, imgs):
    imgs.requires_grad = True
    logits = model(imgs)
    entropies = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    loss = entropies.mean(0)
    input_gradients = torch.autograd.grad(outputs=loss, inputs=imgs, create_graph=True)[0].detach()
    imgs.requires_grad = False
    model.zero_grad()
    return input_gradients, entropies, logits

def compute_pairwise_dissimilarity(gradients_list):
    num_models = len(gradients_list)
    dissimilarity_matrix = torch.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                dissimilarity_matrix[i, j] = torch.norm(gradients_list[i] - gradients_list[j], p=2)
    
    mean_dissimilarity = dissimilarity_matrix.mean()
    return mean_dissimilarity


def compute_pairwise_alignment(gradients_list):
    num_models = len(gradients_list)
    alignment_matrix = torch.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            if i != j:
                alignment_matrix[i, j] = torch.dot(gradients_list[i].view(-1), gradients_list[j].view(-1))
    
    mean_alignment = alignment_matrix.mean()
    return mean_alignment



def compute_cosine_similarity(gradients_list):
    num_models = len(gradients_list)
    similarity_matrix = torch.zeros((num_models, num_models))
    
    for i in range(num_models):
        for j in range(num_models):
            if i != j:

                dot_product = torch.dot(gradients_list[i].view(-1), gradients_list[j].view(-1))
                norm_i = torch.norm(gradients_list[i].view(-1))
                norm_j = torch.norm(gradients_list[j].view(-1))
                
                similarity_matrix[i, j] = dot_product / (norm_i * norm_j)
    
    mean_similarity = similarity_matrix.mean()
    return mean_similarity



def update_gradiants(all_pgs, h_kernel):

    if np.random.rand() < 0.6:
        return
    if h_kernel is None or h_kernel <= 0:
        h_kernel = 0.008  # 1
    dists = []
    alpha = 0.001  # if t < 100 else 0.0
    new_parameters = [None] * len(all_pgs)

    for i in range(len(all_pgs)):
        new_parameters[i] = {}
        for l, p in enumerate(all_pgs[i].parameters()):
            if p.grad is None:
                new_parameters[i][l] = None
            else:
                new_parameters[i][l] = p.grad.data.new(
                    p.grad.data.size()).zero_()
        for j in range(len(all_pgs)):
            # if i == j:
            #     continue
            for l, params in enumerate(
                    zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                p, p2 = params
                if p.grad is None or p2.grad is None:
                    continue
                if p is p2:
                    dists.append(0)
                    new_parameters[i][l] = new_parameters[i][l] + \
                        p.grad.data
                else:
                    d = (p.data - p2.data).norm(2)
                    # if p is not p2:
                    dists.append(d.cpu().item())
                    kij = torch.exp(-(d**2) / h_kernel**2 / 2)
                    new_parameters[i][l] = (
                        ((new_parameters[i][l] + p2.grad.data) -
                         (d / h_kernel**2) * alpha) /
                        float(len(all_pgs))) * kij
    h_kernel = np.median(dists)
    h_kernel = np.sqrt(0.5 * h_kernel / np.log(len(all_pgs)) + 1)
    for i in range(len(all_pgs)):
        for l, p in enumerate(all_pgs[i].parameters()):
            if p.grad is not None:
                p.grad.data = new_parameters[i][l]
    return h_kernel


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt_btta(x, iter_, particles, optimizers, deyo_margin, margin, targets=None, flag=True, group=None):
    """Forward and adapt model input data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    ent_flags = [False, False, False, False, False]
    ent_flag = ent_flags[:int(config('opt'))]
    h_kernel = 0

    logits, all_entropies, all_input_gradients = [], [], []
    for i in range(len(particles)):
        optimizers[i].zero_grad() 

        input_grads, entropys, outputs = compute_input_gradients(particles[i], x)
        l2_norm = torch.norm(input_grads, p=2, dim=(1, 2, 3))
        logits.append(outputs)
        all_entropies.append(entropys)
        all_input_gradients.append(input_grads)

        if int(config('filter_ent')):
            filter_ids_1 = torch.where((entropys < deyo_margin))# & (l2_norm < args.grad_threshold))
        else:    
            filter_ids_1 = torch.where((entropys <= math.log(1000)))
        entropys = entropys[filter_ids_1]
        
        # backward = len(entropys)

   
        if len(entropys) !=0:
            patch_len = 4
            x_prime = x[filter_ids_1]
            x_prime = x_prime.detach()
            if config('aug_type')=='occ':
                first_mean = x_prime.view(x_prime.shape[0], x_prime.shape[1], -1).mean(dim=2)
                final_mean = first_mean.unsqueeze(-1).unsqueeze(-1)
                occlusion_window = final_mean.expand(-1, -1, int(config('occlusion_size')), int(config('occlusion_size')))
                x_prime[:, :, int(config('row_start')):int(config('row_start'))+int(config('occlusion_size')),int(config('column_start')):int(config('column_start'))+int(config('occlusion_size'))] = occlusion_window
            elif config('aug_type')=='patch':
                resize_t = torchvision.transforms.Resize(((x.shape[-1]//patch_len)*patch_len,(x.shape[-1]//patch_len)*patch_len))
                resize_o = torchvision.transforms.Resize((x.shape[-1],x.shape[-1]))
                x_prime = resize_t(x_prime)
                x_prime = rearrange(x_prime, 'b c (ps1 h) (ps2 w) -> b (ps1 ps2) c h w', ps1=patch_len, ps2=patch_len)
                perm_idx = torch.argsort(torch.rand(x_prime.shape[0],x_prime.shape[1]), dim=-1)
                x_prime = x_prime[torch.arange(x_prime.shape[0]).unsqueeze(-1),perm_idx]
                x_prime = rearrange(x_prime, 'b (ps1 ps2) c h w -> b c (ps1 h) (ps2 w)', ps1=patch_len, ps2=patch_len)
                x_prime = resize_o(x_prime)
            elif config('aug_type')=='pixel':
                x_prime = rearrange(x_prime, 'b c h w -> b c (h w)')
                x_prime = x_prime[:,:,torch.randperm(x_prime.shape[-1])]
                x_prime = rearrange(x_prime, 'b c (ps1 ps2) -> b c ps1 ps2', ps1=x.shape[-1], ps2=x.shape[-1])
            with torch.no_grad():
                outputs_prime = particles[i](x_prime)
            
            prob_outputs = outputs[filter_ids_1].softmax(1)
            prob_outputs_prime = outputs_prime.softmax(1)

            cls1 = prob_outputs.argmax(dim=1)

            plpd = torch.gather(prob_outputs, dim=1, index=cls1.reshape(-1,1)) - torch.gather(prob_outputs_prime, dim=1, index=cls1.reshape(-1,1))
            plpd = plpd.reshape(-1)
            
            if int(config('filter_plpd')):
                filter_ids_2 = torch.where(plpd > float(config('plpd_threshold')))
            else:
                filter_ids_2 = torch.where(plpd >= -2.0)
            entropys = entropys[filter_ids_2]

            
            if len(entropys)!=0:
                ent_flag[i]= True

            plpd = plpd[filter_ids_2]
            

            if int(config('reweight_ent')) or int(config('reweight_plpd')):
                coeff = (int(config('reweight_ent')) * (1 / (torch.exp(((entropys.clone().detach()) - margin)))) +
                        int(config('reweight_plpd')) * (1 / (torch.exp(-1. * plpd.clone().detach())))
                        )            
                entropys = entropys.mul(coeff)
            all_entropies[i]= entropys

            del x_prime
            del plpd    

        if  config('div_type').lower() =='ens':   

            if len(entropys) !=0:
        
                loss = entropys.mean(0)
                loss.backward()
                optimizers[i].step()



    if config('div_type').lower() =='kl':
        if ent_flag != [False, False, False]:
            total_kl_loss = 0.0
            num_pairs = 0
            for i in range(len(particles)):
                for k in range(i + 1, len(particles)):
                    kl_loss = kl_divergence(logits[i], logits[k])
                    total_kl_loss += kl_loss
                    num_pairs += 1
            # Avoid zero in denom
            mean_kl_loss = total_kl_loss / max(num_pairs, 1)  

            for p, (model, ent) in enumerate(zip(particles, all_entropies)):
                loss = ent.mean(0) - float(config('lambda_kl')) * mean_kl_loss
                retain_graph = (p < len(particles) - 1)  
                loss.backward(retain_graph=retain_graph)


            for v, optimizer in enumerate(optimizers):
                if ent_flag[v] == True:
                    optimizers[v].step()

    
    if config('div_type').lower()=='grad':

        if ent_flag != [False, False, False]:
            diversification_loss = compute_pairwise_dissimilarity(all_input_gradients)

            for p, (model, ent) in enumerate(zip(particles, all_entropies)):
                loss = ent.mean(0) - float(config('lambda_diversification'))  * diversification_loss
                retain_graph = (p < len(particles) - 1)  
                loss.backward(retain_graph=retain_graph)

            for v, optimizer in enumerate(optimizers):
                if ent_flag[v] == True:
                    optimizers[v].step()

    if config('div_type').lower()=='svgd':
        if ent_flag != [False, False, False]:
            h_kernel = update_gradiants(particles, h_kernel)
            for p, (model, ent) in enumerate(zip(particles, all_entropies)):
                loss = ent.mean(0)
                retain_graph = (p < len(particles) - 1)  
                loss.backward(retain_graph=retain_graph)

            for v, optimizer in enumerate(optimizers):
                if ent_flag[v] == True:
                    optimizers[v].step()



    logits = torch.stack(logits).mean(0)

    return logits

def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        # skip top layers for adaptation: layer4 for ResNets and blocks9-11 for Vit-Base
        if 'layer4' in nm:
            continue
        if 'blocks.9' in nm:
            continue
        if 'blocks.10' in nm:
            continue
        if 'blocks.11' in nm:
            continue
        if 'norm.' in nm:
            continue
        if nm in ['norm']:
            continue

        if isinstance(m, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    return params, names


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    """Configure model for use with btta."""
    # train mode, because btta optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what btta updates
    model.requires_grad_(False)
    # configure norm for btta updates: enable grad + force batch statisics (this only for BN models)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        # LayerNorm and GroupNorm for ResNet-GN and Vit-LN models
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model

