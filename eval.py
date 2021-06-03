import torch
import torch.nn as nn
import torchvision
import os
import h5py
import numpy as np
from PIL import Image
from sacred import Experiment, cli_option
from lib.datasets import ds
from lib.datasets import HdF5Dataset
from lib.model import net
from lib.model import EfficientMORL, SlotAttention 
from lib.geco import GECO
from lib.metrics import adjusted_rand_index, matching_iou
from lib.dci import _compute_dci
from pathlib import Path
from tqdm import tqdm 
import pickle as pkl


@cli_option('-r','--local_rank')
def local_rank_option(args, run):
    run.info['local_rank'] = args

ex = Experiment('EVAL', ingredients=[ds, net], additional_cli_options=[local_rank_option])

torch.set_printoptions(threshold=10000, linewidth=300)

# Eval options (eval_type)
# 1. disentanglement_preprocessing: For each image in the test dataset, estimate the min,max values of each latent dim
# 2. disentanglement_viz: For a certain slot, store each recons image created by perturbing each latent dim, run (1) first
# 3. activeness: compute the average image pixel variance for 100 images by perturbing random latent dims/slots and store, run (1) first
# 4. dci_clevr: Compute the DCI metric for the CLEVR dataset
# 5. ARI_MSE_KL: For each image, calculate and store the final-level posterior ARI, MSE, and KL
# 6. sample_viz: For each image, sample and store intermediate-level reconstruction, masks, attention

@ex.config
def cfg():
    test = {
            'batch_size': 1,
            'output_size': [3,64,64],  # for visualization
            'mode': 'test',
            'num_images_to_process': 320,  # some eval/viz metrics will exit after 1 or 10 images
            'eval_type': 'disentanglement',
            'disentangle_slot': 0,  # The slot whose latent dims are perturbed, can be chosen via visual inspection
            'model': 'EfficientMORL',
            'num_workers': 8,
            'DDP_port': 29500,
            'out_dir': '',
            'checkpoint_dir': 'weights',
            'checkpoint': '',
            'use_geco': False,
            'geco_reconstruction_target': -20500,
            'geco_ema_alpha': 0.99,
            'geco_beta_stepsize': 1e-6,
            'experiment_name': 'NAME_HERE'
        }

def restore_from_checkpoint(test, checkpoint, local_rank):
    state = torch.load(checkpoint)
    
    if test['model'] == 'EfficientMORL':
        model = EfficientMORL(batch_size=test['batch_size'], use_geco=test['use_geco'])
    elif test['model'] == 'SlotAttention':
        model = SlotAttention(batch_size=test['batch_size'])
    else:
        raise RuntimeError('Model {} unknown'.format(test['model']))

    model = model.to(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                      output_device=local_rank)
    model.load_state_dict(state['model'])
    print(f'loaded {checkpoint}')
    model_geco = None
    if test['use_geco']:
        model_geco = GECO(test['geco_reconstruction_target'], test['geco_ema_alpha'])
    return model, model_geco


@ex.capture
def do_eval(test, seed, _run):

    # Fix random seed
    print(f'setting random seed to {seed}')
    
    # Auto-set by sacred
    # torch.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

    local_rank = 'cuda:{}'.format(_run.info['local_rank'])
    assert local_rank == 'cuda:0', 'Eval should be run with a single process'

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = str(test['DDP_port'])
    torch.distributed.init_process_group(backend='nccl')
    #device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)

    # Data
    te_dataset = HdF5Dataset(d_set=test['mode'])
    te_dataloader = torch.utils.data.DataLoader(te_dataset, batch_size=test['batch_size'],
                                                shuffle=True, num_workers=test['num_workers'],
                                                drop_last=True)
    checkpoint = Path(test['checkpoint_dir'], test['checkpoint'])
    
    if test['experiment_name'] == 'NAME_HERE':
        print('Please provide a valid name for this experiment')
        exit(1)
    
    out_dir = Path(test['out_dir'], 'results', test['experiment_name'],
                   checkpoint.stem + '.pth' + f'-seed={seed}')
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    print(f'saving results in {out_dir}')

    model, model_geco = restore_from_checkpoint(test, checkpoint, local_rank)
    model.eval()

    if test['eval_type'] == 'disentanglement_preprocessing':
        # creates a file `z_range.txt` storing min/max values for each latent dim
        _disentanglement_preprocessing(test, te_dataloader, model, model_geco, out_dir)
    elif test['eval_type'] == 'disentanglement_viz':
        # saves...
        _disentanglement_viz(test, te_dataloader, model, model_geco, out_dir)
    elif test['eval_type'] == 'activeness':
        # saves a numpy array of the average pixel variance for each latent dim
        _activeness(test, te_dataloader, model, model_geco, out_dir)
    elif test['eval_type'] == 'dci_clevr':
        _dci_clevr(test, te_dataloader, model, model_geco, out_dir)
    elif test['eval_type'] == 'ARI_MSE_KL':
        _ari_mse_kl(test, te_dataloader, model, model_geco, out_dir)
    elif test['eval_type'] == 'sample_viz':
        _sample_viz(test, te_dataloader, model, model_geco, out_dir)


@ex.automain
def run(_run, seed):
    do_eval(_run=_run, seed=seed)

#####################################################
### Metrics definitions
#####################################################
def _disentanglement_preprocessing(test, te_dataloader, model, model_geco, out_dir):
    total_images = 0
    min_maxes = []
    for i,batch in enumerate(tqdm(te_dataloader)):
        if total_images >= test['num_images_to_process']:
            break

        imgs = batch['imgs'].to('cuda')    
    
        if test['model'] == 'EfficientMORL':
            posterior = model.module.two_stage_inference(imgs, model_geco,
                                                 i, 1., get_posterior=True)
            sampled_z = posterior.sample()
        elif test['model'] == 'SlotAttention':
            sampled_z = model(imgs)['slots']
        # for all slots and all dims, compute the dim-wise max and min
        sampled_z = sampled_z.view(test['batch_size'], model.module.K,
                                   model.module.z_size)
        min_maxes += [sampled_z.detach()[0]]
        total_images += imgs.shape[0]

    min_maxes = torch.cat(min_maxes, 0)  # [100*K, z_size]
    maxes,_ = torch.max(min_maxes, 0)  # [z_size]
    mins,_ = torch.min(min_maxes,0)  # [z_size]
    with open(out_dir / 'z_range.txt', 'a+') as f:
        for i in range(model.module.z_size):
            f.write('{},{}\n'.format(mins[i].data.cpu().numpy(),
                                     maxes[i].data.cpu().numpy()))


def _disentanglement_viz(test, te_dataloader, model, model_geco, out_dir):
    total_images = 0
    for i,batch in enumerate(tqdm(te_dataloader)):
        if total_images >= test['num_images_to_process']:
            break

        imgs = batch['imgs'].to('cuda')
        slot_id = test['disentangle_slot']
        
        limit_file = open(out_dir / 'z_range.txt', 'r')
        limits = limit_file.readlines()
        limit_file.close()

        num_steps = 8
        if test['model'] == 'EfficientMORL':
            posterior = model.module.two_stage_inference(imgs, model_geco,
                                                    i, 1., get_posterior=True)
            sampled_z = posterior.sample()
        else:
            sampled_z = model(imgs)['slots']
        all_images = []
        with torch.no_grad():
            # for each latent dim
            for j in range(sampled_z.shape[-1]):
                new_z = sampled_z.clone()
                lower_lim = float(limits[j].split(',')[0])
                upper_lim = float(limits[j].split(',')[1])
                vals = torch.linspace(lower_lim, upper_lim, num_steps)
                dim_images = []
                new_z = new_z.view(test['batch_size'], model.module.K,
                                                       model.module.z_size)
                # for each interp step reconstruct the image
                for k in range(num_steps):
                    new_z[:,slot_id,j] = vals[k]
                    new_z_ = new_z.view(test['batch_size'] * model.module.K,
                                        model.module.z_size)
                    if test['model'] == 'EfficientMORL':
                        x_locs, mask_logits = model.module.hvae_networks.image_decoder(new_z_)
                    else:
                        x_locs, mask_logits = model.module.image_decoder(new_z_)
                    mask_logits = mask_logits.view(test['batch_size'],
                                       model.module.K, 1, test['output_size'][1],
                                       test['output_size'][2]
                                  )
                    mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
                    x_locs = x_locs.view(
                        test['batch_size'], model.module.K, test['output_size'][0],
                        test['output_size'][1], test['output_size'][2]
                    )
                    images = torch.sum(mask_logprobs.exp() * x_locs, dim=1)
                    dim_images += [images[0]]
                # Create a grid of latent dim x interp steps images
                dim_image_grid = torchvision.utils.make_grid(dim_images, nrow=11)
                masks = mask_logprobs.exp()[0]  # [k,1,h,w]
                mask_grid = torchvision.utils.make_grid(masks)
                all_images += [dim_image_grid]
        total_images += imgs.shape[0]

        all_images = torch.cat(all_images, 1)
        np.save(out_dir / f'slots_{slot_id}_disentanglement.npy',
                all_images.data.cpu().numpy())
        np.save(out_dir / f'ground_truth.npy', imgs.data.cpu().numpy())
        np.save(out_dir / f'masks.npy', mask_grid.data.cpu().numpy())

def _activeness(test, te_dataloader, model, model_geco, out_dir):
    total_images = 0
    all_var = []
    for i,batch in enumerate(tqdm(te_dataloader)):
        if total_images >= test['num_images_to_process']:
            break

        imgs = batch['imgs'].to('cuda')
                
        limit_file = open(out_dir / 'z_range.txt', 'r')
        limits = limit_file.readlines()
        limit_file.close()

        num_steps = 6
        if test['model'] == 'EfficientMORL':
            posterior = model.module.two_stage_inference(imgs, model_geco,
                                                         i, 1., get_posterior=True)
            sampled_z = posterior.sample()
        elif test['model'] == 'SlotAttention':
            sampled_z = model(imgs)['slots']
        
        all_images = []

        # randomly select a slot
        slot_id = np.random.randint(model.module.K)
        with torch.no_grad():
            var_vector = []
            # for each latent dim
            for j in range(sampled_z.shape[-1]):
                new_z = sampled_z.clone()
                lower_lim = float(limits[j].split(',')[0])
                upper_lim = float(limits[j].split(',')[1])
                vals = torch.linspace(lower_lim,upper_lim,num_steps)
                dim_images = []
                new_z = new_z.view(test['batch_size'], model.module.K, model.module.z_size)
                for k in range(num_steps):
                    new_z[:,slot_id,j] = vals[k]
                    new_z_ = new_z.view(test['batch_size'] * model.module.K,
                                        model.module.z_size)
                    if test['model'] == 'EfficientMORL':
                        x_locs, mask_logits = model.module.hvae_networks.image_decoder(new_z_)
                    else:
                        x_locs, mask_logits = model.module.image_decoder(new_z_)
                    mask_logits = mask_logits.view(test['batch_size'], model.module.K, 1,
                                                   test['output_size'][1], test['output_size'][2])
                    mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
                    x_locs = x_locs.view(test['batch_size'], model.module.K, test['output_size'][0],
                                         test['output_size'][1], test['output_size'][2])
                    images = torch.sum(mask_logprobs.exp() * x_locs, dim=1)
                    dim_images += [images[0]]
                dim_images = torch.stack(dim_images)
                pixel_mean_var = torch.mean(torch.var(dim_images, 0))
                var_vector += [pixel_mean_var]
            var_vector = torch.stack(var_vector)  # [z_size]
            all_var += [var_vector]
        total_images += imgs.shape[0]
    np.save(out_dir / 'activeness.npy', torch.stack(all_var).data.cpu().numpy())

def _dci_clevr(test, te_dataloader, model, model_geco, out_dir):
    total_images = 0
    all_var = []
    for i,batch in enumerate(tqdm(te_dataloader)):
        if total_images >= test['num_images_to_process']:
            break

        imgs = batch['imgs'].to('cuda')

        if test['model'] == 'EfficientMORL':
            posterior = model.module.two_stage_inference(imgs, model_geco,
                                                            i, 1., get_posterior=True)
            # Use the mean of the posterior distributions 
            q_mean = posterior.mean   # [N*K,D]
            x_locs, mask_logits = model.module.hvae_networks.image_decoder(q_mean)
            q_mean = q_mean.data.cpu().numpy()
            mask_logits = mask_logits.view(test['batch_size'], model.module.K, 1,
                                            test['output_size'][1],
                                            test['output_size'][2])
            mask_logprobs = nn.functional.log_softmax(mask_logits, dim=1)
            pred_masks = mask_logprobs.exp()
        elif test['model'] == 'SlotAttention':
            outs = model(imgs)
            q_mean = outs['slots'][0].data.cpu().numpy()
            pred_masks = outs['masks'].exp()
        
        resized_masks = []
        # Resizing masks to match ground truth mask size on CLEVR6
        if test['output_size'][1] == 96 or test['output_size'][1] == 192:
            for slot_idx in range(pred_masks.shape[1]):
                PIL_mask = Image.fromarray(
                         pred_masks[0,slot_idx,0].data.cpu().numpy(), mode="F")
                PIL_mask = PIL_mask.resize((192,192))
                resized_masks += [np.array(PIL_mask)[...,None]]
            resized_masks = np.stack(resized_masks)[None]  # [1,K,H,W,C]
            resized_masks = np.transpose(resized_masks, (0,1,4,2,3))
            pred_masks = torch.from_numpy(resized_masks).to(pred_masks.device)
        true_masks = batch['masks'].to(pred_masks.device)
        true_factors = batch['factors'].numpy()  # [N, max # objects, num factors]
        # Use visibility 
        object_presence = true_factors[0,:,-1]
        # Find the best matching slot ID for each ground truth object
        GT_idxs, pred_idxs = matching_iou(true_masks[0], pred_masks[0], object_presence)
        # Select the GT factors
        sel_factors = true_factors[0, GT_idxs]  # [selected, num_factors]
        # Select the codes
        q_mean = q_mean[pred_idxs]  # [selected, D]
        # Select GT masks
        sel_masks = true_masks.data.cpu().numpy()[0, GT_idxs, 0]
        # Select pred masks
        sel_pred_masks = pred_masks.data.cpu().numpy()[0, pred_idxs, 0]

        np.save(out_dir / f'mus_{i}.npy', q_mean)
        np.save(out_dir / f'ys_{i}.npy', sel_factors)
        np.save(out_dir / f'gt_masks_{i}.npy', sel_masks)  # for debugging the matches
        np.save(out_dir / f'pred_masks_{i}.npy', sel_pred_masks)  # for debugging the matches

        total_images += imgs.shape[0]

    codes_dataset = []
    factors_dataset = []
    
    for i in range(320):
        codes = np.load(out_dir / f'mus_{i}.npy')
        factors = np.load(out_dir / f'ys_{i}.npy')
        codes_dataset += [codes]
        factors_dataset += [factors]

    codes_dataset = np.concatenate(codes_dataset)
    factors_dataset = np.concatenate(factors_dataset)
    # remove presence
    factors_dataset = factors_dataset[:, :8]
    # x, y, z, rot, size, material, shape, color
    factor_types = ['continuous', 'continuous', 'continuous', 'continuous', 'discrete', 'discrete', 'discrete', 'discrete']

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(codes_dataset, factors_dataset, test_size=0.5, random_state=5, shuffle=True)

    print(X_train.shape, X_test.shape)

    scores = _compute_dci(X_train.T, y_train.T,  X_test.T, y_test.T, factor_types)
    
    print(scores)
    with open(out_dir / 'dci.txt', 'a+') as f:
        f.write(str(scores))
    
def _ari_mse_kl(test, te_dataloader, model, model_geco, out_dir):
    total_images = 0
    all_ARI, all_MSE, all_KL = [], [], []
    for i,batch in enumerate(tqdm(te_dataloader)):
        if total_images >= test['num_images_to_process']:
            break

        imgs = batch['imgs'].to('cuda')
        true_masks = batch['masks'].to('cuda')

        if test['model'] == 'EfficientMORL':
                outs = model(imgs, model_geco, i, 1)
        elif test['model'] == 'SlotAttention'
                outs = model(imgs)
        
        H = imgs.shape[2]
        W = imgs.shape[3]
        if test['model'] == 'EfficientMORL':
            pred_means = outs[f'means_{model.module.stochastic_layers-1+model.module.refinement_iters}']
            mask_logprobs = outs[f'masks_{model.module.stochastic_layers-1+model.module.refinement_iters}']
        else:
            pred_means = outs['means']
            mask_logprobs = outs['masks']
        pred_masks = mask_logprobs.exp()
        resized_masks = []
        pred_masks_ = pred_masks.data.cpu().numpy()
        all_pred_masks += [pred_masks_]

        # Hack for resizing on CLEVR6
        if test['output_size'][1] == 96 or test['output_size'][1] == 192:
            for slot_idx in range(pred_masks.shape[1]):
                PIL_mask = Image.fromarray(pred_masks_[0,slot_idx,0], mode="F")
                PIL_mask = PIL_mask.resize((192,192))
                resized_masks += [np.array(PIL_mask)[...,None]]
            resized_masks = np.stack(resized_masks)[None]  # [1,K,H,W,C]
            resized_masks = np.transpose(resized_masks, (0,1,4,2,3))
            pred_masks = torch.from_numpy(resized_masks).to(true_masks.device)
        
        # ARI
        ari = adjusted_rand_index(true_masks, pred_masks)
        ari = ari.data.cpu().numpy().reshape(-1)
        all_ARI += [ari]

        # MSE
        pred_img = (pred_means * pred_masks).sum(1)
        imgs = (imgs + 1) / 2.
        mse = torch.mean((imgs - pred_img) ** 2)
        mse = mse.data.cpu().numpy().reshape(-1)
        all_MSE += [mse]

        # top-level KL
        if test['model'] == 'EfficientMORL':
            kl = outs['kl'].data.cpu().numpy().reshape(-1)
            all_KL += [kl]                

        total_images += imgs.shape[0]

    print('mean ARI: {}, std dev: {}'.format(np.mean(all_ARI), np.std(all_ARI)))
     
    with open(out_dir / 'ARI.txt', 'a+') as f:
        f.write('{:.3f},{:.3f}\n'.format(np.mean(all_ARI), np.std(all_ARI)))

    print('mean MSE: {}, std dev: {}'.format(np.mean(all_MSE), np.std(all_MSE)))
    with open(out_dir / 'MSE.txt', 'a+') as f:
        f.write('{},{}\n'.format(np.mean(all_MSE), np.std(all_MSE)))

    if test['model'] == 'EfficientMORL':
        print('mean KL: {}, std dev: {}'.format(np.mean(all_KL), np.std(all_KL)))
        with open(out_dir / 'KL.txt', 'a+') as f:
            f.write('{},{}\n'.format(np.mean(all_KL), np.std(all_KL)))

def _sample_viz(test, te_dataloader, model, model_geco, out_dir):
    total_images = 0
    for i,batch in enumerate(tqdm(te_dataloader)):
        if total_images >= test['num_images_to_process']:
            break

        rinfo = {}
        outs = model(imgs, model_geco, i, 1, debug=True)
        rinfo["data"] = {}
        imgs = (imgs + 1) / 2
        rinfo["data"]["image"] = imgs.permute(0,2,3,1).data.cpu().numpy()
        rinfo["data"]["true_mask"] = batch["masks"].data.cpu().numpy()

        rinfo["outputs"] = {
            "recons": [],
            "pred_mask": [],
            "pred_mask_logits": [],
            "components": []
        }
        for t in range(model.module.stochastic_layers+model.module.refinement_iters):
            pred_mask_logits = outs[f"masks_{t}"]
            pred_mask = pred_mask_logits.exp()
            rgb = outs[f"means_{t}"]
            components = pred_mask * rgb + (1 - pred_mask) * torch.ones(pred_mask.shape).to(pred_mask.device)
            recons = torch.sum(pred_mask * rgb, 1)
            rinfo["outputs"]["pred_mask"] += [pred_mask]
            rinfo["outputs"]["pred_mask_logits"] += [pred_mask_logits]
            rinfo["outputs"]["recons"] += [recons]
            rinfo["outputs"]["components"] += [components]
        rinfo["outputs"]["pred_mask"] = torch.stack(rinfo["outputs"]["pred_mask"], 1).permute(0,1,2,4,5,3).data.cpu().numpy()
        rinfo["outputs"]["pred_mask_logits"] = torch.stack(rinfo["outputs"]["pred_mask_logits"], 1).permute(0,1,2,4,5,3).data.cpu().numpy()
        rinfo["outputs"]["recons"] = torch.stack(rinfo["outputs"]["recons"], 1).permute(0,1,3,4,2).data.cpu().numpy()
        rinfo["outputs"]["components"] = torch.stack(rinfo["outputs"]["components"], 1).permute(0,1,2,4,5,3).data.cpu().numpy()
        pkl.dump(rinfo, open(out_dir / f'rinfo_{i}.pkl', 'wb'))
        
        total_images += imgs.shape[0]