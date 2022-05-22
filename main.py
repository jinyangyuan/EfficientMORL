import torch
from torch.utils.tensorboard import SummaryWriter
from sacred import Experiment
import datetime
from tqdm import tqdm
from pathlib import Path
import os
import numpy as np
import h5py
from warmup_scheduler import GradualWarmupScheduler
from lib.datasets import ds
from dataset import CustomizedDataset
from lib.third_party.clevrtex_eval import CLEVRTEX, collate_fn, texds
from lib.model import net
from lib.model import EfficientMORL, SlotAttention
from lib.geco import GECO
from lib.visualization import visualize_output, visualize_slots
from sacred.observers import FileStorageObserver


ex = Experiment('TRAINING', ingredients=[ds,texds, net])

@ex.config
def cfg():
    data_path = '',
    num_tests = 5,
    training = {
            'batch_size': 16,  # training mini-batch size
            'num_workers': 8,  # pytorch dataloader workers
            'mode': 'train',  # dataset
            'model': 'EfficientMORL',  # model name
            'iters': 500000,  # gradient steps to take
            'refinement_curriculum': [(-1,3), (100000,1), (200000,1)],   # (step,I): Update refinement iters I at step 
            'lr': 3e-4,  # Adam LR
            'warmup': 10000,  # LR warmup
            'decay_rate': 0.5,  # LR decay
            'decay_steps': 100000,  # LR decay steps
            'kl_beta_init': 1,  # kl_beta from beta-VAE
            'use_scheduler': False,  # LR scheduler
            'tensorboard_freq': 100,  # how often to write to TB
            'checkpoint_freq': 25000,  # save checkpoints every % steps
            'load_from_checkpoint': False,  # whether to load from a checkpoint or not
            'checkpoint': '',  # name of .pth file to load model state
            'run_suffix': 'debug',  # string to append to run name
            'out_dir': 'experiments',  # where output folders for run results go
            'use_geco': True,  # Use GECO (Rezende & Viola 2018)
            'clip_grad_norm': True,  # Grad norm clipping to 5.0
            'geco_reconstruction_target': -23000,  # GECO C
            'geco_step_size_acceleration': 1,  # multiplies beta once the target is reached
            'geco_ema_alpha': 0.99,  # GECO EMA step parameter
            'geco_beta_stepsize': 1e-6,  # GECO Lagrange parameter beta
            'tqdm': False  # Show training progress in CLI
        }


def save_checkpoint(step, kl_beta, model, model_opt, filepath):
    state = {
        'step': step,
        'model': model.state_dict(),
        'model_opt': model_opt.state_dict(),
        'kl_beta': kl_beta
    }
    torch.save(state, filepath)


@ex.automain
def run(data_path, num_tests, training, seed, _run):
    num_tests = num_tests[0]

    run_dir = Path(training['out_dir'], 'runs')
    checkpoint_dir = Path(training['out_dir'], 'weights')
    tb_dir = Path(training['out_dir'], 'tb')
    
    # Avoid issues with torch distributed and just create directory structure 
    # beforehand
    # training['out_dir']/runs
    # training['out_dir']/weights
    # training['out_dir']/tb    
    for dir_ in [run_dir, checkpoint_dir, tb_dir]:
        if not dir_.exists():
            print(f'Create {dir_} before running!')
            exit(1)

    tb_dbg = tb_dir / (training['run_suffix'] + '_' + \
                       datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))

    writer = SummaryWriter(tb_dbg)
    
    # Auto-set by sacred
    # torch.manual_seed(seed)
#     torch.backends.cudnn.deterministic=True
#     torch.backends.cudnn.benchmark=False
    # Auto-set by sacred 
    #np.random.seed(seed)
        
    if training['model'] == 'EfficientMORL':
        model = EfficientMORL(batch_size=training['batch_size'],
                              use_geco=training['use_geco'])
    elif training['model'] == 'SlotAttention':
        model = SlotAttention(batch_size=training['batch_size'])
    else:
        raise RuntimeError('Model {} unknown'.format(training['model']))
    
    model_geco = None
    if training['use_geco']:
        C, H, W = model.input_size
        recon_target = training['geco_reconstruction_target'] * (C * H * W)
        model_geco = GECO(recon_target,
                          training['geco_ema_alpha'],
                          training['geco_step_size_acceleration'])        
    
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.train()

    # Optimization
    model_opt = torch.optim.Adam(model.parameters(), lr=training['lr'])
    if training['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                  model_opt,
                  lr_lambda=lambda epoch: 0.5 ** (epoch / 100000)
        )
        scheduler_warmup = GradualWarmupScheduler(model_opt, multiplier=1,
                                                  total_epoch=training['warmup'],
                                                  after_scheduler=scheduler)
    else:
        scheduler_warmup = None

    if not training['load_from_checkpoint']:    
        step = 0 
        kl_beta = training['kl_beta_init']
    else:
        checkpoint = checkpoint_dir / training['checkpoint']
        state = torch.load(checkpoint)
        model.load_state_dict(state['model'])
        model_opt.load_state_dict(state['model_opt'])
        kl_beta = state['kl_beta']
        step = state['step']

    batch_size = training['batch_size']
    data_loaders = {}
    with h5py.File(data_path, 'r') as f:
        for phase in f:
            data = f[phase]['image'][()]
            data_loaders[phase] = torch.utils.data.DataLoader(
                CustomizedDataset(data),
                batch_size=batch_size if phase == 'train' else batch_size // 2,
                num_workers=training['num_workers'],
                shuffle=(phase == 'train'),
                drop_last=(phase == 'train'),
                pin_memory=True,
            )
    
    max_iters = training['iters']
    
    print('Num parameters: {}'.format(sum(p.numel() for p in model.parameters())))

    epoch_idx = 0

    while step <= max_iters:

        for batch in data_loaders['train']:
            # Update refinement iterations by curriculum
            for rf in range(len(training['refinement_curriculum'])-1,-1,-1):
                if step >= training['refinement_curriculum'][rf][0]:
                    model.module.refinement_iters = training['refinement_curriculum'][rf][1]
                    break
            
            img_batch = batch['imgs'].cuda()
            model_opt.zero_grad()

            # Forward
            if training['model'] == 'SlotAttention':
                raise AssertionError
                out_dict = model(img_batch)
            else:
                out_dict = model(img_batch, model_geco, step, kl_beta)
            if out_dict['nll'].ndim != 0:
                out_dict['deltas'] = out_dict['deltas'].reshape([out_dict['nll'].shape[0], -1]).sum(0)
                for key in ['nll', 'kl', 'total_loss']:
                    out_dict[key] = out_dict[key].sum(0)

            # Backward
            total_loss = out_dict['total_loss']
            total_loss.backward()
            if training['use_scheduler']:
                scheduler_warmup.step(step)
            if training['clip_grad_norm']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.)
            model_opt.step()
            
            if training['use_geco']:
                if step == model.module.geco_warm_start:
                    model.module.geco_C_ema = model_geco.init_ema(model.module.geco_C_ema, out_dict['nll'])
                elif step > model.module.geco_warm_start:
                    model.module.geco_C_ema = model_geco.update_ema(model.module.geco_C_ema, out_dict['nll'])
                    model.module.geco_beta = model_geco.step_beta(model.module.geco_C_ema,
                            model.module.geco_beta, training['geco_beta_stepsize'])

            # logging
            if step % training['tensorboard_freq'] == 0:
                if training['model'] == 'SlotAttention':
                    writer.add_scalar('train/MSE', total_loss, step)
                    visualize_slots(writer, (img_batch+1)/2., out_dict, step)
                else:
                    writer.add_scalar('train/total_loss', total_loss, step)
                    writer.add_scalar('train/KL', out_dict['kl'], step)
                    writer.add_scalar('train/KL_beta', kl_beta, step)
                    writer.add_scalar('train/NLL', out_dict['nll'], step)
                    visualize_output(writer, (img_batch+1)/2., out_dict,
                                     model.module.stochastic_layers, model.module.refinement_iters,
                                     step)

                if training['use_geco']:
                    writer.add_scalar('train/geco_beta', model.module.geco_beta, step)
                    writer.add_scalar('train/geco_C_ema', model.module.geco_C_ema, step)

                if 'deltas' in out_dict:
                    for refine_iter in range(out_dict['deltas'].shape[0]):
                        writer.add_scalar(f'train/norm_delta_lamda_{refine_iter}',
                                          out_dict['deltas'][refine_iter], step)


            if step > 0 and step % training['checkpoint_freq'] == 0:
                # Save the model
                prefix = training['run_suffix']
                save_checkpoint(step, kl_beta, model, model_opt, 
                       checkpoint_dir / f'{prefix}-state-{step}.pth')
                        
            if step >= max_iters:
                step += 1
                break
            step += 1
        epoch_idx += 1

    for phase in ['test', 'general']:
        if phase == 'general':
            model.module.K = model.module.K_general
            model.module.hvae_networks.K = model.module.hvae_networks.K_general
            model.module.hvae_networks.indep_prior.K = model.module.hvae_networks.indep_prior.K_general
        else:
            model.module.K = model.module.K_test
            model.module.hvae_networks.K = model.module.hvae_networks.K_test
            model.module.hvae_networks.indep_prior.K = model.module.hvae_networks.indep_prior.K_test
        outputs = {key: [] for key in ['recon', 'apc', 'mask']}
        for data in data_loaders[phase]:
            img_batch = data['imgs'].cuda()
            sub_outputs = {key: [] for key in outputs}
            for _ in range(num_tests):
                out_dict = model(img_batch, model_geco, step, kl_beta)
                if out_dict['nll'].ndim != 0:
                    out_dict['deltas'] = out_dict['deltas'].reshape([out_dict['nll'].shape[0], -1]).sum(0)
                    for key in ['nll', 'kl', 'total_loss']:
                        out_dict[key] = out_dict[key].sum(0)
                total_loss = out_dict['total_loss']
                total_loss.backward()
                model_opt.zero_grad()
                apc = out_dict['means_final']
                mask = out_dict['masks_final'].exp()
                recon = (apc * mask).sum(1)
                sub_outputs['recon'].append(recon)
                sub_outputs['apc'].append(apc)
                sub_outputs['mask'].append(mask)
            sub_outputs = {key: (torch.clamp(torch.stack(val), 0, 1) * 255).to(torch.uint8).cpu().numpy()
                           for key, val in sub_outputs.items()}
            sub_outputs = {key: np.rollaxis(val, -3, len(val.shape)) for key, val in sub_outputs.items()}
            for key, val in sub_outputs.items():
                outputs[key].append(val)
        outputs = {key: np.concatenate(val, axis=1) for key, val in outputs.items()}
        with h5py.File(os.path.join(training['out_dir'], '{}.h5'.format(phase)), 'w') as f:
            for key, val in outputs.items():
                f.create_dataset(key, data=val, compression='gzip')

    writer.close()
