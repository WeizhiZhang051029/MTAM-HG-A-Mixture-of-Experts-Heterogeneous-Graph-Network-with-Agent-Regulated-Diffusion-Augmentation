import torch
from tabdiff.main import main as tabdiff_main
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training of TabDiff')

    # General configs
    parser.add_argument('--dataname', type=str, default='adult', help='Name dataset, one of those in data/ dir')
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no_wandb', action='store_true', help='disable wandb')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name, used to name log directories and the wandb run name')
    parser.add_argument('--deterministic', action='store_true', help='Whether to make the entire process deterministic, i.e., fix global random seeds')
    
    # Configs for tabdiff
    
    # Configs for testing tabdiff
    parser.add_argument('--num_samples_to_generate', type=int, default=None, help='Number of samples to be generated while testing')
    parser.add_argument('--ckpt_path', type=str, default=None, help='Path to the model checkpoint to be tested')
    
    # Mechanism-aware TabDiff fine-tuning and sampling guidance
    parser.add_argument('--mechanism_constraint', action='store_true', help='Enable CAPL mechanism loss/guidance.')
    parser.add_argument('--mechanism_lambda', type=float, default=0.0, help='CAPL mechanism loss weight.')
    parser.add_argument('--guidance_scale', type=float, default=0.0, help='CAPL mechanism sampling guidance strength.')
    parser.add_argument('--guidance_start_step', type=int, default=0)
    parser.add_argument('--guidance_stop_step', type=int, default=None)
    parser.add_argument('--trainable_scope', type=str, default='all', choices=['all', 'mlp', 'detokenizer', 'mlp_detokenizer'])
    parser.add_argument('--min_save_epoch', type=int, default=1)
    parser.add_argument('--reset_train_epoch', action='store_true', help='Load ckpt weights but start fine-tuning epoch count from zero.')
    parser.add_argument('--mechanism_low_quantile', type=float, default=0.01)
    parser.add_argument('--mechanism_high_quantile', type=float, default=0.99)
    parser.add_argument('--mechanism_window_margin', type=float, default=0.0)
    parser.add_argument('--mechanism_ridge_alpha', type=float, default=1.0e-3)
    parser.add_argument('--mechanism_temperature_hold_tolerance', type=float, default=10.0)
    parser.add_argument('--mechanism_yield_tolerance', type=float, default=0.0)
    parser.add_argument('--mechanism_temperature_weight', type=float, default=1.0)
    parser.add_argument('--mechanism_window_weight', type=float, default=1.0)
    parser.add_argument('--mechanism_yield_weight', type=float, default=1.0)
    parser.add_argument('--finetune_lr', type=float, default=None)
    parser.add_argument('--finetune_steps', type=int, default=None)
    parser.add_argument('--num_timesteps_override', type=int, default=None)
    parser.add_argument('--stochastic_sampler', type=str, default=None, choices=['true', 'false'])

    parser.set_defaults(y_only=False, non_learnable_schedule=False)
    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
    
    tabdiff_main(args)
