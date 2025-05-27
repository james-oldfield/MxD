#!/usr/bin/env python
"""
Transcoder training sample code

This sample script can be used to train a transcoder on a model of your choice.
This code, along with the transcoder training code more generally, was largely
    adapted from an older version of Joseph Bloom's SAE training repo, the latest
    version of which can be found at https://github.com/jbloomAus/SAELens.
Most of the parameters given here are the same as the SAE training parameters
    listed at https://jbloomaus.github.io/SAELens/training_saes/.
Transcoder-specific parameters are marked as such in comments.
"""

import argparse
import os
import sys
import numpy as np
import torch

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sae_training.config import LanguageModelSAERunnerConfig
from sae_training.utils import LMSparseAutoencoderSessionloader
from sae_training.train_sae_on_language_model import train_sae_on_language_model


# compute the MoD hidden layer size to parametermatch
def compute_expert_count(multiplier):
    # Dimensions for the transcoder
    d_in = 768
    d_out = 768
    d_sae = d_in * multiplier

    # Total parameters in the transcoder
    total_transcoder = (d_sae * d_out) + (d_in * d_sae) + d_sae + d_out

    # For the MoE model, the weight matrices use a fixed expansion factor of 4.
    moe_expansion_factor = 4
    d_sae_moe = d_in * moe_expansion_factor

    # Parameters that do NOT depend on the number of experts
    base_moe_params = (d_sae_moe * d_out) + (d_in * d_sae_moe) + d_sae + d_out

    # Parameters that scale with the number of experts:
    # - gate:      d_in parameters per expert
    # - factor:    d_out parameters per expert
    # - gate_bias: 1 parameter per expert
    gating_params_per_expert = d_in + d_out + 1

    # Solve for the expert_count needed to match parameter counts:
    expert_count = (total_transcoder - base_moe_params) / gating_params_per_expert

    # Round to the nearest integer to get as close as possible to a match.
    return int(round(expert_count))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcoder training sample code with configurable options."
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=0.0004,
                        help="Learning rate")
    parser.add_argument("--l1_coeff", type=float, default=0.00001,
                        help="L1 sparsity regularization coefficient")
    parser.add_argument("--total_training_steps", type=int, default=10000,
                        help="Total training steps (unused when total tokens are set directly)")
    parser.add_argument("--batch_size", type=int, default=4096,
                        help="Train batch size")

    # Total training tokens can be computed as a base multiplied by a factor.
    parser.add_argument("--base_training_tokens", type=int, default=60_000_000,
                        help="Base number of training tokens (e.g. 1,000,000 * 60)")
    parser.add_argument("--training_length_multiplier", type=int, default=1,
                        help="Multiplier for the base training tokens")

    # Transformer / model parameters
    parser.add_argument("--layer", type=int, default=8,
                        help="Transformer layer to hook (e.g., 8)")
    parser.add_argument("--d_in", type=int, default=768,
                        help="Input dimension")
    parser.add_argument("--d_out", type=int, default=768,
                        help="Output dimension")
    parser.add_argument("--dataset_path", type=str, default="Skylion007/openwebtext",
                        help="Path or identifier for the dataset")
    parser.add_argument("--is_dataset_tokenized", action="store_true",
                        help="Flag indicating if the dataset is already tokenized")
    parser.add_argument("--model_name", type=str, default="gpt2-small",
                        help="Name of the language model to load")
    parser.add_argument("--b_dec_init_method", type=str, default="zeros",
                        help="Decoder initialization method")
    parser.add_argument("--use_kaiming_init", action="store_true",
                        help="Use default Kaiming initialization?")
    parser.add_argument("--use_transpose_decoder_init", action="store_true",
                        help="Use decoder transpose init?")
    parser.add_argument("--use_glu", action="store_true",
                        help="Use gated linear unit-style transcoder?")
    parser.add_argument("--use_skip", action="store_true",
                        help="Use transcoder skip connection?")

    # LR schedule and warmup
    parser.add_argument("--lr_scheduler_name", type=str, default="constantwithwarmup",
                        help="Name of the learning rate scheduler")
    parser.add_argument("--lr_warm_up_steps", type=int, default=5000,
                        help="Number of warm-up steps for the learning rate")

    # Activation storage parameters
    parser.add_argument("--n_batches_in_buffer", type=int, default=128,
                        help="Number of batches to keep in the activation buffer")
    parser.add_argument("--store_batch_size", type=int, default=32,
                        help="Batch size used for storing activations")
    parser.add_argument("--context_size", type=int, default=128,
                        help="Context size for activation storage")

    # Dead neurons and sparsity parameters
    parser.add_argument("--feature_sampling_window", type=int, default=1000,
                        help="Window size for feature sampling")
    parser.add_argument("--resample_batches", type=int, default=1028,
                        help="Number of batches between feature resampling")
    parser.add_argument("--dead_feature_window", type=int, default=5000,
                        help="Window for dead feature checking")
    parser.add_argument("--dead_feature_threshold", type=float, default=1e-8,
                        help="Threshold for considering a feature 'dead'")
    parser.add_argument("--topk", type=int, default=-1,
                        help="Top-k features to select (-1 for none; then uses l1 penalty)")
    # By default, norm_dec is True; use --no-norm_dec to disable.
    parser.add_argument("--no_norm_dec", dest="norm_dec", action="store_false",
                        help="Disable normalization in the decoder")
    parser.set_defaults(norm_dec=True)
    parser.add_argument("--topk_aux_loss", action="store_true", help="Use aux loss?")

    # Custom run name
    parser.add_argument("--custom_name", type=str, default="",
                        help="Custom name for the training run")
    parser.add_argument("--mlp_activation_fn", type=str, default="relu", help="which activation to use in hidden layer of MLP? E.g. transcoders use 'relu'")

    # Transcoder-specific parameters
    parser.add_argument("--expert_multiplier", type=int, default=32,
                        help="Multiplier used in computing number of experts (i.e. numnber of transcoder features in the SAE)")
    parser.add_argument("--expansion_factor", type=int, default=4,
                        help="Expansion factor for the MLP layer")

    parser.add_argument("--use_conditional", action="store_true", help="Use conditional")
    parser.add_argument("--use_sae", action="store_true", help="train an SAE instead")

    # use_ghost_grads is False by default; provide flag to enable.
    parser.add_argument("--use_ghost_grads", action="store_true", default=False,
                        help="Enable ghost gradients")

    # WANDB logging
    # log_to_wandb is True by default; use --no_log_to_wandb to disable.
    parser.add_argument("--no_log_to_wandb", dest="log_to_wandb", action="store_false",
                        help="Disable logging to WANDB")
    parser.set_defaults(log_to_wandb=True)
    parser.add_argument("--wandb_log_frequency", type=int, default=50,
                        help="Frequency (in steps) to log to WANDB")

    # Miscellaneous options
    # use_tqdm is True by default; use --no_use_tqdm to disable.
    parser.add_argument("--no_use_tqdm", dest="use_tqdm", action="store_false",
                        help="Disable tqdm progress bars")
    parser.set_defaults(use_tqdm=True)
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--n_checkpoints", type=int, default=0,
                        help="Number of checkpoints to save during training")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/gpt2-small-transcoders",
                        help="Path to save checkpoints")
    # For dtype, allow a string choice and then map it to the actual torch dtype.
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64", "float16"],
                        help="Data type to use (will be mapped to torch dtype)")
    # Feature reinitialization scale (if applicable)
    parser.add_argument("--feature_reinit_scale", type=float, default=1.0,
                        help="Scale for feature reinitialization")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Compute total training tokens as base * multiplier.
    total_training_tokens = args.base_training_tokens * args.training_length_multiplier

    # Map the dtype string to the corresponding torch dtype.
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "float64":
        dtype = torch.float64
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

    # Compute the number of experts based on the provided multiplier.
    n_experts = compute_expert_count(args.expert_multiplier) if args.use_conditional else 0

    # Create the configuration object.
    cfg = LanguageModelSAERunnerConfig(
        # Data generating function settings
        hook_point_layer=args.layer,
        hook_point=f"blocks.{args.layer}.ln2.hook_normalized" if not args.use_sae else f"blocks.{args.layer}.hook_mlp_out",
        d_in=args.d_in,
        dataset_path=args.dataset_path,
        is_dataset_tokenized=args.is_dataset_tokenized,
        model_name=args.model_name,

        # Transcoder-specific settings
        is_transcoder=(not args.use_sae),
        out_hook_point=f"blocks.{args.layer}.hook_mlp_out",
        out_hook_point_layer=args.layer,
        d_out=args.d_out,

        # SAE Parameters
        b_dec_init_method=args.b_dec_init_method,

        # Training parameters
        lr=args.lr,
        l1_coefficient=args.l1_coeff,
        lr_scheduler_name=args.lr_scheduler_name,
        train_batch_size=args.batch_size,
        lr_warm_up_steps=args.lr_warm_up_steps,

        # Activation store parameters
        total_training_tokens=total_training_tokens,
        n_batches_in_buffer=args.n_batches_in_buffer,
        store_batch_size=args.store_batch_size,
        context_size=args.context_size,

        # Dead Neurons and Sparsity
        feature_sampling_method=None,
        feature_sampling_window=args.feature_sampling_window,
        resample_batches=args.resample_batches,
        dead_feature_window=args.dead_feature_window,
        dead_feature_threshold=args.dead_feature_threshold,
        
        use_kaiming_init=args.use_kaiming_init,
        use_transpose_decoder_init=args.use_transpose_decoder_init,
        use_glu=args.use_glu,
        use_skip=args.use_skip,

        topk=args.topk,
        norm_dec=args.norm_dec,
        topk_aux_loss=args.topk_aux_loss,

        # Custom name for the run
        custom_name=args.custom_name,
        mlp_activation_fn=args.mlp_activation_fn,

        # Transcoder defaults
        n_experts=n_experts,
        use_conditional=args.use_conditional,
        use_ghost_grads=args.use_ghost_grads,
        expansion_factor=args.expansion_factor,

        # WANDB Logging
        log_to_wandb=args.log_to_wandb,

        # Miscellaneous
        use_tqdm=args.use_tqdm,
        device=args.device,
        seed=args.seed,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=args.checkpoint_path,
        dtype=dtype,
        
        # Additional parameters referenced during training:
        feature_reinit_scale=args.feature_reinit_scale,
        wandb_log_frequency=args.wandb_log_frequency
    )

    print(f"About to start training with lr {args.lr} and l1 coefficient {args.l1_coeff}")
    print(f"Checkpoint path: {cfg.checkpoint_path}")
    print(cfg)

    # Load the session
    loader = LMSparseAutoencoderSessionloader(cfg)
    model, sparse_autoencoder, activations_loader = loader.load_session()

    # Train the SAE
    sparse_autoencoder = train_sae_on_language_model(
        model, sparse_autoencoder, activations_loader,
        n_checkpoints=cfg.n_checkpoints,
        batch_size=cfg.train_batch_size,
        feature_sampling_method=cfg.feature_sampling_method,
        feature_sampling_window=cfg.feature_sampling_window,
        feature_reinit_scale=cfg.feature_reinit_scale,
        dead_feature_threshold=cfg.dead_feature_threshold,
        dead_feature_window=cfg.dead_feature_window,
        use_wandb=cfg.log_to_wandb,
        wandb_log_frequency=cfg.wandb_log_frequency
    )

    # Save the trained SAE model
    save_path = f"{cfg.checkpoint_path}/{sparse_autoencoder.get_name()}.pt"
    sparse_autoencoder.save_model(save_path)
    print(f"Model saved to {save_path}")


if __name__ == '__main__':
    main()