import argparse
import glob
import json
import re
import numpy as np
from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *
import re
from datasets import load_dataset
from huggingface_hub import HfApi

from utils import tokenize_and_concatenate
import tqdm
from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
import os
import torch
import glob
import json

def eval_transcoder_l0_ce(model, all_tokens, transcoder, num_batches=200, batch_size=128):
    l0s = []
    transcoder_losses = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(range(0, num_batches)):
            torch.cuda.empty_cache()
            cur_batch_tokens = all_tokens[batch*batch_size:(batch+1)*batch_size]
            with TranscoderReplacementContext(model, [transcoder]):
                cur_losses, cache = model.run_with_cache(cur_batch_tokens, return_type="loss", names_filter=[transcoder.cfg.hook_point])
                # measure losses
                transcoder_losses.append(utils.to_numpy(cur_losses))
                # measure l0s
                acts = cache[transcoder.cfg.hook_point]
                binarized_transcoder_acts = 1.0*(transcoder(acts)[1] > 0)
                l0s.append(
                    (binarized_transcoder_acts.reshape(-1, binarized_transcoder_acts.shape[-1])).sum(dim=1).mean().item()
                )

    return {
        'l0s': np.mean(l0s).item(),
        'lambda': transcoder.cfg.l1_coefficient,
        'ce_loss': np.mean(transcoder_losses).item(),
    }

def eval_base_and_zero_ce(model, transcoders, all_tokens, num_batches=200, batch_size=128):
    l0s = []
    losses = []
    zero_ablated_losses = []

    layers = [t.cfg.hook_point_layer for t in transcoders]
    print('zero-ablation layer:', layers)
    
    with torch.no_grad():
        for batch in tqdm.tqdm(range(0, num_batches)):
            torch.cuda.empty_cache()
            cur_batch_tokens = all_tokens[batch*batch_size:(batch+1)*batch_size]
            # cur_losses, cache = model.run_with_cache(cur_batch_tokens, return_type="loss")
            # # measure losses
            # losses.append(utils.to_numpy(cur_losses))
            losses.append(utils.to_numpy(model(cur_batch_tokens, return_type="loss")))

            with ZeroAblationContext(model, layers):
                zero_ablated_losses.append(utils.to_numpy(model(cur_batch_tokens, return_type="loss")))

    return {
        'ce_loss': np.mean(losses).item(),
        'zero_ablation_loss': np.mean(zero_ablated_losses).item(),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance with custom parameters")
    parser.add_argument("--model_type", type=str, required=True, help="Type of the model to evaluate", default="gpt2")
    parser.add_argument("--custom_path", type=str, required=True, help="Custom path for the pretrained model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Dataset path")
    parser.add_argument("--num_batches", type=int, required=False, help="Number of batches to do eval over. Default is 200", default=200)
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size", default=128)
    args = parser.parse_args()

    model_type = args.model_type
    custom_path = args.custom_path
    num_batches = args.num_batches
    batch_size = args.batch_size

    with torch.no_grad():
        model = HookedTransformer.from_pretrained(model_type)

        dataset = load_dataset(args.dataset_path, split='train', streaming=True)
        dataset = dataset.shuffle(seed=42, buffer_size=10_000)
        tokenized_owt = tokenize_and_concatenate(dataset, model.tokenizer, max_length=128, streaming=True)
        tokenized_owt = tokenized_owt.shuffle(42)
        tokenized_owt = tokenized_owt.take(12800*2)
        owt_tokens = np.stack([x['tokens'] for x in tokenized_owt])
        owt_tokens_torch = torch.from_numpy(owt_tokens).cuda()

        # run_type = custom_path.split('_')[1]
        # no_suffix = re.sub(r'\.pt$', '', custom_path)
        # run_type = re.sub(r'-l1_.*?(?=-lr_)', '', no_suffix)

        run_type = custom_path.split('/')[-1].split('.pt')[0]

        paths = glob.glob(f"{custom_path}")
        print('paths:')
        print(paths)
        print(f'evaluating {len(paths)} models...')
        print('---')

        results = []
        for path in paths:
            sae = SparseAutoencoder.load_from_pretrained(path).eval()
            result = eval_transcoder_l0_ce(model, owt_tokens_torch, sae, num_batches=num_batches, batch_size=batch_size)
            results.append(result)
            print(result, path)
            
        with open(f"evals/{run_type}_eval_numbatches{num_batches}.json", "w") as f:
            json.dump(results, f)
            print("...saved")

        print("now performing base model eval:")
        base_results = eval_base_and_zero_ce(model, [sae], owt_tokens_torch, num_batches=num_batches, batch_size=batch_size)
        
        base_model_name = custom_path.split('_')[2]
        
        print(base_results)
        with open(f"evals/base_{base_model_name}_eval_numbatches{num_batches}.json", "w") as f:
            json.dump(base_results, f)
            print("...saved")