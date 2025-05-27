import os
import sys

from transcoder_circuits.circuit_analysis import *
from transcoder_circuits.feature_dashboards import *
from transcoder_circuits.replacement_ctx import *

from sae_training.sparse_autoencoder import SparseAutoencoder
from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer
from natsort import natsorted
from datasets import load_dataset
from datasets import Dataset
from huggingface_hub import HfApi
from utils import tokenize_and_concatenate
from tqdm import tqdm
from sklearn.metrics import * 
import itertools
from sklearn.linear_model import LogisticRegression
import pickle
import seaborn as sns

import dataset_utils

import warnings
from transformers import logging
warnings.filterwarnings('ignore')
logging.set_verbosity_error()


transcoder_template = "final_mod_gpt2-small_blocks.8.ln2.hook_normalized-saewidth_3072-conditional_True-nexperts_21490-activfn_gelu-glu_False-topk_32-lr_0.0001_totaltokens_480000000"; layer_type='MxD'
base_model = 'gpt2'

transcoder = SparseAutoencoder.load_from_pretrained(f"{transcoder_template}.pt").eval()
    
model = HookedTransformer.from_pretrained(base_model)

tokenizer = AutoTokenizer.from_pretrained(model.tokenizer.name_or_path)

############################# dataset choice
# dataset_name = "fancyzhx/ag_news"
# dataset_name = "codeparrot/github-code"
# dataset_name = "canrager/amazon_reviews_mcauley_1and5_sentiment"
dataset_name = "Helsinki-NLP/europarl"

tokenizer.pad_token = tokenizer.eos_token
data = dataset_utils.get_multi_label_train_test_data(dataset_name, train_set_size=10000, test_set_size=0, random_seed=42)

classes = [x for x in data[0].keys()]

print('training example from each class')
for ni, n in enumerate(classes):
    print(f'class {ni}: ', data[0][f'{str(n)}'][0])

# collapse dataset into a big list of dicts, with "text" and "class" keys; like the gurnee format
dataset = []
for cls in data[0].keys():
    for x in data[0][cls]:
        dataset.append({'text': x, 'class': cls})
print('new dataset length: ', len(dataset))

# shuffle dataset
np.random.seed(42)
np.random.shuffle(dataset)

feature_acts = []
labels = []
names = []
max_samples = 10_000

prepend_bos = True
max_seq_length = 128

if prepend_bos:
    max_seq_length = max_seq_length - 1 # always save a space for first token, if we're using BOS.
    
safety_buffer = 2 if 'code' in dataset_name else 0 # hard-remove extra tokens from the start; when we split "code" strings, sometimes we get too many tokens. this is a bit of a hack.

for i in tqdm(range(min(len(dataset), max_samples))):
    with torch.no_grad():
        # note that gurnee arrow format is e.g. dataset['text'][i], here it's dataset[i]['text']
        pre_eot_string = dataset[i]['text']
        pre_eot_tokens = tokenizer.encode(pre_eot_string, add_special_tokens=False)
        
        # skip strings with fewer than 5 tokens
        if len(pre_eot_tokens) < 5:
            continue

        ########## 2.5 handle the case when we have more than max_seq_length tokens from the dataset's ['text'].
        if len(pre_eot_tokens) > max_seq_length:
            if dataset_name == "codeparrot/github-code":
                # # slice the token IDs to keep only the LAST ctx_length tokens
                pre_eot_tokens = pre_eot_tokens[-(max_seq_length-safety_buffer):]
            else:
                # slice the token IDs to keep only the FIRST ctx_length tokens
                pre_eot_tokens = pre_eot_tokens[:(max_seq_length-safety_buffer)]

            # convert the clipped tokens back into a string
            # pre_eot_string = tokenizer.decode(pre_eot_tokens, clean_up_tokenization_spaces=True)
            pre_eot_string = tokenizer.decode(pre_eot_tokens)

            # re-encode to ensure we have exactly the same tokenization of the new string
            pre_eot_tokens = tokenizer.encode(pre_eot_string, add_special_tokens=False)

        length_content_tokens = len(pre_eot_tokens) # cache how many tokens actually correspond to non-padding text

        #######
        ### 5. re-pad the new string with EOT up to 128 context length
        #######
        num_eot_to_add = max_seq_length - len(pre_eot_tokens)
        post_eot_string = pre_eot_string + "<|endoftext|>"*num_eot_to_add
        post_eot_tokens = tokenizer.encode(post_eot_string, add_special_tokens=False) # can use this to check that the tokenization post-eot appending is as expected.
        if len(post_eot_tokens) != max_seq_length:
            print(f"Encoding error: post_eot_string length is not {max_seq_length} tokens, it's {len(post_eot_tokens)}")
            continue

        logit_cache = model.run_with_cache(post_eot_string, prepend_bos=prepend_bos)
        
        # if we're using a BOS token tho, now the features are going to be *offset* by 1 (bc we append a BOS token to the start)
        offset = 1 if prepend_bos else 0
        
        acts = transcoder(logit_cache[1][transcoder.cfg.hook_point])[1].detach().cpu().numpy()

        # actually append the activations that span the tokens.
        target_acts = acts[:, offset:length_content_tokens]
        
        # mean pool the relevant tokens' activations (SAEbench)
        target_acts = np.mean(target_acts, axis=1, keepdims=True)
        labels.extend([dataset[i]["class"]])
        #######################
        
        feature_acts.extend(target_acts)

        del logit_cache
        del acts
        del target_acts

feature_acts_concat = np.concatenate(feature_acts, axis=0)
feature_acts_concat = feature_acts_concat[None, ...]

all_labels = list(itertools.chain.from_iterable([labels])) # note: [labels] is wrapped in array literal
classes = []
for x in dataset:
    if x["class"] not in classes:
        classes.append(x["class"])
classes = natsorted(classes)
num_classes = len(classes)
print(num_classes)
print(classes)

def downsample_perf_curves(curve, pts_to_keep=100):
    n = len(curve)
    if n <= pts_to_keep:
        return curve
    else:
        idx = np.round(np.linspace(0, n - 1, pts_to_keep)).astype(int)
        return curve[idx]

def get_binary_cls_perf_metrics(y_test, y_pred, y_score):
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    fowlkes_mallows_index = (precision_score(
        y_test, y_pred) * recall_score(y_test, y_pred))**0.5
    classifier_results = {
        'test_mcc': matthews_corrcoef(y_test, y_pred),
        'test_cohen_kappa': cohen_kappa_score(y_test, y_pred),
        'test_fmi': fowlkes_mallows_index,
        'test_f1_score': f1_score(y_test, y_pred),
        'test_f0.5_score': fbeta_score(y_test, y_pred, beta=0.5),
        'test_f2_score': fbeta_score(y_test, y_pred, beta=2),
        'test_pr_auc': auc(recall, precision),
        'test_acc': accuracy_score(y_test, y_pred),
        'test_balanced_acc': balanced_accuracy_score(y_test, y_pred),
        'test_precision': precision_score(y_test, y_pred),
        'test_recall': recall_score(y_test, y_pred),
        'test_average_precision': average_precision_score(y_test, y_pred),
        'test_roc_auc': roc_auc_score(y_test, y_score),
        'test_precision_curve': downsample_perf_curves(precision),
        'test_recall_curve': downsample_perf_curves(recall),
    }
    return classifier_results

lr = LogisticRegression(
    class_weight='balanced',
    penalty='l2',
    solver='newton-cholesky',
    n_jobs=-1,
    max_iter=200
)

occupation_results = {}
num_top = 100 ; print(f'WARN: considering {num_top} feature/expert!')

for occupation in classes:
    print("#### OCCUPATION : " + occupation)
    occupation_labels = np.array([1 if x==occupation else 0 for x in all_labels])

    train_split = 0.8
    split_index = int(len(all_labels)*train_split)
    X_train = feature_acts_concat[0][:split_index]
    X_test = feature_acts_concat[0][split_index:]
    y_train = np.array(occupation_labels[:split_index])
    y_test = np.array(occupation_labels[split_index:])

    pos_class_mean = np.mean(X_train[np.where(y_train == 1)], axis=0)
    neg_class_mean = np.mean(X_train[np.where(y_train == 0)], axis=0)
    class_means_diff = pos_class_mean - neg_class_mean
    top100_ind = np.argpartition(class_means_diff, -num_top)[-num_top:]

    feature_results = {}
    for i in tqdm(top100_ind):
        lr = lr.fit(X_train[:, i:i+1], y_train)
        lr_score = lr.decision_function(X_test[:, i:i+1])
        lr_pred = lr.predict(X_test[:, i:i+1])
        results = get_binary_cls_perf_metrics(y_test, lr_pred, lr_score)
        results['coef'] = lr.coef_[0]
        feature_results[i] = results 

    f1s = [feature_results[i]['test_f1_score'] for i in feature_results]
    accs = [feature_results[i]['test_balanced_acc'] for i in feature_results]
    max_index = np.argmax(f1s)
    expert_index = list(feature_results.keys())[max_index]
    print(f1s[max_index], accs[max_index])
    print(expert_index)
    expert_scores_pos = X_test[np.where(y_test == 1)][:, expert_index]
    expert_scores_neg = X_test[np.where(y_test == 0)][:, expert_index]
    occupation_results[occupation] = (f1s[max_index], accs[max_index], expert_index, expert_scores_pos, expert_scores_neg)

dataset_name = dataset_name.split('/')[1]
out_path = f'feature-eval/results/{dataset_name}-{layer_type}-{base_model}.json'

with open(out_path, "wb") as f:
    pickle.dump(occupation_results, f, protocol=pickle.HIGHEST_PROTOCOL)

fig, axs = plt.subplots(2, 3, figsize=(16, 10))
sns.set_style('whitegrid')

for i, occupation in enumerate(classes):
    f1, acc, index, expert_scores_pos, expert_scores_neg = occupation_results[occupation]    
    sns.kdeplot(expert_scores_pos, bw_method=0.1, color='#55cc55', ax=axs[i//3][i%3])
    sns.kdeplot(expert_scores_neg, bw_method=0.1, color='#cc5555', ax=axs[i//3][i%3])

    pos_density = axs[i//3][i%3].hist(expert_scores_pos, density=True, bins=30, color='#338833')
    neg_density = axs[i//3][i%3].hist(expert_scores_neg, density=True, bins=30, color='#883333')
    axs[i//3][i%3].set_title(occupation)
plt.tight_layout()
plt.savefig(f'/results/density-{dataset_name}-{layer_type}-{base_model}.pdf')
plt.show()