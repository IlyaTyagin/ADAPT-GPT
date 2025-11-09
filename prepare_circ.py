import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import networkx as nx
import numpy as np
from collections import Counter
import random
import argparse
from joblib import Parallel, delayed
from FEATHER.src.feather import FEATHERG
from itertools import islice

tqdm.pandas()

# rounding_digits = 2
# val_frac = 0.1
# approx_ratio_thr = 0.98
# max_abs_param_val = 10

parser = argparse.ArgumentParser(description='Parser for ADAPT GPT circuit preparation.')
parser.add_argument('--adapt_results_dir', type=str, help='Path to read results from', required=True)
parser.add_argument('--debug_limit', default=0, type=int, help='Number of input files to sample for speed up (debugging)')
parser.add_argument('--save_dir', type=str, help='Path to save files', required=True)
parser.add_argument('--n_nodes', type=int, help='Number of nodes in the dataset', required=True)
parser.add_argument('--rounding_digits', type=int, default=2, help='Number of digits to round to')
parser.add_argument('--min_block_size', type=int, default=128, help='min sequence length in sliding window')
parser.add_argument('--max_block_size', type=int, default=256, help='nanoGPT block size')
parser.add_argument('--val_frac', type=float, default=0.1, help='Validation fraction')
parser.add_argument('--test_frac', type=float, default=0.1, help='Test fraction')
parser.add_argument('--approx_ratio_thr', type=float, default=0.97, help='Approximation ratio threshold')
parser.add_argument('--max_abs_param_val', type=float, default=10, help='Maximum absolute value of gamma and beta params')
parser.add_argument('--perform_coef_mod_range', type=int, default=True, help='Wrap beta to [0; pi] range; 1 is true (default), 0 is false')
parser.add_argument('--apply_sliding_window', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Apply sliding window to generate training samples')
parser.add_argument('--apply_feather_graph', type=bool, default=True, action=argparse.BooleanOptionalAction, help='Apply feather graph to generate graph embeddings')
parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to use to process ADAPT results')
parser.add_argument('--skip_only_qaoa_circ', type=bool, default=False, action=argparse.BooleanOptionalAction, help='Exclude circuits with only QAOA mixer present')
parser.add_argument('--allowed_graph_generators', type=str, default="all", help='Allowed graph generators. Default: all. Should be separated with ;. Allowed values: erdos_renyi;barabasi_albert;watts_strogatz;random_regular;bipartite')

# Parse the arguments
args = parser.parse_args()

print("Preparing ADAPT circuits for GPT training with the following arguments:")
for arg, value in vars(args).items():
    print(f"\t{arg}: {value}")

results_fpath_str = args.adapt_results_dir
save_path_str = args.save_dir
rounding_digits = args.rounding_digits
val_frac = args.val_frac
test_frac = args.test_frac
approx_ratio_thr = args.approx_ratio_thr
max_abs_param_val = args.max_abs_param_val
perform_coef_mod_range = bool(args.perform_coef_mod_range)
debug_limit = args.debug_limit
min_block_size = args.min_block_size
max_block_size = args.max_block_size
apply_sliding_window = args.apply_sliding_window
apply_feather_graph = args.apply_feather_graph
n_workers = args.n_workers
n_nodes = args.n_nodes
skip_only_qaoa_circ = args.skip_only_qaoa_circ
allowed_graph_generators_list = args.allowed_graph_generators.split(';')

if debug_limit:
    print(f'For debugging purposes, limit input results to {debug_limit} files.')

results_fpath_list = [Path(el) for el in results_fpath_str.split(';')]
save_path = Path(save_path_str)

for results_fpath in results_fpath_list:
    assert results_fpath.exists() and results_fpath.is_dir(), "Results path is invalid."

# Opening all files

## ADAPT results
results_folders_list = results_fpath_list
print("Reading ADAPT.jl results from:")
for res_fpath in results_folders_list:
    print('\t', str(res_fpath.absolute()))
    
df_list = []
df_list_all = []

def open_df_from_res_csv(fname):
    #df_list = []
    try:
        cur_df = pd.read_csv(fname)
        cur_df['worker_id'] = fname.stem
        #df_list.append(cur_df)
    except Exception as e:
        print(f'{e} (file: {fname})')
        cur_df = None
    return cur_df
 
for cur_dataset_res_path in results_folders_list:
    cur_dataset_res_flist = sorted(cur_dataset_res_path.joinpath('res').glob('*.csv'))
    if debug_limit:
        cur_dataset_res_flist = cur_dataset_res_flist[:debug_limit]
    # for fname in tqdm(cur_dataset_res_flist, desc='Opening ADAPT results'):
    #     try:
    #         cur_df = pd.read_csv(fname)
    #         cur_df['worker_id'] = fname.stem
    #         df_list.append(cur_df)
    #     except Exception as e:
    #         print(f'{e} (file: {fname})')
    df_list = Parallel(n_jobs=n_workers)(
        delayed(open_df_from_res_csv)(fname) for fname in tqdm(cur_dataset_res_flist, desc=f'Opening ADAPT results ({cur_dataset_res_path.stem})')
    )
    df_list_all += df_list
df_list  = [df for df in df_list_all if df is not None]
print("df_list len:", len(df_list))

full_run_df = pd.concat(df_list)
full_run_df['prefix'] = full_run_df['worker_id'].apply(
    lambda x: x[:-15]
)

## Graphs
df_list = []
df_list_all = []
for cur_dataset_res_path in results_folders_list:
    cur_dataset_res_flist = sorted(cur_dataset_res_path.joinpath('graphs').glob('*.csv'))
    if debug_limit:
        cur_dataset_res_flist = cur_dataset_res_flist[:debug_limit]
    # for fname in tqdm(cur_dataset_res_flist, desc='Opening graphs'):
    #     cur_df = pd.read_csv(fname)
    #     cur_df['worker_id'] = fname.stem
    #     df_list.append(cur_df)
    df_list = Parallel(n_jobs=n_workers)(
        delayed(open_df_from_res_csv)(fname) for fname in tqdm(cur_dataset_res_flist, desc=f'Opening graphs ({cur_dataset_res_path.stem})')
    )
    df_list_all += df_list
    
    for df in df_list:
        if df is not None:
            if 'g_method' not in df.columns:
                #print("Graphs were generated with older version of ADAPT-GPT preprocessor. Most likely, they are ER.")
                df['g_method'] = "erdos_renyi"
        
df_list  = [df for df in df_list_all if df is not None]
print("df_list len:", len(df_list))


full_run_graphs_df = pd.concat(df_list)
full_run_graphs_df['edgelist_list'] = (
    full_run_graphs_df['edgelist_json'].progress_apply(
        lambda x: json.loads(x)
    )
)
full_run_graphs_df['edgelist_list_len'] = (
    full_run_graphs_df['edgelist_list'].progress_apply(
        lambda x: len(x)
    )
)
full_run_graphs_df['num_connected_comp'] = full_run_graphs_df['edgelist_list'].progress_apply(
    lambda x: len(
        list(
            nx.connected_components(
                nx.Graph([edge[:2] for edge in x])
            )
        )
    )
)
full_run_graphs_df['prefix'] = full_run_graphs_df['worker_id'].apply(
    lambda x: x[:-12]
)

print("Graphs count:")
print(full_run_graphs_df['g_method'].value_counts())

# Aggregating results
print("Aggregating results...")

full_run_df['approx_ratio'] = (
    full_run_df['energy'] / full_run_df['energy_mqlib']
)

full_run_agg_stat_df = (
    full_run_df.groupby(
        [
            #"graph_name",
            "graph_num", "run", "prefix", "method",
            'optimizer',
            "gamma0", "pooltype", "graph_name", 'n_nodes']
    )
    .agg(
        {
            #'pooltype': 'count',
            'energy': list,
            #'method': 'last',
            'took_time': 'last',
            'energy_mqlib': 'last',
            'generator_index_in_pool': list,
            'approx_ratio': 'last',
            #'approx_ratio_eig': 'last',
            'edge_weight_norm_coef': 'last',
            'β_coeff': list,
            'γ_coeff': list,
            'coeff': list,
            'energy_mqlib': 'last',
            'energy_eigen': 'last',
            #'n_nodes': 'last',
            'cut_mqlib': 'last',
            'cut_adapt': 'last',
            'cut_eig': 'last',
            'state_vect_adapt': 'last',
            'success_flag': 'last'
        }
    )
    .reset_index()
    .rename(
        columns={
            'pooltype': 'pooltype',
            'energy': 'energy_list',
            'generator_index_in_pool': 'op_list',
        }
    )
)

combined_res_df = pd.merge(
    left=full_run_agg_stat_df,
    right=full_run_graphs_df,
    left_on=['prefix', 'graph_num'],
    right_on=['prefix', 'graph_num'],
)
combined_res_df["n_layers"] = combined_res_df['energy_list'].apply(len)

combined_res_df['graph_id'] = (
      combined_res_df['prefix']
    + '_^_'
    + combined_res_df['graph_num'].astype(str)
)

combined_res_df['only_qaoa_circ'] = combined_res_df['op_list'].progress_apply(
    lambda x: all(e == n_nodes+1 for e in x)
)

if allowed_graph_generators_list != ['all']:
    print(f"Filtering graphs based on allowed generators: {allowed_graph_generators_list}")
    print(f"N circuits before: {len(combined_res_df)}")
    combined_res_df = combined_res_df[
        combined_res_df['g_method'].isin(allowed_graph_generators_list)
    ]
    print(f"N circuits after: {len(combined_res_df)}")

print(combined_res_df['g_method'].value_counts())

#-----------------------#
# Graph embedding
print("Applying FEATHER graph embedding...")

combined_unique_graphs_df = (
    combined_res_df[['graph_id', 'edgelist_json']]
        .drop_duplicates()
)


def create_weighted_graph_nx(w_elist):
    G = nx.Graph()
    G.add_weighted_edges_from(w_elist)
    return G

combined_unique_graphs_df['edgelist_py_list'] = combined_unique_graphs_df['edgelist_json'].progress_apply(
    lambda x: [
        (e[0]-1, e[1]-1, e[2]) for e in json.loads(x)
        #(e[0]-1, e[1]-1) for e in x
    ]
)

combined_unique_graphs_df['graph_nx'] = (
    combined_unique_graphs_df['edgelist_py_list']
        .progress_apply(lambda x: create_weighted_graph_nx(x))
)

combined_unique_graphs_w_idx_df = combined_unique_graphs_df.set_index('graph_id')
graphs_nx_dict = combined_unique_graphs_w_idx_df['graph_nx'].to_dict()
graphs_nx_filt_dict = dict(
    [(name, g) for name, g in tqdm(graphs_nx_dict.items()) if g.number_of_nodes() == n_nodes]
)
graphs_nx_filt_names = list(graphs_nx_filt_dict.keys())
graphs_nx_filt_list = list(graphs_nx_filt_dict.values())

emb_graph_idx_to_id_dict = {k:v for k,v in enumerate(graphs_nx_filt_names)}
emb_graph_id_to_idx_dict = {v:k for k,v in enumerate(graphs_nx_filt_names)}

def get_feather_emb(g_list):
    feather_model = FEATHERG()
    feather_model.fit(graphs=g_list)
    return feather_model.get_embedding()

def split_list(lst, n):
    it = iter(lst)
    return [list(islice(it, i)) for i in [len(lst) // n + (1 if x < len(lst) % n else 0) for x in range(n)]]

def embed_nx_w_feather_parallel(graphs_list, n_workers=2):
    graphs_chunked_list = split_list(graphs_list, n_workers)
    
    #graphs_chunked_list=[graphs_list]
    
    emb_np_list = Parallel(n_jobs=n_workers)(
        delayed(get_feather_emb)(g_chunk) for g_chunk in graphs_chunked_list
    )
    
    return np.vstack(emb_np_list)

feather_par_emb = embed_nx_w_feather_parallel(graphs_nx_filt_list[:], n_workers=n_workers)
feather_par_emb = feather_par_emb.round(rounding_digits)

combined_res_df['has_emb'] = combined_res_df['graph_id'].apply(
    lambda x: True if x in emb_graph_id_to_idx_dict else False
)

#-----------------------#

# Filtering 
print("Selecting suitable ansatz...")

combined_res_filt_df = combined_res_df[
    # (
    #     combined_res_df['β_coeff'].apply(
    #         lambda x: all([abs(coef) < max_abs_param_val for coef in x])
    #     )
    # )
    # &
    (
        combined_res_df['γ_coeff'].apply(
            lambda x: all([abs(coef) < max_abs_param_val for coef in x])
        )
    )
    &
    (
        combined_res_df['approx_ratio'] > approx_ratio_thr
    )
]

if skip_only_qaoa_circ:
    print("Filtering out only QAOA circuits...")
    n_only_qaoa_circ = combined_res_filt_df['only_qaoa_circ'].sum()
    print(f"Removing {n_only_qaoa_circ} out of total {len(combined_res_filt_df)}")
    combined_res_filt_df = combined_res_filt_df[
        combined_res_filt_df['only_qaoa_circ'] == False
    ]
    

# Tokenization
print("Tokenizing...")

tokens_list = []
token_to_token_type_dict = dict()

## Special symbols
special_symbols_list = [
    'pad',
    'bos',
    'eos',
    'new_layer_p',
    'end_of_graph'
]
tokens_list += special_symbols_list
for tok in special_symbols_list:
    token_to_token_type_dict[tok] = 'SPECIAL'

## Edges
all_edges_list = []
for g in combined_res_filt_df['edgelist_list']:
    for e in g:
        all_edges_list.append(tuple(e[:2]))
all_edges_set = set(all_edges_list)

print(f"\tTotal tokens for edges: {len(all_edges_set)}")
tokens_list += list(all_edges_set)
for tok in all_edges_set:
    token_to_token_type_dict[tok] = 'EDGE'

## Coeffs

n_steps = int((max_abs_param_val * 2 / (10 ** -rounding_digits) ) + 1)

all_coefs_round_set = set(
    [
        round(coef, rounding_digits) for coef in np.linspace(start=-max_abs_param_val, stop=max_abs_param_val, num=n_steps).tolist()
    ]
)
len(all_coefs_round_set)
tokens_list += list(all_coefs_round_set)
for tok in all_coefs_round_set:
    token_to_token_type_dict[tok] = 'NUM'

print(f"\tTotal tokens for coefs: {len(all_coefs_round_set)}")

## Operator pool
ops_list = []
for l in combined_res_filt_df['op_list']:
    ops_list += l

ops_set = set(ops_list)
ops_list = [f"op_{op_int_idx}" for op_int_idx in ops_set]
print(f"\tTotal tokens for operator pool: {len(ops_list)}")
tokens_list += ops_list
for tok in ops_list:
    token_to_token_type_dict[tok] = 'OP'

## Tokenization
int_idx_to_token_dict = dict(enumerate(tokens_list))
token_to_int_idx_dict = {v:k for k,v in int_idx_to_token_dict.items()}

vocab_size = len(int_idx_to_token_dict)
print(f"\tTotal tokens in vocab: {vocab_size}")

assert vocab_size == len(token_to_token_type_dict)

def julia_mod(a, b):
    result = a % b
    return result if a >= 0 else result - b

def tokenize_row(row, coef_mod=True):

    tokens_seq_list = ['bos']

    for edge in row['edgelist_list']:
        edge_tuple = tuple(edge[:2])
        edge_weight = edge[2]
        tokens_seq_list.append(edge_tuple)
        tokens_seq_list.append(edge_weight)

    tokens_seq_list.append('end_of_graph')

    for p in range(row['n_layers']):
        tokens_seq_list.append('new_layer_p')
        op_int_idx = row['op_list'][p]
        tokens_seq_list.append(f"op_{op_int_idx}")

        cur_beta = row['β_coeff'][p]
        if coef_mod:
            # cur_beta = cur_beta % (np.pi)
            cur_beta = julia_mod(cur_beta, np.pi)
        if cur_beta > -max_abs_param_val and cur_beta < max_abs_param_val:
            cur_beta_round = round(cur_beta, rounding_digits)
            tokens_seq_list.append(cur_beta_round)
        else:
            return None

        cur_gamma = row['γ_coeff'][p]
        # if coef_mod:
        #     cur_gamma = cur_gamma % (2*np.pi)
        if cur_gamma > -max_abs_param_val and cur_gamma < max_abs_param_val:
            cur_gamma_round = round(cur_gamma, rounding_digits)
            tokens_seq_list.append(cur_gamma_round)
        else:
            return None
    
    tokens_seq_list.append('eos')
    
    return tokens_seq_list

combined_res_filt_df[f'token_seq_round_d{rounding_digits}'] = combined_res_filt_df.progress_apply(
    lambda x: tokenize_row(x, coef_mod=perform_coef_mod_range),
    axis=1,
)
combined_res_tok_df = combined_res_filt_df.dropna()
combined_res_tok_df[f'token_int_seq_round_d{rounding_digits}'] = (
    combined_res_tok_df[f'token_seq_round_d{rounding_digits}'].progress_apply(
        lambda x: [token_to_int_idx_dict[token] for token in x]
    )
)

# Generating training split for nanoGPT

print("Preparing training data...")

n = len(combined_res_tok_df)

combined_res_tok_shf_df = (
    combined_res_tok_df
        .sample(frac=1)
        .reset_index(drop=True)
)

print(f"combined_res_df shape: {combined_res_df.shape}")
print(f"combined_res_tok_df shape: {combined_res_tok_df.shape}")
print(f"combined_res_tok_shf_df shape: {combined_res_tok_shf_df.shape}")

graph_ids = combined_res_tok_shf_df['graph_id'].drop_duplicates().to_list()

# Compute the number of graphs for each set
n_total = len(graph_ids)
n_test = int(n_total * test_frac)  # Define test_frac for the size of the test set
n_val = int(n_total * val_frac)  # val_frac defines the validation set size
n_train = n_total - n_test - n_val  # Remaining will be the training set

# Split into train, val, and test sets
train_graph_ids_set = set(graph_ids[:n_train])
val_graph_ids_set = set(graph_ids[n_train:n_train + n_val])
test_graph_ids_set = set(graph_ids[n_train + n_val:])

assert len(train_graph_ids_set.intersection(val_graph_ids_set)) == 0
assert len(train_graph_ids_set.intersection(test_graph_ids_set)) == 0
assert len(val_graph_ids_set.intersection(test_graph_ids_set)) == 0

def pad_with_zeros(seq, target_len):
    pad_len = target_len - len(seq)
    if pad_len > 0:
        padded_seq = seq + [0]*pad_len
    else:
        padded_seq = seq
        
    if len(padded_seq) !=max_block_size:
        print(f"padded_seq len: {len(padded_seq)}")
    return padded_seq

def sliding_window(numbers, min_block_size, max_block_size):
    
    if min_block_size != max_block_size:
        block_size = random.randint(min_block_size, max_block_size)
    else:
        block_size = min_block_size

    if block_size >= len(numbers):
        window = numbers[:-1]
        window_shifted = numbers[1:]   
        return [
            [
                pad_with_zeros(window, target_len=max_block_size),
                pad_with_zeros(window_shifted, target_len=max_block_size)
            ]
        ]
    
    result_xy_list = []
    result = []
    for i in range(0, len(numbers) - block_size + 1):
        window = numbers[i:i + block_size]
        result.append(window)
        
    for x, y in zip(result, result[1:]):
        result_xy_list.append(
            [
                pad_with_zeros(x, target_len=max_block_size),
                pad_with_zeros(y, target_len=max_block_size)
            ]
        )
    
    return result_xy_list


# Assign the 'label' column based on the split
combined_res_tok_shf_df['label'] = 'train'
combined_res_tok_shf_df.loc[combined_res_tok_shf_df['graph_id'].isin(val_graph_ids_set), 'label'] = 'val'
combined_res_tok_shf_df.loc[combined_res_tok_shf_df['graph_id'].isin(test_graph_ids_set), 'label'] = 'test'

if apply_sliding_window:
    print('Applying sliding window...')
    combined_res_tok_shf_df[f'token_int_seq_round_d{rounding_digits}_sw'] = combined_res_tok_shf_df[f'token_int_seq_round_d{rounding_digits}'].progress_apply(
        lambda x: sliding_window(
            x,
            min_block_size=min_block_size,
            max_block_size=max_block_size
        )
    )
    
train_data = combined_res_tok_shf_df[
    combined_res_tok_shf_df['label'] == 'train'
]
val_data = combined_res_tok_shf_df[
    combined_res_tok_shf_df['label'] == 'val'
]
test_data = combined_res_tok_shf_df[
    combined_res_tok_shf_df['label'] == 'test'
]

print(f"\tNumber of training samples: {len(train_data)}, val samples: {len(val_data)}, test samples: {len(test_data)}")

if apply_sliding_window:

    train_data_conc = []
    train_data_graph_idx_list = []
    for cur_graph_id, l in zip(
        train_data['graph_id'],
        train_data[f'token_int_seq_round_d{rounding_digits}_sw']
    ):
        if cur_graph_id in emb_graph_id_to_idx_dict:
            train_data_conc += l
            train_data_graph_idx_list += [emb_graph_id_to_idx_dict[cur_graph_id]] * len(l)
    train_data_conc_np = np.array(train_data_conc, dtype=np.uint16)

    val_data_conc = []
    val_data_graph_idx_list = []
    for cur_graph_id, l in zip(
        val_data['graph_id'],
        val_data[f'token_int_seq_round_d{rounding_digits}_sw']
    ):
        if cur_graph_id in emb_graph_id_to_idx_dict:
            val_data_conc += l
            val_data_graph_idx_list += [emb_graph_id_to_idx_dict[cur_graph_id]] * len(l)
    val_data_conc_np = np.array(val_data_conc, dtype=np.uint16)

    test_data_conc = []
    test_data_graph_idx_list = []
    for cur_graph_id, l in zip(
        test_data['graph_id'],
        test_data[f'token_int_seq_round_d{rounding_digits}_sw']
    ):
        if cur_graph_id in emb_graph_id_to_idx_dict:
            test_data_conc += l
            test_data_graph_idx_list += [emb_graph_id_to_idx_dict[cur_graph_id]] * len(l)
    test_data_conc_np = np.array(test_data_conc, dtype=np.uint16)


    # print(f"\tTrain has {len(train_data_conc_np):,} samples")
    # print(f"\tVal has {len(val_data_conc_np):,} samples")
    # print(f"\tTest has {len(test_data_conc_np):,} samples")

# Saving

save_path.mkdir(parents=True, exist_ok=True)

if apply_sliding_window:
    
    np.save(
        save_path.joinpath('train.npy'),
        train_data_conc_np
    )
    np.save(
        save_path.joinpath('val.npy'),
        val_data_conc_np
    )
    np.save(
        save_path.joinpath('test.npy'),
        test_data_conc_np
    )

combined_res_df.to_pickle(
    save_path.joinpath('combined_res_df.pkl')
)

combined_res_tok_shf_df.to_pickle(
    save_path.joinpath('combined_res_tok_shf_df.pkl')
)

target_val_size = 1000
sample_size_per_w_bucket = int(
      target_val_size
    / len(combined_res_df['edgelist_list_len'].drop_duplicates())
)
val_data_sampled = (
    val_data[
          (~val_data['token_seq_round_d2'].isna())
    ]
    .groupby('edgelist_list_len').apply(
        lambda x: x.sample(sample_size_per_w_bucket) if len(x) > sample_size_per_w_bucket else x
    )
    .reset_index(drop=True)
)

val_data_sampled.to_pickle(
    save_path.joinpath('combined_res_tok_shf_val_df.pkl')
)

emb_size = feather_par_emb.shape[1]
np.save(
    save_path.joinpath(f'feather_emb_d{emb_size}.npy'),
    feather_par_emb
)

meta = {
    'vocab_size': vocab_size,
    'itos': int_idx_to_token_dict,
    'stoi': token_to_int_idx_dict,
    'train_data_graph_idx_list': train_data_graph_idx_list,
    'val_data_graph_idx_list': val_data_graph_idx_list,
    'test_data_graph_idx_list': test_data_graph_idx_list,
    'emb_graph_id_to_idx_dict': emb_graph_id_to_idx_dict,
    'emb_graph_idx_to_id_dict': emb_graph_idx_to_id_dict,
    'token_to_token_type_dict': token_to_token_type_dict, # Addon from 11/08/25
}

pd.to_pickle(
    meta,
    save_path.joinpath('meta.pkl')
)

with open('train_adapt_gpt_config_template.py') as f:
    config_template_str = f.read()

pool_type = combined_res_df['pooltype'].iloc[0]

dataset_name = save_path.stem
config_to_save_str = config_template_str.format(
    out_dir=f'out-{dataset_name}',
    dataset=dataset_name,
    block_size=max_block_size,
    use_graph_emb="True",
    pool_type=pool_type,
    n_nodes=n_nodes,
)

with open(save_path.joinpath('train_adapt_gpt_config.py'), 'w') as f:
    f.write(config_to_save_str)

print(f"Data is saved to: {str(save_path.absolute())}")
print("Done!")

