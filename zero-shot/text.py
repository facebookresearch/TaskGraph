# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from tqdm import tqdm
import torch
import numpy as np
import sys
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

from coin_loader import COINDataset
from crosstask_loader import CrossTaskDataset
from graph_utils import dijkstra_shortest_path
from construct_graph import TaskGraph
from metrics import IoU_class, Acc_class

class Print():
    def __init__(self):
        self.data = []
        try:
            self.hold = True
        except:
            self.hold = False
        
    def hold_and_print(self, datum):
        if not self.hold:
            print(datum)
        self.data.append(datum)
    
    def release(self):
        if self.hold:
            print(self.data)
print_obj = Print()

eval_modality = 'text'
dataset = sys.argv[1]

if dataset == 'crosstask':
    # We get good performance without using clutering
    beam_search_thresh = 0.32
    keystep_thresh = 0.46
    prune_keysteps = True
    use_clusters = False
    graph_type = 'overall'
    method = 'beam-search'
elif dataset == 'coin':
    beam_search_thresh = 0.3
    keystep_thresh = 0.0
    prune_keysteps = False
    use_clusters = True
    graph_type = 'overall'
    method = 'beam-search'
    match_thresh = 0.46
    clustering_distance_thresh = 1.0

# Statistics
data_statistics = {'coin':
                {'video': {'max': 24.12, 'min': -23.83},
                'text': {'max': 1.0, 'min': -1.0},
                },
            'crosstask':
                {'video': {'max': 27.45, 'min': -18.0},
                'text': {'max': 1.0, 'min': -1.0},
                }
            }

# Load dataset
if dataset == 'coin':
    eval_dataset = COINDataset(modality=eval_modality, graph_type=graph_type)
elif dataset == 'crosstask':
    eval_dataset = CrossTaskDataset(modality=eval_modality, graph_type=graph_type)
else:
    raise NotImplementedError

if use_clusters:
    text_labels = eval_dataset.get_text_labels()
    # Clustering
    embedder = SentenceTransformer('all-mpnet-base-v2')
    # Corpus with example sentences
    corpus_embeddings = embedder.encode(text_labels)
    corpus_embeddings_orig = embedder.encode(text_labels)

    # Normalize the embeddings to unit length
    corpus_embeddings = corpus_embeddings /  np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(n_clusters=None, distance_threshold=clustering_distance_thresh) #, affinity='cosine', linkage='average', distance_threshold=0.4)
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(text_labels[sentence_id])

    cluster_per_sentence = {}
    embeds_per_cluster = {}

    import json
    import os
    with open('/datasets01/COIN/053122/COIN.json') as f:
        coin_meta = json.load(f)['database']

    # get video classes
    video_classes = []
    for vid in coin_meta:
        # Obtain video classes
        if coin_meta[vid]['class'] not in video_classes:
            video_classes.append(coin_meta[vid]['class'])

    # for every video, match it to the nearest cluster
    from tqdm import tqdm
    cluster_assignment_per_video = {}
    for vid in tqdm(coin_meta):
        if coin_meta[vid]['subset'] == 'testing' and os.path.exists(os.path.join('/path/to/coin_all_scores/', '{}.pkl'.format(vid))):
            curr_video_class = video_classes.index(coin_meta[vid]['class'])
            with open('/path/to/coin_processed/{}.json'.format(vid)) as fvv:
                captions = json.load(fvv)
            caption_embeds = embedder.encode(captions['text'])
            sentences = np.matmul(caption_embeds, corpus_embeddings_orig.transpose())
            # now match it to per cluster embeds
            num_matches_per_cluster = {}
            for key in clustered_sentences:
                cluster_sents = clustered_sentences[key]
                scores = sentences[:, [text_labels.index(x) for x in cluster_sents]]
                num_matches_per_cluster[key] = np.sum(scores > match_thresh) / scores.shape[1]
            best_cluster = max(num_matches_per_cluster, key=num_matches_per_cluster.get)
            cluster_assignment_per_video[vid] = best_cluster

# Now do evaluation
result = {'{}-all'.format(method): []}

all_preds = []
all_labels = []
if method == 'random':
    import random
    avg_classes = []
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum['labels']
        scores = datum['sim_score']
        preds = []
        total_classes = None
        for score in scores:
            if not isinstance(score, int):
                total_classes = score.size
                break
        if total_classes is not None:
            avg_classes.append(total_classes)
        for score in scores:
            if total_classes is not None:
                preds.append(random.randint(0, total_classes-1))
            else:
                preds.append(-1)
        preds = np.array(preds)
        label = np.array(label)
        iou_cls = IoU_class(preds, label)
        acc_cls = Acc_class(preds, label, use_negative=False)
        all_preds.append(preds)
        all_labels.append(label)
    print('Avg classes: {}'.format(sum(avg_classes)/len(avg_classes)))
elif method == 'baseline':
    max_logit_val = -1000 # To determine thresholds
    min_logit_val = 1000
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum['labels']
        scores = datum['sim_score']
        preds = []
        for score in scores:
            if isinstance(score, int) and score == -1:
                preds.append(-1)
            else:
                if np.max(score) > max_logit_val:
                    max_logit_val = np.max(score)
                if np.min(score) < min_logit_val:
                    min_logit_val = np.min(score)
                preds.append(int(np.argmax(score)))
        preds = np.array(preds)
        label = np.array(label)
        iou_cls = IoU_class(preds, label)
        acc_cls = Acc_class(preds, label, use_negative=False)
        all_preds.append(preds)
        all_labels.append(label)
elif method == 'beam-search':
    task_graph = TaskGraph(dataset=dataset, graph_type=graph_type, graph_modality=eval_modality)
    text_labels_overall = eval_dataset.get_text_labels()
    print('Populating task graph...')
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum['labels']
        scores = datum['sim_score']
        class_idx = datum['video_class'] if graph_type == 'task' else -1
        if graph_type == 'overall':
            text_labels = text_labels_overall
        elif graph_type == 'task':
            text_labels = text_labels_overall[class_idx]
        # Populate task graph
        # Ignore -1s
        scores = [x for x in scores if not isinstance(x, int)]
        if len(scores) > 0:
            scores = np.concatenate(scores, axis=0)
            task_graph.register_sequence(sim=scores, keystep_sents=text_labels, class_idx=class_idx)
    # Normalize task graph
    task_graph.check_and_finalize(apply_log=False)
    # Now perform correction
    for idx, datum in enumerate(tqdm(eval_dataset)):
        label = datum['labels']
        scores = datum['sim_score']
        class_idx = datum['video_class'] if graph_type == 'task' else -1
        if graph_type == 'overall':
            text_labels = text_labels_overall
            final_task_graph = task_graph.task_graph
        elif graph_type == 'task':
            text_labels = text_labels_overall[class_idx]
            final_task_graph = task_graph.task_graph[class_idx]
        
        if prune_keysteps and not use_clusters:
            search_nodes = []
            for score in scores:
                if not isinstance(score, int):
                    if np.max(score) > keystep_thresh:
                        if text_labels[np.argmax(score)] not in search_nodes:
                            # print(len(text_labels))
                            # print(np.argmax(scores[time_idx]))
                            search_nodes.append(text_labels[np.argmax(score)])
        elif use_clusters:
            search_nodes = clustered_sentences[cluster_assignment_per_video[datum['video_id']]]
        else:
            search_nodes = 'all'
        # print(search_nodes)
        # print('Starting search nodes..............')

        preds = []
        # Replace the low scores with -1
        for score in scores:
            if isinstance(score, int) and score == -1:
                preds.append(-1)
            else:
                # Normalize score to be in the range [-1, 1]
                score = 2 * (score - data_statistics[dataset][eval_modality]['min']) / (data_statistics[dataset][eval_modality]['max'] - data_statistics[dataset][eval_modality]['min']) - 1
                if search_nodes == 'all' or len(search_nodes) == 0:
                    if np.max(score) < beam_search_thresh:
                        preds.append(-1)
                    else:
                        preds.append(int(np.argmax(score)))
                else:
                    focus_labels = [text_labels.index(x) for x in search_nodes]
                    subset_score = score[:, focus_labels]
                    max_val = np.max(subset_score)
                    max_idx = np.argmax(subset_score)
                    orig_max_idx = focus_labels[max_idx]
                    if max_val < beam_search_thresh:
                        preds.append(-1)
                    else:
                        preds.append(orig_max_idx)

        # Replace -1s with beam search
        preds_before = preds.copy()
        prev_label_time = -1
        prev_label = -1
        for time_idx in range(len(preds)):
            if preds[time_idx] != -1:
                # Do backward correction
                curr_label_time = time_idx
                curr_label = preds[time_idx]

                if curr_label_time - prev_label_time == 1:
                    continue

                if prev_label != -1:
                    best_path, weight = dijkstra_shortest_path(final_task_graph, text_labels[prev_label], text_labels[curr_label], search_nodes=search_nodes)
                else:
                    best_path = [text_labels[curr_label]]
                    prev_label = curr_label
    
                # We need to update starting prev_label_time+1 to curr_label_time
                if len(best_path) > 0:
                    update_labels = [prev_label] + [text_labels.index(y) for y in best_path] + [curr_label]
                else:
                    update_labels = [prev_label, curr_label]
                n_segs = len(update_labels)
                for count_idx, corrected_time_idx in enumerate(range(prev_label_time+1, curr_label_time)):
                    preds[corrected_time_idx] = update_labels[int(n_segs * count_idx/len(range(prev_label_time+1, curr_label_time)))]

                prev_label_time = curr_label_time
                prev_label = curr_label
        preds = np.array(preds)
        label = np.array(label)
        all_preds.append(preds)
        all_labels.append(label)

all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

result['{}-all'.format(method)].append((Acc_class(all_preds, all_labels, use_negative=False), IoU_class(all_preds, all_labels)))
for name in result:
    acc = np.mean([r[0] for r in result[name]])
    iou = np.mean([r[1] for r in result[name]])
    print(f"Acc {acc}, IoU: {iou}")