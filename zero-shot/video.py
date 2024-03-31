# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import statistics
from metrics import IoU_class, Acc_class
from construct_graph import TaskGraph

from coin_loader import COINDataset
from crosstask_loader import CrossTaskDataset
from graph_utils import dijkstra_shortest_path
from tqdm import tqdm
import sys
import torch

class Print():
    def __init__(self):
        self.data = []
        try:
            b = float(sys.argv[1])
            self.hold = True
        except:
            b = -100.
            self.hold = False
        
    def hold_and_print(self, datum):
        if not self.hold:
            print(datum)
        self.data.append(datum)
    
    def release(self):
        if self.hold:
            print(self.data)
print_obj = Print()

eval_modality = 'video'
dataset = sys.argv[1]

if dataset == 'crosstask':
    # We get good performance without using clutering
    beam_search_thresh = 0.06
    beam_search_thresh_video = 0.06
    beam_search_thresh_text = 0.06
    keystep_thresh = 0.16
    prune_keysteps = False
    use_clusters = True
    graph_type = 'overall'
    method = 'beam-search-with-cluster'
elif dataset == 'coin':
    beam_search_thresh = 0.0
    beam_search_thresh_video = -1.0
    beam_search_thresh_text = 0.4
    keystep_thresh = 0.0
    prune_keysteps = False
    use_clusters = True
    graph_type = 'overall'
    method = 'beam-search-with-cluster'
    match_thresh = 0.3
    clustering_distance_thresh = 1.25

data_statistics = {'coin':
                {'video': {'max': 24.12, 'min': -23.83}, # for videoclip
                # {'video': {'max': 57.33, 'min': -74.26}, # for s3d
                'text': {'max': 1.0, 'min': -1.0},
                },
            'crosstask':
                # {'video': {'max': 27.45, 'min': -18.0}, #for videoclip
                {'video': {'max': 58.8, 'min': -63.42}, # for s3d
                'text': {'max': 1.0, 'min': -1.0},
                }
            }

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

    for key, values in clustered_sentences.items():
        indices = [text_labels.index(x) for x in values]
        embeds_per_cluster[key] = corpus_embeddings[indices, :]
        for value in values:
            cluster_per_sentence[value] = key

    per_class_text = eval_dataset.text_labels_per_class
    percent_match = 0.0
    cluster_assignment_per_class = {}
    print(per_class_text)
    # exit()
    for class_idx in per_class_text:
        chosen_clusters = []
        for text in per_class_text[class_idx]:
            chosen_clusters.append(cluster_per_sentence[text])
        mode_value = statistics.multimode(chosen_clusters)[0]
        cluster_assignment_per_class[class_idx] = mode_value
        mode_ratio = chosen_clusters.count(mode_value) / len(chosen_clusters)
        percent_match += mode_ratio
    percent_match /= len(per_class_text)
    print('Percent match', percent_match)
    print('Elements per cluster', sum(len(lst) for lst in clustered_sentences.values()) / len(clustered_sentences))
    print('Num clusters: {}'.format(len(clustered_sentences)))
    ############# Now load COIN video feature and then see how many are correctly assigned to a cluster ###########
    import json
    import os
    import pickle
    if dataset == 'coin':
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
            clustering_modality = eval_modality if eval_modality != 'video-text' else 'video'
            if coin_meta[vid]['subset'] == 'testing' and os.path.exists(os.path.join('/path/to/videoclip_video_features/', '{}.pt'.format(vid))):
                curr_video_class = video_classes.index(coin_meta[vid]['class'])
                features = torch.load(os.path.join('/path/to/videoclip_video_features/', '{}.pt'.format(vid))).numpy()
                features = 2 * (features - data_statistics[dataset][clustering_modality]['min']) / (data_statistics[dataset][clustering_modality]['max'] - data_statistics[dataset][clustering_modality]['min']) - 1
                # now match it to per cluster embeds
                num_matches_per_cluster = {}
                for key in clustered_sentences:
                    cluster_sents = clustered_sentences[key]
                    scores = features[:, [text_labels.index(x) for x in cluster_sents]]
                    num_matches_per_cluster[key] = np.sum(scores > match_thresh) / scores.shape[1]
                best_cluster = max(num_matches_per_cluster, key=num_matches_per_cluster.get)
                cluster_assignment_per_video[vid] = best_cluster
    elif dataset == 'crosstask':
        crosstask_annotation_path = '/datasets01/CrossTask/053122/crosstask_release/annotations/'
        # with open('/datasets01/COIN/053122/COIN.json') as f:
        #     coin_meta = json.load(f)['database']
        crosstask_annotations = os.listdir(crosstask_annotation_path)


        with open('/datasets01/CrossTask/053122/crosstask_release/tasks_primary.txt') as f:
            lines = f.readlines()
            tasks = [x.strip() for idx, x in enumerate(lines) if idx % 6 == 4]
            video_classes = [int(x.strip()) for idx, x in enumerate(lines) if idx % 6 == 0]

        # for every video, match it to the nearest cluster
        from tqdm import tqdm
        cluster_assignment_per_video = {}
        clustering_modality = eval_modality if eval_modality != 'video-text' else 'video'
        for annt_file in tqdm(crosstask_annotations):
            curr_video_class, vid = annt_file.split('_')[0], annt_file[-15:-4]
            if os.path.exists(os.path.join('/path/to/videoclip_video_features_crosstask_s3d/', '{}.pt'.format(vid))):
                # curr_video_class = video_classes.index(coin_meta[vid]['class'])
                features = torch.load(os.path.join('/path/to/videoclip_video_features_crosstask_s3d/', '{}.pt'.format(vid))).numpy()
                features = 2 * (features - data_statistics[dataset][clustering_modality]['min']) / (data_statistics[dataset][clustering_modality]['max'] - data_statistics[dataset][clustering_modality]['min']) - 1
                # sentences = np.concatenate(sentences)
                # now match it to per cluster embeds
                num_matches_per_cluster = {}
                for key in clustered_sentences:
                    cluster_sents = clustered_sentences[key]
                    scores = features[:, [text_labels.index(x) for x in cluster_sents]]
                    num_matches_per_cluster[key] = np.sum(scores > match_thresh) / scores.shape[1]
                best_cluster = max(num_matches_per_cluster, key=num_matches_per_cluster.get)
                cluster_assignment_per_video[vid] = best_cluster


# Now do evaluation
result = {'{}-all'.format(method): []}
all_preds = []
all_labels = []
if method == 'baseline-with-cluster':
    if eval_modality == 'video-text':
        for idx, datum in enumerate(tqdm(eval_dataset)):
            label = datum['labels']
            scores = datum['sim_score']

            preds = []
            keystep_sentences = clustered_sentences[cluster_assignment_per_video[datum['video_id']]]
            keystep_indices = [text_labels.index(x) for x in keystep_sentences]
            text_scores = scores['text']
            video_scores = scores['video']
            for time_idx in range(len(video_scores)):
                v_score = video_scores[time_idx]
                t_score = text_scores[time_idx]

                if isinstance(v_score, int) and v_score == -1 and isinstance(t_score, int) and t_score == -1:
                    preds.append(-1)
                elif isinstance(v_score, int) and v_score == -1:
                    # v_score = 2 * (v_score - statistics[dataset]['video']['min']) / (statistics[dataset]['video']['max'] - statistics[dataset]['video']['min']) - 1
                    t_score = 2 * (t_score - data_statistics[dataset]['text']['min']) / (data_statistics[dataset]['text']['max'] - data_statistics[dataset]['text']['min']) - 1
                    # if search_nodes == 'all' or len(search_nodes) == 0:
                    if np.max(t_score) < beam_search_thresh_text:
                        preds.append(-1)
                    else:
                        curr_score = t_score[:, keystep_indices]
                        preds.append(keystep_indices[np.argmax(curr_score)])
                else: #Assuming video as priority
                    v_score = 2 * (v_score - data_statistics[dataset]['video']['min']) / (data_statistics[dataset]['video']['max'] - data_statistics[dataset]['video']['min']) - 1

                    curr_score = v_score[:, keystep_indices]
                    subset_score = v_score[:, keystep_indices]
                    max_val = np.max(subset_score)
                    max_idx = np.argmax(subset_score)
                    orig_max_idx = keystep_indices[max_idx]
                    if max_val < beam_search_thresh_video:
                        assert False
                        # Before inserting -1 check for lower priority text scores
                        if not isinstance(t_score, int):
                            subset_score_text = t_score[:, keystep_indices]
                            max_val_text = np.max(subset_score_text)
                            max_idx_text = np.argmax(subset_score_text)
                            orig_max_idx_text = keystep_indices[max_idx_text]
                            if max_val_text < beam_search_thresh_text:
                                preds.append(-1)
                            else:
                                preds.append(orig_max_idx_text)
                        else:
                            preds.append(-1)
                    else:
                        preds.append(orig_max_idx)
            preds = np.array(preds)
            label = np.array(label)
            iou_cls = IoU_class(preds, label)
            acc_cls = Acc_class(preds, label, use_negative=False)
            result['{}'.format(method)].append((acc_cls, iou_cls))
            all_preds.append(preds)
            all_labels.append(label)
    else:
        for idx, datum in enumerate(tqdm(eval_dataset)):
            label = datum['labels']
            raw_sentences = datum['sim_score']
            preds = []
            keystep_sentences = clustered_sentences[cluster_assignment_per_video[datum['video_id']]]
            keystep_indices = [text_labels.index(x) for x in keystep_sentences]
            for raw_sentence in raw_sentences:
                if isinstance(raw_sentence, int) and raw_sentence == -1:
                    preds.append(-1)
                else:
                    curr_score = raw_sentence[:, keystep_indices]
                    preds.append(keystep_indices[np.argmax(curr_score)])
            preds = np.array(preds)
            label = np.array(label)
            all_preds.append(preds)
            all_labels.append(label)
elif method == 'beam-search-with-cluster':
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