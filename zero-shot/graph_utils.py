# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import warnings
import random
import heapq

def dijkstra_shortest_path(graph, start, end, search_nodes='all'):
    # Check if start node exists in the graph
    if start not in graph:
        return [], float('inf')
    # If end node is not in the graph, add it with edges from its connected nodes with infinite weight
    if end not in graph:
        graph[end] = {node: float('inf') for node in graph.keys() if end in graph[node]}

    assert search_nodes == 'all' or isinstance(search_nodes, list)
    # Initialize distances and previous nodes
    if search_nodes == 'all':
        distances = {node: float('inf') for node in graph}
        previous_nodes = {node: None for node in graph}
    else:
        if start not in search_nodes:
            search_nodes.append(start)
        if end not in search_nodes:
            search_nodes.append(end)
        distances = {node: float('inf') for node in graph if node in search_nodes}
        previous_nodes = {node: None for node in graph if node in search_nodes}
    distances[start] = 0
    
    # Priority queue to store nodes to visit
    priority_queue = [(0, start)]
    
    while priority_queue:
        # Get node with smallest distance
        current_distance, current_node = heapq.heappop(priority_queue)
        
        # Stop if we reach the end node
        if current_node == end:
            break
        
        # Update distances for neighboring nodes
        if current_node not in graph:
            continue
        for neighbor, weight in graph[current_node].items():
            if search_nodes != 'all' and neighbor not in search_nodes:
                continue
            distance = current_distance + weight
            if neighbor not in distances:
                distances[neighbor] = float('inf')
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))
    
    # Trace back the shortest path from end to start
    path = []
    current_node = end
    while current_node is not None:
        path.append(current_node)
        current_node = previous_nodes[current_node]
    
    # Reverse the path and return it as a list
    path.reverse()
    return path, distances[end]