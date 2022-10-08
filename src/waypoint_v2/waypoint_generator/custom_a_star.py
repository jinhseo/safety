from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function

def astar_path_custom(G, source, target, heuristic=None, weight="weight"):
    if source not in G or target not in G:
        msg = "Either source {} or target {} is not in G".foramt(source, target)
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        def heuristic(u, v):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            print('SELECTED PATH:',path)
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G[curnode].items():
            multiplier = 300
            acc_imbalance, path, changes_occured = imbalance_measure(G, curnode, neighbor, explored)

            ncost = dist + weight(curnode, neighbor, w) #+ acc_imbalance * multiplier
            # if path[-1] == target:
            print('PATH:', path, 'CHANGES:', changes_occured)
            print('{} = {} + {} + {}'.format(ncost, dist, weight(curnode, neighbor, w), acc_imbalance * multiplier))

            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target) + acc_imbalance * multiplier

            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath("Node {} not reachable from {}".format(target, source))

def imbalance(lane_change_idx):
    prev_change = 0
    imbalance = []
    for i, j in zip(lane_change_idx[:-1], lane_change_idx[1:]):
        imbalance.append(abs(abs(j-i) - abs(i-prev_change)))
        prev_change = j
    
    # result = sum(imbalance)/len(imbalance)
    result = sum(imbalance)
    return result


def imbalance_measure(graph, curnode, parent, explored):
    path = [parent]
    node = curnode

    while node is not None:
        path.append(node)
        node = explored[node]
    path.reverse()
    
    lane_change_indices = [0]
    for i, _ in enumerate(path[1:-2]):
        start, dest = path[i], path[i+1]
        change = graph.edges[start,dest]['change']
        if change == 'LANE_CHANGE_LEFT' or change == 'LANE_CHANGE_RIGHT':
            lane_change_indices.append(i)
    
    if(len(path)-1 not in lane_change_indices):
        lane_change_indices.append(len(path)-1)


    accumulated_imbalance = imbalance(lane_change_indices)
    changes_occured = [path[i] for i in lane_change_indices]
    
    return accumulated_imbalance, path, changes_occured