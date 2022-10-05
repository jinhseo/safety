import csv
import networkx as nx
from geodesy.utm import fromLatLong as proj
import numpy as np
from functools import partial

class path_generator():
    def __init__(self, type = 'A', crossroad = False):
        self.center = proj(35.64838, 128.40105, 0)
        self.center = np.array([self.center.easting, self.center.northing, self.center.altitude])

        self.node_index = 0
        self.G, self.id_map = self.generate_graph(type, crossroad)

        self.points = np.array([self.gps_to_utm(float(lat), float(lon), 0) for (lat, lon) in [self.G.nodes[n]['vertex'] for n in self.G.nodes()]])
        self.dist_heuristic = partial(self.distance_heuristic, self.G)

    def gps_to_utm(self, latitude, longitude, altitude):
        pos = proj(latitude, longitude, altitude)
        pos = np.array([pos.easting, pos.northing, pos.altitude])
        pos[:2] -= self.center[:2]
        return pos

    def get_closest_index(self, target_pos):
        min_dist = np.finfo(np.float64).max
        min_idx = -1
        for i in range(len(self.points)):
            dist = np.linalg.norm(target_pos - self.points[i])
            if min_dist >= dist:
                min_idx = self.id_map[self.G.nodes[i]['vertex']]
                min_dist = dist
        return min_idx

    def get_closest_index_among(self, target_pos, among):
        min_dist = np.finfo(np.float64).max
        min_idx = -1
        for i in range(len(among)):
            dist = np.linalg.norm(target_pos - among[i])
            if min_dist >= dist:
                min_idx = i
                min_dist = dist
        return min_idx


    def generate_graph(self, type, crossroad):
        graph = nx.DiGraph()
        id_map = dict()

        print('Loading the graph')
        self.load_graph(graph, id_map, type, crossroad)

        # print('Adding custom nodes')
        # self.load_custom(graph, id_map)
        # self.add_node_from_text(graph, id_map, 'right33.txt')

        # print('Saving the graph')
        # self.store_graph(graph, '1')

        return graph, id_map

    def add_node_from_text(self, graph, id_map, file_name):
        temp_v = []
        str_v = []
        str_e = []
        with open(file_name, 'rt') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                vertex = tuple([float(row[0]), float(row[1])])
                temp_v.append(vertex)

        INTERVAL = 10
        for i,j in zip(range(0, len(temp_v)-INTERVAL, INTERVAL), range(INTERVAL, len(temp_v), INTERVAL)):
            if temp_v[i] not in id_map.keys():
                vertex = temp_v[i]
                id_map[vertex] = self.node_index
                self.node_index += 1
                graph.add_node(id_map[vertex], vertex=vertex, stop='TEST')
                str_v.append(str(vertex[0])+','+str(vertex[1]))

            if temp_v[j] not in id_map.keys():
                vertex = temp_v[j]
                id_map[vertex] = self.node_index
                self.node_index += 1
                graph.add_node(id_map[vertex], vertex=vertex, stop='TEST')
                str_v.append(str(vertex[0])+','+str(vertex[1]))

            v1, v2 = id_map[temp_v[i]], id_map[temp_v[j]]
            change = 'RIGHT'
            length = self.latlon_dist(graph.nodes[v1]['vertex'], graph.nodes[v2]['vertex'])
            graph.add_edge(v1, v2, length=length, change=change)
            str_e.append(str(temp_v[i][0])+','+str(temp_v[i][1])+','+str(temp_v[j][0])+','+str(temp_v[j][1]))

    def store_graph(self, graph, postfix):
        latlon_list = [(float(lat), float(lon)) for lat, lon in [graph.nodes[n]['vertex'] for n in graph.nodes()]]
        stop_list = [stop for stop in [graph.nodes[n]['stop'] for n in graph.nodes()]]

        f = open('node_'+str(postfix)+'.txt', 'wt')
        for (lat,lon),stop in zip(latlon_list, stop_list):
            f.write(str(lat)+','+str(lon)+','+str(stop)+'\n')
        f.close()

        f = open('edge_'+str(postfix)+'.txt', 'wt')
        for start, dest in list(graph.edges()):
            lat1, lon1 = graph.nodes[start]['vertex']
            lat2, lon2 = graph.nodes[dest]['vertex']
            change = graph.edges[start,dest]['change']
            f.write(str(lat1)+','+str(lon1)+','+str(lat2)+','+str(lon2)+','+str(change)+'\n')
        f.close()

    def load_custom(self, graph, id_map):
        with open('edge_remove.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                if len(row)!=4:
                    continue
                v1 = id_map[tuple([float(row[0]), float(row[1])])]
                v2 = id_map[tuple([float(row[2]), float(row[3])])]
                graph.remove_edge(v1, v2)

        with open('node_remove.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                if len(row)!=2:
                    continue
                vertex = tuple([float(row[0]), float(row[1])])
                if vertex in id_map.keys():
                    graph.remove_node(id_map[vertex])
                    id_map.pop(vertex, None)
                else:
                    print('Tried to remove non-existing node:', vertex)

        with open('node_add.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                if len(row)!=3:
                    continue
                vertex = tuple([float(row[0]), float(row[1])])
                stop = str(row[2])
                if vertex not in id_map.keys():
                    id_map[vertex] = self.node_index
                    self.node_index += 1
                    graph.add_node(id_map[vertex], vertex=vertex, stop=stop)
                else:
                    print('Tried to add already existing node')
                    print('Existing:',vertex, graph.nodes[id_map[vertex]]['stop'])
                    print('Tried:',vertex, stop)

        with open('edge_add.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                if len(row)!=5:
                    continue
                v1 = id_map[tuple([float(row[0]), float(row[1])])]
                v2 = id_map[tuple([float(row[2]), float(row[3])])]
                change = str(row[4])
                length = self.latlon_dist(graph.nodes[v1]['vertex'], graph.nodes[v2]['vertex'])
                graph.add_edge(v1, v2, length=length, change=change)


    def load_graph(self, graph, id_map, postfix, crossroad):
        if crossroad:
            cross_lines = []
            f = open('cross_node_v2.txt', 'rt')
            for l in f.read().splitlines():
                lat, lon = l.split(',')
                lat, lon = float(lat), float(lon)
                cross_lines.append((lat, lon))
            for vertex in cross_lines:
                if vertex not in id_map.keys():
                    id_map[vertex] = self.node_index
                    self.node_index += 1
                    graph.add_node(id_map[vertex], vertex=vertex, stop='CROSS')

        if postfix== 'A':
            stop_lines = []
            f = open('stop_node.txt', 'rt')
            for l in f.read().splitlines():
                lat, lon = l.split(',')
                lat, lon = float(lat), float(lon)
                stop_lines.append((lat, lon))
            for vertex in stop_lines:
                if vertex not in id_map.keys():
                    id_map[vertex] = self.node_index
                    self.node_index += 1
                    graph.add_node(id_map[vertex], vertex=vertex, stop='STOP')



        with open('node_'+str(postfix)+'.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                vertex = tuple([float(row[0]), float(row[1])])
                stop = str(row[2])
                if vertex not in id_map.keys():
                    id_map[vertex] = self.node_index
                    self.node_index += 1
                    graph.add_node(id_map[vertex], vertex=vertex, stop=stop)
                else:
                    print('Tried to add already existing node')
                    print('Existing:',vertex, graph.nodes[id_map[vertex]]['stop'])
                    print('Tried:',vertex, stop)

        with open('edge_'+str(postfix)+'.txt', 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(list(reader)):
                v1 = id_map[tuple([float(row[0]), float(row[1])])]
                v2 = id_map[tuple([float(row[2]), float(row[3])])]
                change = str(row[4])
                length = self.latlon_dist(graph.nodes[v1]['vertex'], graph.nodes[v2]['vertex'])
                graph.add_edge(v1, v2, length=length, change=change)

    def generate_path(self, source_index, target_index):
        route = nx.astar_path(self.G, source=source_index, target=target_index, heuristic=self.dist_heuristic, weight='length')
        route_list = []
        stops =[]
        changes = []
        dists = []
        print('ROUTE (list of node indices):')
        for i in range(len(route)):
            route_list.append([self.G.nodes[route[i]]['vertex'][0], self.G.nodes[route[i]]['vertex'][1]])
            stop = self.G.nodes[route[i]]['stop']
            stops.append(stop)
            change = self.G.edges[route[i-1], route[i]]['change'] if i>=1 else 'STRAIGHT'
            changes.append(change)
            dist = self.G.edges[route[i-1], route[i]]['length'] if i>=1 else 0.0
            dists.append(dist)
            #print(self.G.nodes[route[i]]['vertex'][0], self.G.nodes[route[i]]['vertex'][1], change, stop, dist)
            print('{},{}'.format(self.G.nodes[route[i]]['vertex'][0], self.G.nodes[route[i]]['vertex'][1]))
        return route_list, route, changes, stops, dists

    def distance_heuristic(self, G, n1, n2):
        node1 = np.array(G.nodes[n1]['vertex'])
        node2 = np.array(G.nodes[n2]['vertex'])
        return self.latlon_dist(node1, node2)

    def latlon_dist(self, src, dest):
        if len(np.shape(src)) == 1:
            lat1 = np.radians(src[0])
            lon1 = np.radians(src[1])
        else:
            lat1 = np.radians(src[:, 0])
            lon1 = np.radians(src[:, 1])
        if len(np.shape(dest)) == 1:
            lat2 = np.radians(dest[0])
            lon2 = np.radians(dest[1])
        else:
            lat2 = np.radians(dest[:, 0])
            lon2 = np.radians(dest[:, 1])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = 6378137 * c
        return distance

if __name__ == '__main__':
    PG = path_generator()
    target_pos = PG.gps_to_utm(36.012656362843, 129.324007392945, 0)
    print('target_pos: ', target_pos)
    min_idx = PG.get_closest_index(target_pos)
    print('min_idx: ', min_idx)

