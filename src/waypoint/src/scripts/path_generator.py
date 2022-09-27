import csv
import networkx as nx
import plotly.graph_objects as go
from geodesy.utm import fromLatLong as proj
import numpy as np
from functools import partial

class path_generator():
    def __init__(self):
        self.center = proj(36.01300, 129.3198799, 0)
        self.center = np.array([self.center.easting, self.center.northing, self.center.altitude])
        self.sampling_rate = 1  # Take every n-th nodes
        self.CONCATENATE_FIRST_NODE_AND_LAST_NODE = True
        self.index = []
        self.lat = []
        self.lon = []
        self.points = []
        # with open('gps_postech_new_bestpos.csv', newline='') as csvfile:

        with open('gps_postech_new_bestpos.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for i, row in enumerate(list(reader)[::self.sampling_rate]):
                # row[0]: index, row[1]: steer (degree), row[2]: speed (km/h),
                # row[3]: lat, row[4]: lon, row[5]: heading (degree)
                self.index.append(int(row[0]))
                self.lat.append(float(row[3]))
                self.lon.append(float(row[4]))
                self.points.append(self.gps_to_utm(float(row[3]), float(row[4]), 0))
                # points.append([float(row[3]), float(row[4])])
                # print(i)
        self.points = np.array(self.points)
        self.generate_graph()

    def get_points(self):
        return self.points

    def get_total_waypoints(self):
        return len(self.index)

    def gps_to_utm(self, latitude, longitude, altitude):
        pos = proj(latitude, longitude, altitude)
        pos = np.array([pos.easting, pos.northing, pos.altitude])
        pos[:2] -= self.center[:2]
        return pos

    def get_closest_index(self, target_pos):
        min_dist = 999
        min_idx = -1

        for i in range(len(self.points)):
            dist = np.linalg.norm(target_pos - self.points[i])
            if min_dist >= dist:
                min_idx = i + 1
                min_dist = dist
            # if min_idx != -1 and dist < 1:   # updated once
            #     break

        return min_idx


    def generate_graph(self):
        ##### NETWORKX GRAPH #####
        self.G = nx.DiGraph()
        for idx, latitude, longitude in zip(self.index, self.lat, self.lon):
            self.G.add_node(idx, vertex=(longitude, latitude), index=idx)

        for i in range(len(self.index)):
            if i == len(self.index) - 1:
                if self.CONCATENATE_FIRST_NODE_AND_LAST_NODE:
                    length = self.latlon_dist(self.G.nodes[self.index[i]]['vertex'], self.G.nodes[self.index[0]]['vertex'])
                    self.G.add_edge(self.index[i], self.index[0], length=length)
                break
            length = self.latlon_dist(self.G.nodes[self.index[i]]['vertex'], self.G.nodes[self.index[i + 1]]['vertex'])
            self.G.add_edge(self.index[i], self.index[i + 1], length=length)

        ##### EDGE #####
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = self.G.nodes[edge[0]]['vertex']
            x1, y1 = self.G.nodes[edge[1]]['vertex']
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        ##### NODE #####
        node_x = []
        node_y = []
        node_idx = []
        for node in self.G.nodes():
            x, y = self.G.nodes[node]['vertex']
            node_x.append(x)
            node_y.append(y)
            node_idx.append(self.G.nodes[node]['index'])

        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text')

        node_adjacencies = []
        node_trace.marker.color = node_adjacencies
        node_trace.text = ['NODE {}<br>{},{}'.format(idx, x, y) for idx, x, y in zip(node_idx, node_x, node_y)]

        self.dist_heuristic = partial(self.distance_heuristic, self.G)

    def generate_path(self, source_index, target_index):
        ##########################################################################################
        # route=nx.astar_path(G, source=1, target=2160, heuristic=dist_heuristic, weight='length')
        route = nx.astar_path(self.G, source=source_index, target=target_index, heuristic=self.dist_heuristic, weight='length')
        ##########################################################################################

        route_list = []
        print('ROUTE (list of node indices):', route)
        for i in range(len(route)):
            route_list.append([self.G.nodes[route[i]]['vertex'][0], self.G.nodes[route[i]]['vertex'][1]])
            # print(route[i], self.G.nodes[route[i]]['vertex'])
        return route_list

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
    PG.generate_path(2000, 10)
    target_pos = PG.gps_to_utm(36.012656362843, 129.324007392945, 0)
    print('target_pos: ', target_pos)
    min_idx = PG.get_closest_index(target_pos)
    print('min_idx: ', min_idx)

