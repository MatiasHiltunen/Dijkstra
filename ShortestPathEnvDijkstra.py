import gymnasium as gym
import osmnx as ox
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os

class ShortestPathEnv(gym.Env):
    def __init__(self, random_start=False, random_end=False, plot_size=10, max_steps=1000, random_road_friction=False):
        super(ShortestPathEnv, self).__init__()
        graph_file = 'graph.graphml'
        if os.path.exists(graph_file):
            self.graph = ox.load_graphml(graph_file)
        else:
            self.graph = ox.graph_from_point((66.50845, 25.67410), dist=5000, network_type='drive')
            ox.save_graphml(self.graph, graph_file)
        
        # Remove nodes with no edges and which are not reachable
        self.graph.remove_nodes_from(list(nx.isolates(self.graph)))
        largest_cc = max(nx.strongly_connected_components(self.graph), key=len)
        self.graph = self.graph.subgraph(largest_cc).copy()
        self.nodes = list(self.graph.nodes)
        
        # Main parameters
        self.start_node = np.random.choice(self.nodes)
        self.end_node = np.random.choice(self.nodes)
        self.random_start = random_start
        self.random_end = random_end
        self.plot_size = plot_size
        self.max_steps = max_steps
        self.steps = 0
        self.random_road_friction = random_road_friction
        self.num_edges = len(self.graph.edges)

        # Apply random road friction to the edges
        self._apply_random_friction()

        # Add speeds
        self._add_speeds()

        # Add travel times
        #self._add_travel_times()

        # Observation space
        self.observation_space = gym.spaces.Dict({
            "node_features": gym.spaces.Box(low=0, high=len(self.nodes), shape=(3,), dtype=np.int32),
            "speed_limits": gym.spaces.Box(low=0, high=100, shape=(self.num_edges,), dtype=np.float32),
        })



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.distance_travelled = 0
        self.time_travelled = 0
        self.steps = 0
        if self.random_start:
            self.start_node = np.random.choice(self.nodes)
        if self.random_end:
            self.end_node = np.random.choice(self.nodes)

        if self.random_road_friction:
            self._apply_random_friction

        self.current_node = self.start_node
        self.path = [(self.graph.nodes[self.start_node]['x'], self.graph.nodes[self.start_node]['y'])]

        observation, possible_actions = self._get_observation()
        self.possible_actions = possible_actions
        return observation, {"possible_actions": possible_actions, "finished": False}

    def step(self, action):
        self.steps += 1
        
        # Map the chosen action to the actual next node
        next_node = action

        # Get the edge data for the chosen action
        edge_data = self.graph[self.current_node][next_node][0]
        length = edge_data['length']
        speed_limit = self._get_speed_limit(edge_data)
        
        # Calculate travel time for this step
        time_taken = (length / 1000) / speed_limit # in hours
        
        # Update total distance and time traveled
        self.distance_travelled += length
        self.time_travelled += time_taken
        
        # Calculate progress toward the goal
        self.current_node = next_node

        # Combine rewards
        reward = 0
        
        # Check if the agent has reached the goal
        done = self.current_node == self.end_node
        if done:
            # Completion bonus: inversely proportional to time taken
            completion_bonus = 1 / (self.time_travelled + 1)
            reward += completion_bonus
        
        # Append the edge and current node to the path
        if 'geometry' in edge_data:
            self.path.extend(list(edge_data['geometry'].coords))
        self.path.append((self.graph.nodes[next_node]['x'], self.graph.nodes[next_node]['y']))
        
        # Check if the episode has exceeded the maximum number of steps
        truncated = self.steps >= self.max_steps
        if truncated:
            reward = -10  # Penalty for exceeding max steps
            done = True
        
        # Get the new observation and possible actions
        observation, possible_actions = self._get_observation()
        self.possible_actions = possible_actions
        
        return observation, reward, done, truncated, {"possible_actions": possible_actions, "finished": done and not truncated}

    def render(self, mode='human'):

        # color the edges based on friction (default value is #999999)
        edge_colors = []
        for edge in self.graph.edges:
            edge_data = self.graph[edge[0]][edge[1]][0]
            if 'friction' in edge_data:
                friction = edge_data['friction']
                if friction > 0.4:
                    edge_colors.append('#a2cf97')
                elif friction > 0.2:
                    edge_colors.append('#e8e0be')
                else:
                    edge_colors.append('#f2bebd')
            else:
                edge_colors.append('#999999')

        fig, ax = ox.plot_graph(self.graph, show=False, close=False, figsize=(self.plot_size, self.plot_size), edge_color=edge_colors)
        start_x, start_y = self.graph.nodes[self.start_node]['x'], self.graph.nodes[self.start_node]['y']
        end_x, end_y = self.graph.nodes[self.end_node]['x'], self.graph.nodes[self.end_node]['y']
        current_x, current_y = self.graph.nodes[self.current_node]['x'], self.graph.nodes[self.current_node]['y']
        ax.scatter([start_x], [start_y], c='red', s=100, label='Start')
        ax.scatter([end_x], [end_y], c='blue', s=100, label='End')
        ax.scatter([current_x], [current_y], c='pink', s=100, label='Current')

        if len(self.path) > 1:
            #print(self.path )
            path_x, path_y = zip(*self.path)
            ax.plot(path_x, path_y, c='green', linewidth=3, label='Path')
        plt.legend()
        plt.show()

    def _get_observation(self):
        node_features = np.zeros(3, dtype=np.int32)
        # start, current, end
        node_features[0] = self.nodes.index(self.start_node)
        node_features[1] = self.nodes.index(self.current_node)
        node_features[2] = self.nodes.index(self.end_node)

        speed_limits = []
        for edge in self.graph.edges:
            edge_data = self.graph[edge[0]][edge[1]][0]
            speed_limit = self._get_speed_limit(edge_data)
            speed_limits.append(speed_limit)

        return {
            "node_features": node_features,
            "speed_limits": np.array(speed_limits, dtype=np.float32),
        }

    def _get_speed_limit(self, edge_data):
        if 'maxspeed' in edge_data:
            speed_limit = int(edge_data['maxspeed'][0]) if isinstance(edge_data['maxspeed'], list) else int(edge_data['maxspeed'])
        elif edge_data['highway'] == 'residential':
            speed_limit = 30
        elif edge_data['highway'] == 'secondary':
            speed_limit = 50
        elif edge_data['highway'] == 'tertiary':
            speed_limit = 80
        elif edge_data['highway'] == 'primary':
            speed_limit = 100
        else:
            speed_limit = 50
        
        if 'friction' in edge_data:
            if edge_data['friction'] > 0.4:
                pass
            elif edge_data['friction'] > 0.2:
                speed_limit *= 0.5
            else:
                speed_limit *= 0.25

        return speed_limit
    
    def _apply_random_friction(self):
        for edge in self.graph.edges:
            edge_data = self.graph[edge[0]][edge[1]][0]
            edge_data['friction'] = np.random.uniform(0.1, 0.8)
    
    def _add_speeds(self):
        for edge in self.graph.edges:
            edge_data = self.graph[edge[0]][edge[1]]
            for connection in edge_data:
                edge_data[connection]['speed_kph'] = self._get_speed_limit(edge_data[connection])
    
    def _add_travel_times(self):
        ox.add_edge_travel_times(self.graph)
        ox.plot.get_edge_colors_by_attr(self.graph, 'travel_time', cmap='viridis', num_bins=5)

    def get_end_node(self):
        return self.nodes.index(self.end_node)
    
    def get_shortest_path(self, mode='distance'):
        '''
        Get the shortest path from the current node to the end node
        mode: str, 'distance' or 'travel_time'
        '''
        if mode == 'distance':
            return nx.shortest_path(self.graph, self.current_node, self.end_node, weight='length')
        elif mode == 'travel_time':
            return nx.shortest_path(self.graph, self.current_node, self.end_node, weight='travel_time')
        else: 
            raise ValueError("Invalid mode. Choose 'distance' or 'travel_time'.")
            