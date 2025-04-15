import numpy as np
import networkx as nx
import heapq
from itertools import count
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, Delaunay
from shapely.geometry import Polygon, LineString

class SensorNetwork:
    def __init__(self, points_file=None, custom_point=None, points=None):
        """Initialize sensor network with points from file, array, or with an optional custom point."""
        if points is not None:
            # Use directly provided points array
            self.points = np.array(points)
        elif points_file is not None:
            # Load points from file
            self.points = np.genfromtxt(points_file, delimiter=',')
            if custom_point is not None:
                self.points = np.vstack((self.points, np.array([custom_point])))
        else:
            raise ValueError("Either points or points_file must be provided")
        
    def plot_network_connectivity(self, RC=0.25, save=True):
        """Plot connectivity between nodes within communication range."""
        plt.figure(figsize=(5, 5))
        plt.plot(self.points[:, 0], self.points[:, 1], 'bo', label='Nodes')
        
        for i in range(len(self.points)):
            for j in range(i + 1, len(self.points)):
                if np.linalg.norm(self.points[i] - self.points[j]) <= RC:
                    plt.plot([self.points[i, 0], self.points[j, 0]], 
                             [self.points[i, 1], self.points[j, 1]], 'r-', alpha=0.5)
        
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.xlabel('x-Coordinate (L)'); plt.ylabel('y-Coordinate (L)')
        plt.legend(); plt.title('Network Connectivity')
        
        if save: plt.savefig('network_connectivity.png')
        plt.show()

    def plot_network_coverage(self, RS=0.2, save=True):
        """Plot sensor coverage areas."""
        plt.figure(figsize=(5, 5))
        plt.plot(self.points[:, 0], self.points[:, 1], 'bo', label='Nodes')
        
        for i in range(len(self.points)):
            plt.gca().add_patch(plt.Circle((self.points[i, 0], self.points[i, 1]), 
                                           RS, color='g', fill=True, alpha=0.2))
        
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.xlabel('x-Coordinate (L)'); plt.ylabel('y-Coordinate (L)')
        plt.legend(); plt.title('Network Coverage')
        
        if save: plt.savefig('network_coverage.png')
        plt.show()
    
    def compute_voronoi(self):
        """Compute Voronoi diagram for the sensor points."""
        self.vor = Voronoi(self.points)
        self.regions, self.vertices, self.point_pairs_to_edges = self._create_bounded_voronoi()
        return self.regions, self.vertices, self.point_pairs_to_edges
    
    def compute_delaunay(self):
        """Compute Delaunay triangulation for the sensor points."""
        self.tri = Delaunay(self.points)
        return self.tri
    
    def _create_bounded_voronoi(self):
        """Create bounded Voronoi diagram within unit square."""
        def voronoi_finite_polygons_2d(vor, radius=None):
            """Get finite polygons from Voronoi diagram."""
            if vor.points.shape[1] != 2:
                raise ValueError("Requires 2D input")

            new_regions = []
            new_vertices = vor.vertices.tolist()
            center = vor.points.mean(axis=0)
            radius = radius or np.ptp(vor.points, axis=0).max() * 2

            for point_idx, region_idx in enumerate(vor.point_region):
                region = vor.regions[region_idx]
                
                if -1 in region:
                    ridges = []
                    for i, (p1, p2) in enumerate(vor.ridge_points):
                        if p1 == point_idx or p2 == point_idx:
                            p3 = p2 if p1 == point_idx else p1
                            ridge = vor.ridge_vertices[i]
                            
                            if -1 in ridge:
                                t = vor.points[p3] - vor.points[point_idx]
                                t /= np.linalg.norm(t)
                                t = np.array([-t[1], t[0]])
                                
                                midpoint = vor.points[[point_idx, p3]].mean(axis=0)
                                direction = np.sign(np.dot(midpoint - center, t)) * t * radius
                                
                                non_neg_idx = [idx for idx in ridge if idx >= 0][0]
                                far_point = vor.vertices[non_neg_idx] + direction
                                new_vertices.append(far_point.tolist())
                                ridge = [non_neg_idx, len(new_vertices) - 1]
                            ridges.append(ridge)
                    
                    region = list(set([v for ridge in ridges for v in ridge]))
                    
                    if len(region) >= 3:
                        vs = np.array([new_vertices[v] for v in region])
                        c = vs.mean(axis=0)
                        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
                        region = [region[i] for i in np.argsort(angles)]
                        new_regions.append(region)
                else:
                    new_regions.append(region)
                    
            return new_regions, np.array(new_vertices)

        regions, vertices = voronoi_finite_polygons_2d(self.vor)
        plot_bounds = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        clipped_regions = []
        clipped_vertices = []
        
        for region in regions:
            polygon = Polygon([vertices[i] for i in region])
            clipped_polygon = polygon.intersection(plot_bounds)
            
            if clipped_polygon.is_empty or clipped_polygon.area < 1e-10:
                continue
                
            if isinstance(clipped_polygon, Polygon):
                if len(list(clipped_polygon.exterior.coords)) <= 2:
                    continue
                    
                new_region = []
                for x, y in list(clipped_polygon.exterior.coords)[:-1]:
                    clipped_vertices.append([x, y])
                    new_region.append(len(clipped_vertices) - 1)
                
                if len(new_region) >= 3:
                    clipped_regions.append(new_region)
        
        regions = clipped_regions
        vertices = np.array(clipped_vertices)
        
        # Find shared edges between regions
        point_pairs_to_edges = {}
        for i in range(len(regions)):
            polygon1 = Polygon([vertices[v] for v in regions[i]])
            
            for j in range(i+1, len(regions)):
                polygon2 = Polygon([vertices[v] for v in regions[j]])
                
                try:
                    intersection = polygon1.intersection(polygon2)
                    
                    if isinstance(intersection, LineString) and not intersection.is_empty and intersection.length > 1e-10:
                        edge = list(intersection.coords)
                        if len(edge) >= 2:
                            point_pairs_to_edges[(i, j)] = edge
                    
                    elif hasattr(intersection, 'geoms'):
                        for line in intersection.geoms:
                            if isinstance(line, LineString) and not line.is_empty and line.length > 1e-10:
                                edge = list(line.coords)
                                if len(edge) >= 2:
                                    point_pairs_to_edges[(i, j)] = edge
                                    break
                except:
                    continue
        
        return regions, vertices, point_pairs_to_edges
    
    @staticmethod
    def _calculate_point_to_line_distance(point, line_start, line_end):
        """Calculate minimum distance from point to line segment."""
        point, line_start, line_end = map(np.array, [point, line_start, line_end])
        line_vector = line_end - line_start
        line_length = np.linalg.norm(line_vector)
        
        if line_length == 0:
            return np.linalg.norm(point - line_start)
        
        line_unit_vector = line_vector / line_length
        point_vector = point - line_start
        projection_length = np.dot(point_vector, line_unit_vector)
        
        if projection_length < 0:
            return np.linalg.norm(point - line_start)
        elif projection_length > line_length:
            return np.linalg.norm(point - line_end)
        
        return np.linalg.norm(point - (line_start + projection_length * line_unit_vector))
    
    def find_maximal_breach_path(self, save=True, connectivity_radius=0.25):
        """Find path from (0,0) to (1,1) that maximizes minimum distance to any sensor."""
        if not hasattr(self, 'regions') or not hasattr(self, 'vertices'):
            self.compute_voronoi()
            
        G = nx.Graph()
        
        # Add all Voronoi vertices to graph
        vertex_tuples = [tuple(v) for v in self.vertices]
        for v in vertex_tuples:
            G.add_node(v)
        
        # Add source and target nodes
        G.add_node((0, 0))
        G.add_node((1, 1))
        
        # Connect source and target to closest Voronoi vertices
        start_vertex = min(vertex_tuples, key=lambda x: np.linalg.norm(np.array(x) - np.array([0, 0])))
        end_vertex = min(vertex_tuples, key=lambda x: np.linalg.norm(np.array(x) - np.array([1, 1])))
        
        G.add_edge((0, 0), start_vertex, cost=0, weight=np.linalg.norm(np.array(start_vertex) - np.array([0, 0])))
        G.add_edge((1, 1), end_vertex, cost=0, weight=np.linalg.norm(np.array(end_vertex) - np.array([1, 1])))
        
        # Add edges between adjacent Voronoi vertices
        for region in self.regions:
            region_vertices = [tuple(self.vertices[j]) for j in region]
            
            for j in range(len(region_vertices)):
                v1 = region_vertices[j]
                v2 = region_vertices[(j+1) % len(region_vertices)]
                
                if not G.has_edge(v1, v2):
                    G.add_edge(v1, v2, cost=0, weight=np.linalg.norm(np.array(v1) - np.array(v2)))
        
        # Calculate cost of each edge (minimum distance to sensors)
        for (p1_idx, p2_idx), edge_points in self.point_pairs_to_edges.items():
            if p1_idx >= len(self.points) or p2_idx >= len(self.points):
                continue
            
            p1, p2 = self.points[p1_idx], self.points[p2_idx]
            
            for i in range(len(edge_points) - 1):
                v1, v2 = tuple(edge_points[i]), tuple(edge_points[i+1])
                
                if G.has_edge(v1, v2):
                    dist_p1 = self._calculate_point_to_line_distance(p1, edge_points[i], edge_points[i+1])
                    dist_p2 = self._calculate_point_to_line_distance(p2, edge_points[i], edge_points[i+1])
                    G[v1][v2]['cost'] = round(min(dist_p1, dist_p2) * 1000) / 1000
        
        # Set cost to 0 for boundary edges
        for u, v in G.edges():
            u_arr, v_arr = np.array(u), np.array(v)
            
            on_boundary_u = any(abs(u_arr[i] % 1) < 1e-5 for i in range(2))
            on_boundary_v = any(abs(v_arr[i] % 1) < 1e-5 for i in range(2))
            
            if on_boundary_u and on_boundary_v:
                G[u][v]['cost'] = 0
        
        # Find path that maximizes minimum distance to sensors
        shortest_distances = nx.single_source_dijkstra_path_length(G, (1, 1), weight='weight')
        
        mbp, max_cost = self._find_max_min_path(G, (0, 0), (1, 1), shortest_distances)
        
        # Plot the results
        self._plot_breach_path(G, mbp, max_cost, save, connectivity_radius)
        
        return mbp, max_cost
    
    def _find_max_min_path(self, graph, start, end, shortest_distances):
        """Find path that maximizes the minimum edge cost."""
        tiebreak = count()
        pq = [(0, 0, next(tiebreak), start, [start])]
        visited = set()
        
        while pq:
            min_cost, cum_dist, _, current, path = heapq.heappop(pq)
            min_cost = -min_cost
            
            if current == end:
                return path, min_cost
            
            if current in visited:
                continue
                
            visited.add(current)
            
            for neighbor in graph.neighbors(current):
                if neighbor not in visited and neighbor in shortest_distances:
                    edge_cost = graph[current][neighbor]['cost']
                    edge_weight = graph[current][neighbor]['weight']
                    new_min_cost = min(min_cost, edge_cost)
                    heapq.heappush(pq, (-new_min_cost, cum_dist + edge_weight, 
                                      next(tiebreak), neighbor, path + [neighbor]))
        
        return None, float('-inf')
    
    def _plot_breach_path(self, G, mbp, max_cost, save=True, connectivity_radius=0.25):
        """Plot the maximal breach path on Voronoi diagram."""
        plt.figure(figsize=(10, 10))
        
        # Plot Voronoi regions
        for region in self.regions:
            polygon = [self.vertices[i] for i in region]
            plt.fill(*zip(*polygon), alpha=0.1, edgecolor='orange', linewidth=1)
        
        # Plot graph edges
        for u, v, data in G.edges(data=True):
            if 'cost' in data:
                x1, y1 = u
                x2, y2 = v
                plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.2)
                
                if data['cost'] > 0.05:
                    plt.text((x1 + x2) / 2, (y1 + y2) / 2, f"{data['cost']:.3f}", 
                            ha='center', va='center', color='gray', fontsize=7, alpha=0.7)
        
        # Plot the breach path
        if mbp:
            points_array = np.array(mbp)
            plt.plot(points_array[:, 0], points_array[:, 1], 'r-', linewidth=2.5, 
                    label='Maximal Breach Path')
            
            for i in range(len(mbp)-1):
                u, v = mbp[i], mbp[i+1]
                if G.has_edge(u, v):
                    edge_cost = G[u][v]['cost']
                    midx, midy = (u[0] + v[0]) / 2, (u[1] + v[1]) / 2
                    plt.text(midx, midy, f"{edge_cost:.3f}", ha='center', va='center',
                            fontsize=9, color='red', fontweight='bold', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        plt.plot(self.points[:, 0], self.points[:, 1], 'bo', label='Sensors')
        plt.plot(0, 0, 'go', markersize=10, label='Start (0,0)')
        plt.plot(1, 1, 'mo', markersize=10, label='End (1,1)')
        
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.xlabel('x-Coordinate (L)', fontsize=12); plt.ylabel('y-Coordinate (L)', fontsize=12)
        plt.legend(fontsize=10)
        plt.title(f'Maximal Breach Path on Voronoi Diagram (RC={connectivity_radius:.2f})', fontsize=14)
        
        if save:
            plt.savefig('maximal_breach_path.png', bbox_inches='tight', dpi=300)
        plt.show()
    
    def find_maximal_support_path(self, save=True, coverage_radius=0.2):
        """Find path that maximizes the minimum distance traveled under sensor coverage."""
        if not hasattr(self, 'tri'):
            self.compute_delaunay()
            
        G = nx.Graph()
        
        # Add sensor nodes to graph
        for i, point in enumerate(self.points):
            G.add_node(i, pos=tuple(point))
        
        # Find closest sensors to start and end points
        start_idx = np.argmin(np.sum((self.points - np.array([0, 0]))**2, axis=1))
        end_idx = np.argmin(np.sum((self.points - np.array([1, 1]))**2, axis=1))
        
        # Add edges from Delaunay triangulation
        edges_seen = set()
        for simplex in self.tri.simplices:
            for i in range(3):
                p1_idx, p2_idx = simplex[i], simplex[(i+1)%3]
                
                edge = tuple(sorted([p1_idx, p2_idx]))
                if edge in edges_seen:
                    continue
                edges_seen.add(edge)
                
                p1, p2 = self.points[p1_idx], self.points[p2_idx]
                edge_length = np.linalg.norm(p1 - p2)
                midpoint = (p1 + p2) / 2
                
                basic_cost = edge_length / 2
                
                # Check if any other sensor is closer to midpoint
                is_boundary = any(np.linalg.norm(p - midpoint) < basic_cost 
                                for p in self.points if not np.array_equal(p, p1) and not np.array_equal(p, p2))
                
                if is_boundary:
                    # For boundary edges, sample points to find max-min distance
                    max_min_distance = max(
                        min(np.linalg.norm(p - (p1 + t * (p2 - p1))) for p in self.points)
                        for t in np.linspace(0, 1, 20)
                    )
                    cost = max_min_distance
                else:
                    cost = basic_cost
                    
                G.add_edge(p1_idx, p2_idx, cost=round(cost * 1000) / 1000, weight=round(cost * 1000) / 1000)
        
        # Find maximal support path
        msp_indices, max_cost = self._find_support_path(G, start_idx, end_idx)
        
        # Plot the results
        self._plot_support_path(G, msp_indices, max_cost, save, coverage_radius)
        
        return msp_indices, max_cost
    
    def _find_support_path(self, graph, start, end):
        """Find path that maximizes the minimum edge cost."""
        tiebreak = count()
        pq = [(0, 0, next(tiebreak), start, [start])]
        visited = {}
        
        while pq:
            max_cost, path_len, _, current, path = heapq.heappop(pq)
            
            if current == end:
                return path, max_cost
            
            if current in visited and visited[current] <= max_cost:
                continue
                
            visited[current] = max_cost
            
            for neighbor in graph.neighbors(current):
                if neighbor not in path:
                    edge_cost = graph[current][neighbor]['cost']
                    new_max_cost = max(max_cost, edge_cost)
                    heapq.heappush(pq, (new_max_cost, path_len + 1, 
                                      next(tiebreak), neighbor, path + [neighbor]))
        
        return None, float('inf')
    
    def _plot_support_path(self, G, msp_indices, max_cost, save=True, coverage_radius=0.2):
        """Plot the maximal support path on Delaunay triangulation."""
        if not hasattr(self, 'tri'):
            self.compute_delaunay()
            
        plt.figure(figsize=(10, 10))
        plt.triplot(self.points[:, 0], self.points[:, 1], self.tri.simplices, 'g-', alpha=0.2)
        
        if msp_indices:
            msp = [self.points[i] for i in msp_indices]
            msp_array = np.array(msp)
            plt.plot(msp_array[:, 0], msp_array[:, 1], 'r-', linewidth=2.5, 
                    label='Maximal Support Path')
            
            for i in range(len(msp_indices)-1):
                p1_idx, p2_idx = msp_indices[i], msp_indices[i+1]
                p1, p2 = self.points[p1_idx], self.points[p2_idx]
                midx, midy = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                
                cost = G[p1_idx][p2_idx]['cost']
                plt.text(midx, midy, f"{cost:.3f}", ha='center', va='center',
                        fontsize=9, color='red', fontweight='bold', 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        
        # Find start and end indices
        start_idx = np.argmin(np.sum((self.points - np.array([0, 0]))**2, axis=1))
        end_idx = np.argmin(np.sum((self.points - np.array([1, 1]))**2, axis=1))
        
        plt.plot(self.points[:, 0], self.points[:, 1], 'bo', label='Sensors')
        plt.plot(self.points[start_idx, 0], self.points[start_idx, 1], 'go', 
                markersize=10, label='Start')
        plt.plot(self.points[end_idx, 0], self.points[end_idx, 1], 'mo', 
                markersize=10, label='End')
        
        plt.xlim([0, 1]); plt.ylim([0, 1])
        plt.xlabel('x-Coordinate (L)', fontsize=12); plt.ylabel('y-Coordinate (L)', fontsize=12)
        plt.legend(fontsize=10)
        plt.title(f'Maximal Support Path on Delaunay Triangulation (RS={coverage_radius:.2f})', fontsize=14)
        
        if save:
            plt.savefig('maximal_support_path.png', bbox_inches='tight', dpi=300)
        plt.show()