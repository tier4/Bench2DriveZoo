"""
Planning metric utilities adapted from VAD's metric_stp3.py.
Computes collision metrics without mmcv dependencies.
"""

import numpy as np
import torch

ego_width, ego_length = 1.85, 4.084


def polygon_simple(r, c, shape=None):
    """
    Simple polygon rasterization without scikit-image.
    Returns row and column indices of pixels inside the polygon.
    
    Args:
        r: Row coordinates of polygon vertices
        c: Column coordinates of polygon vertices
        shape: Optional output shape for bounds checking
        
    Returns:
        rr, cc: Row and column indices of pixels inside polygon
    """
    # Simple implementation using point-in-polygon test
    # This is less efficient than skimage but works without the dependency
    
    if shape is not None:
        max_r, max_c = shape
    else:
        max_r = int(np.max(r)) + 1
        max_c = int(np.max(c)) + 1
    
    min_r = max(0, int(np.min(r)))
    min_c = max(0, int(np.min(c)))
    max_r = min(max_r, int(np.max(r)) + 1)
    max_c = min(max_c, int(np.max(c)) + 1)
    
    # Create mesh grid for the bounding box
    rr = []
    cc = []
    
    # Check each point in the bounding box
    for i in range(min_r, max_r):
        for j in range(min_c, max_c):
            # Point-in-polygon test using ray casting
            if point_in_polygon(i, j, r, c):
                rr.append(i)
                cc.append(j)
    
    return np.array(rr, dtype=np.int32), np.array(cc, dtype=np.int32)


def point_in_polygon(x, y, poly_x, poly_y):
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        x, y: Point coordinates
        poly_x, poly_y: Polygon vertex coordinates
        
    Returns:
        True if point is inside polygon
    """
    n = len(poly_x)
    inside = False
    
    p1x, p1y = poly_x[0], poly_y[0]
    for i in range(1, n + 1):
        p2x, p2y = poly_x[i % n], poly_y[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


class PlanningMetric:
    """
    Calculate planner metrics same as STP3/VAD.
    Adapted from mmcv/models/dense_heads/planning_head_plugin/metric_stp3.py
    """
    
    def __init__(self):
        super().__init__()
        self.X_BOUND = [-50.0, 50.0, 0.5]  # Forward
        self.Y_BOUND = [-50.0, 50.0, 0.5]  # Sides
        self.Z_BOUND = [-10.0, 10.0, 20.0]  # Height
        
        dx, bx, _ = self.gen_dx_bx(self.X_BOUND, self.Y_BOUND, self.Z_BOUND)
        self.dx, self.bx = dx[:2], bx[:2]
        
        bev_resolution, bev_start_position, bev_dimension = self.calculate_birds_eye_view_parameters(
            self.X_BOUND, self.Y_BOUND, self.Z_BOUND
        )
        self.bev_resolution = bev_resolution.numpy()
        self.bev_start_position = bev_start_position.numpy()
        self.bev_dimension = bev_dimension.numpy()
        
        self.W = ego_width
        self.H = ego_length
        
        self.category_index = {
            'human': [2, 3, 4, 5, 6, 7, 8],
            'vehicle': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        }
    
    def gen_dx_bx(self, xbound, ybound, zbound):
        dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
        bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
        nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
        return dx, bx, nx
    
    def calculate_birds_eye_view_parameters(self, x_bounds, y_bounds, z_bounds):
        """
        Calculate BEV parameters.
        
        Args:
            x_bounds: Forward direction in the ego-car
            y_bounds: Sides
            z_bounds: Height
            
        Returns:
            bev_resolution: Bird's-eye view resolution
            bev_start_position: Bird's-eye view first element
            bev_dimension: Bird's-eye view tensor spatial dimension
        """
        bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
        bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
        bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                    dtype=torch.long)
        return bev_resolution, bev_start_position, bev_dimension
    
    def get_label(self, gt_agent_states, gt_agent_labels, num_timesteps=6):
        """
        Generate occupancy grids from agent states.
        
        Args:
            gt_agent_states: (B, N, 5) array of (x, y, heading, length, width)
            gt_agent_labels: (B, N) boolean array indicating valid agents
            num_timesteps: Number of future timesteps
            
        Returns:
            segmentation: Occupancy grid for vehicles
            pedestrian: Occupancy grid for pedestrians
        """
        if gt_agent_states.ndim == 2:
            # Add batch dimension if missing
            gt_agent_states = gt_agent_states[np.newaxis, ...]
            gt_agent_labels = gt_agent_labels[np.newaxis, ...]
        
        B = gt_agent_states.shape[0]
        T = num_timesteps
        
        # Initialize occupancy grids
        segmentation = torch.zeros((B, T, int(self.bev_dimension[0]), int(self.bev_dimension[1])))
        pedestrian = torch.zeros((B, T, int(self.bev_dimension[0]), int(self.bev_dimension[1])))
        
        # For now, treat all agents as vehicles (simplified version)
        # In full implementation, would need agent type classification
        for b in range(B):
            for agent_idx in range(gt_agent_states.shape[1]):
                if not gt_agent_labels[b, agent_idx]:
                    continue
                
                x, y, heading, length, width = gt_agent_states[b, agent_idx]
                
                # Convert to BEV grid coordinates
                # Project agent box onto all future timesteps (static assumption)
                for t in range(T):
                    # Get corners of the bounding box
                    corners = self.get_agent_corners(x, y, heading, length, width)
                    
                    # Convert to pixel coordinates
                    pixel_corners = self.world_to_pixel(corners)
                    
                    # Draw polygon on occupancy grid
                    if pixel_corners is not None:
                        rr, cc = polygon_simple(pixel_corners[:, 0], pixel_corners[:, 1], shape=(int(self.bev_dimension[0]), int(self.bev_dimension[1])))
                        
                        # Clip to grid bounds
                        valid_mask = (rr >= 0) & (rr < self.bev_dimension[0]) & \
                                   (cc >= 0) & (cc < self.bev_dimension[1])
                        rr = rr[valid_mask]
                        cc = cc[valid_mask]
                        
                        # Mark as occupied
                        if len(rr) > 0:
                            segmentation[b, t, rr, cc] = 1
        
        return segmentation, pedestrian
    
    def get_agent_corners(self, x, y, heading, length, width):
        """
        Get four corners of an agent's bounding box.
        
        Args:
            x, y: Center position
            heading: Heading angle in radians
            length, width: Box dimensions
            
        Returns:
            corners: (4, 2) array of corner positions
        """
        # Half dimensions
        half_l = length / 2.0
        half_w = width / 2.0
        
        # Corners in agent's local frame
        corners_local = np.array([
            [half_l, half_w],    # front-left
            [half_l, -half_w],   # front-right
            [-half_l, -half_w],  # rear-right
            [-half_l, half_w]    # rear-left
        ])
        
        # Rotation matrix
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        
        # Transform to world coordinates
        corners_world = corners_local @ rot_matrix.T + np.array([x, y])
        
        return corners_world
    
    def world_to_pixel(self, points):
        """
        Convert world coordinates to pixel coordinates.
        
        Args:
            points: (N, 2) array of world coordinates
            
        Returns:
            pixels: (N, 2) array of pixel coordinates
        """
        # Convert to grid coordinates
        pixels = np.zeros_like(points)
        pixels[:, 0] = (points[:, 0] - self.bev_start_position[0]) / self.bev_resolution[0]
        pixels[:, 1] = (points[:, 1] - self.bev_start_position[1]) / self.bev_resolution[1]
        
        # Round to integer pixel indices
        pixels = np.round(pixels).astype(np.int32)
        
        return pixels
    
    def compute_L2(self, pred_traj, gt_traj):
        """
        Compute L2 distance between predicted and ground truth trajectories.
        
        Args:
            pred_traj: (T, 3) predicted trajectory (x, y, heading)
            gt_traj: (T, 3) ground truth trajectory
            
        Returns:
            l2_error: Average L2 distance
        """
        # Convert to numpy if needed
        if torch.is_tensor(pred_traj):
            pred_traj = pred_traj.cpu().numpy()
        if torch.is_tensor(gt_traj):
            gt_traj = gt_traj.cpu().numpy()
        
        # Compute L2 distance for x, y coordinates only
        l2_distances = np.linalg.norm(pred_traj[:, :2] - gt_traj[:, :2], axis=1)
        return np.mean(l2_distances)
    
    def evaluate_coll(self, trajs, gt_trajs, occupancy):
        """
        Evaluate collision between trajectories and occupancy grid.
        
        Args:
            trajs: (B, T, 3) predicted trajectories
            gt_trajs: (B, T, 3) ground truth trajectories (not used in collision)
            occupancy: (B, T, H, W) occupancy grids
            
        Returns:
            obj_coll: Object collision indicators
            obj_box_coll: Bounding box collision indicators
        """
        if torch.is_tensor(trajs):
            trajs = trajs.cpu().numpy()
        
        B, T, _ = trajs.shape
        
        # Initialize collision arrays
        obj_coll = np.zeros((B, T))
        obj_box_coll = np.zeros((B, T))
        
        for b in range(B):
            for t in range(T):
                # Get trajectory point
                x, y = trajs[b, t, 0], trajs[b, t, 1]
                
                # Convert to pixel coordinates
                px = int((x - self.bev_start_position[0]) / self.bev_resolution[0])
                py = int((y - self.bev_start_position[1]) / self.bev_resolution[1])
                
                # Check point collision
                if 0 <= px < self.bev_dimension[0] and 0 <= py < self.bev_dimension[1]:
                    if occupancy[b, t, px, py] > 0:
                        obj_coll[b, t] = 1
                
                # Check box collision (ego vehicle extent)
                # Get ego vehicle corners at this position
                corners = self.get_agent_corners(x, y, trajs[b, t, 2] if trajs.shape[2] > 2 else 0,
                                                self.H, self.W)
                pixel_corners = self.world_to_pixel(corners)
                
                # Check if any part of ego box overlaps with occupancy
                if pixel_corners is not None:
                    rr, cc = polygon_simple(pixel_corners[:, 0], pixel_corners[:, 1], shape=(int(self.bev_dimension[0]), int(self.bev_dimension[1])))
                    
                    # Clip to grid bounds
                    valid_mask = (rr >= 0) & (rr < self.bev_dimension[0]) & \
                               (cc >= 0) & (cc < self.bev_dimension[1])
                    rr = rr[valid_mask]
                    cc = cc[valid_mask]
                    
                    # Check for collision
                    if len(rr) > 0:
                        if torch.is_tensor(occupancy):
                            if torch.any(occupancy[b, t, rr, cc] > 0):
                                obj_box_coll[b, t] = 1
                        else:
                            if np.any(occupancy[b, t, rr, cc] > 0):
                                obj_box_coll[b, t] = 1
        
        return torch.tensor(obj_coll), torch.tensor(obj_box_coll)