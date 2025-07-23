from filterpy.kalman import KalmanFilter
import numpy as np
from scipy.spatial.distance import cdist

class Track:
    def __init__(self, point, track_id):
        self.kf = self.create_kalman_filter(point)
        self.track_id = track_id
        self.age = 0
        self.missed = 0
        self.trace = [point]

    def create_kalman_filter(self, init_point):
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.H = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0]])
        kf.P *= 100.
        kf.R *= 10
        kf.Q *= 1
        kf.x[:2] = np.reshape(init_point, (2, 1))
        return kf

    def predict(self):
        self.kf.predict()
        self.age += 1
        self.missed += 1
        pred = self.kf.x[:2].flatten()
        return pred

    def update(self, point):
        self.kf.update(point)
        self.trace.append(self.kf.x[:2].flatten())
        self.missed = 0


class PointTracker:
    def __init__(self, distance_threshold=30):
        self.prev_points = np.empty((0, 2))
        self.prev_ids = []
        self.next_id = 0
        self.distance_threshold = distance_threshold

    def update(self, current_points):
        current_points = np.array(current_points)
        assigned_ids = [-1] * len(current_points)

        if len(self.prev_points) == 0:
            # First frame or no previous points
            for i, pt in enumerate(current_points):
                assigned_ids[i] = self.next_id
                self.next_id += 1
        else:
            # Compute pairwise distances between current and previous points
            distances = cdist(current_points, self.prev_points)

            # For each current point, find the closest previous point
            used_prev = set()
            for i, dists in enumerate(distances):
                min_idx = np.argmin(dists)
                if dists[min_idx] < self.distance_threshold and min_idx not in used_prev:
                    assigned_ids[i] = self.prev_ids[min_idx]
                    used_prev.add(min_idx)
                else:
                    assigned_ids[i] = self.next_id
                    self.next_id += 1

        # Update state
        self.prev_points = current_points
        self.prev_ids = assigned_ids
        return list(zip(current_points.tolist(), assigned_ids))

# Example usage:
# tracker = PointTracker(distance_threshold=25)
# for frame in frames:
#     points, count = run_inference(model, img_tensor, args.threshold)
#     tracked = tracker.update(points)
#     print(tracked)  # [(x, y, id), ...]



# Inside the while loop, after 'points, count = run_inference(...)':

