import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment

class KalmanPointTracker(object):
    count = 0
    def __init__(self, initial_point):
        self.kf = KalmanFilter(dim_x=4, dim_z=2) 
        
        # State: [x, y, vx, vy]
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])

        self.kf.P *= 10.0
        self.kf.R *= 1.0   # Trust the detection position
        
        # --- STABILITY TUNING ---
        # Q: Process Noise. 
        # Low Q for velocity (indices 2,3) means "assume constant straight line speed"
        # High Q means "expect random turns"
        # We LOWER velocity noise to stop random jitter.
        self.kf.Q[-1, -1] *= 0.01 
        self.kf.Q[-2, -2] *= 0.01
        self.kf.Q[:2, :2] *= 0.05 

        self.kf.x[:2] = initial_point.reshape((2, 1))
        
        self.time_since_update = 0
        self.id = KalmanPointTracker.count
        KalmanPointTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, point):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(point)

    def predict(self):
        # --- YOUNG TRACK DAMPING ---
        # If the track is new (seen < 3 times) and disappears, 
        # assume it is NOT moving. This prevents noise from creating flying ghosts.
        if self.time_since_update > 0 and self.hits < 3:
            self.kf.x[2] = 0  # Kill X Velocity
            self.kf.x[3] = 0  # Kill Y Velocity

        # 1. Standard Prediction
        self.kf.predict()
        
        # --- VELOCITY CLAMPING ---
        # Prevent ghosts from accelerating to infinity.
        # Max speed: 20 pixels per frame (adjust if your people run very fast)
        max_speed = 20
        self.kf.x[2] = np.clip(self.kf.x[2], -max_speed, max_speed)
        self.kf.x[3] = np.clip(self.kf.x[3], -max_speed, max_speed)

        # --- FRICTION CONTROL ---
        # Apply decay to slow down ghosts over time
        if self.time_since_update > 0:
            self.kf.x[2] *= 0.9
            self.kf.x[3] *= 0.9
            
        self.age += 1
        if(self.time_since_update > 0):
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x[:2].reshape((1, 2))

    def get_state(self):
        return self.kf.x[:2].reshape((1, 2))

class SortPointTracker(object):
    def __init__(self, max_age=40, min_hits=1, distance_threshold=80):
        self.max_age = max_age
        self.min_hits = min_hits
        self.distance_threshold = distance_threshold
        self.trackers = []
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1
        
        # 1. Predict
        trks = np.zeros((len(self.trackers), 2))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()
            trks[t] = pos.flatten()
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # 2. Associate
        matched, unmatched_dets, unmatched_trks = self.associate(detections, trks)

        # 3. Update
        for t, d in matched:
            self.trackers[t].update(detections[d, :])

        # 4. Create New
        for i in unmatched_dets:
            trk = KalmanPointTracker(detections[i, :])
            self.trackers.append(trk)

        # 5. Output Logic
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            
            # Show if it's active or a valid ghost
            if (trk.time_since_update <= self.max_age) and (trk.hits >= self.min_hits):
                is_predicted = 1 if trk.time_since_update > 0 else 0
                ret.append(np.concatenate((d.flatten(), [trk.id, is_predicted])).reshape(1, -1)) 
            
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 4))

    def associate(self, detections, trackers):
        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        cost_matrix = np.zeros((len(trackers), len(detections)), dtype=np.float32)
        for t, trk in enumerate(trackers):
            for d, det in enumerate(detections):
                cost_matrix[t, d] = np.linalg.norm(trk - det)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_indices = np.stack((row_ind, col_ind), axis=1)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in col_ind):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in row_ind):
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if (cost_matrix[m[0], m[1]] > self.distance_threshold):
                unmatched_detections.append(m[1])
                unmatched_trackers.append(m[0])
            else:
                matches.append(m.reshape(1, 2))
        
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)