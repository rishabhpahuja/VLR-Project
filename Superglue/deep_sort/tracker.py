# vim: expandtab:ts=4:sw=4
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track
import sys

# sys.path.insert(1,"/home/saharsh2/VLR-Project/Superglue")
sys.path.insert(2,"./models")
import Run_Superglue as sg
from Run_Superglue import SuperGlueClass
import ipdb


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """
# done
    def __init__(self, metric, max_iou_distance=0.7, max_age=75, n_init=1,max_sg_distance=0.4):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.max_sg_distance=max_sg_distance
        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.frame_t = None
        self.frame_t_1 = None
        self.sg_object=SuperGlueClass()

    def predict(self, prev_frame, frame):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        # import ipdb;ipdb.set_trace()
        self.frame_t = prev_frame # t
        self.frame_t_1 = frame # t+1

        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        #initially all unmatched - thus, unmatched detctions - [0, 1, 2, 3, 4, 5, 6]

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx]) # saare naye detection ka naya track and id created 
            # [0, 1, 2, 3, 4, 5, 6]

        UNMATCHED_TRACKS=[self.tracks[i] for i in unmatched_tracks]
        UNMACTHED_DETECTIONS=[detections[i] for i in unmatched_detections] # detctions ka object
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)
        
        return UNMATCHED_TRACKS,UNMACTHED_DETECTIONS

    def _match(self, detections):

        # ipdb.set_trace()

        def gated_metric(tracks, dets, track_indices, detection_indices):
            # ipdb.set_trace()
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        # ipdb.set_trace()
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        if False: #If True, normal cosine distance cascading matching will be done. If false, cascadign matchign will be done usign superglue
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade(gated_metric, self.metric.matching_threshold, self.max_age,
                    self.tracks, detections, confirmed_tracks) # sends gated_metric ka functions
        
        else:#Cascading matchign will be done using superglue
            # ipdb.set_trace()
            matches_a,unmatched_tracks_a,unmatched_detections=linear_assignment.matching_cascade_sg(\
                self.sg_object.Superglue_cost, self.max_sg_distance, self.max_age,self.tracks,
                detections, self.frame_t, self.frame_t_1, confirmed_tracks)
        
        if False: #If true superglue and cosine will be used both for cascade matching
            matches_a, unmatched_tracks_a, unmatched_detections = \
                linear_assignment.matching_cascade_using_two_metrics(gated_metric,sg.Superglue_cost,self.metric.matching_threshold, \
                self.max_sg_distance,self.max_age,self.tracks, detections, confirmed_tracks) # sends gated_metric ka functions
        
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        
        # ipdb.set_trace()
        if False: #if True, iou matching will be done usign superglue
            matches_c,unmatched_tracks_c,unmatched_detections=linear_assignment.min_cost_matching(\
                sg.Superglue_cost, self.max_sg_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
            
            matches=matches_a+matches_c
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_c))
        else: #IOU mtric will be used for IOU matching
            # ipdb.set_trace()
            matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(\
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)
        # checks for 1 after the other
            matches = matches_a + matches_b
            unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections
#done
    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        class_name = detection.get_class()
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, class_name)) # track gets initiated and appended 
        self._next_id += 1 # id updates taken - for each track is unique
