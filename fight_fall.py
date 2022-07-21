import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import cv2
import time
import functools
import joblib
#############################
from numpy import mean
from numpy import std
from numpy import dstack
from keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import ConvLSTM2D
#from keras.utils import to_categorical
from tensorflow.keras.models import load_model

embed = hub.load(r'movenet')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
movenet = embed.signatures['serving_default']


def clean_unwanted_detections(frame, keypoints_with_scores, confidence_threshold):
    valid_detections = []
    y, x, c = frame.shape
    for person in keypoints_with_scores:
        if (person[:, 2] > confidence_threshold).sum() > 12:
            shaped = np.squeeze(np.multiply(person, [y, x, 1]))
            valid_detections.append(shaped)

    return valid_detections


EDGES = {
    (0, 1): (255, 255, 255),
    (0, 2): (0, 255, 0),
    (1, 3): (0, 0, 255),
    (2, 4): (255, 0, 255),
    (0, 5): (218, 28, 255),
    (0, 6): (94, 34, 178),
    (5, 7): (212, 33, 72),
    (7, 9): (202, 84, 26),
    (6, 8): (215, 145, 28),
    (8, 10): (205, 189, 26),
    (5, 6): (144, 181, 22),
    (5, 11): (97, 189, 24),
    (6, 12): (49, 184, 27),
    (11, 12): (58, 143, 181),
    (11, 13): (57, 117, 182),
    (13, 15): (59, 43, 185),
    (12, 14): (53, 194, 132),
    (14, 16): (59, 193, 184)
}

KP_COLORS = [
    (255, 255, 255),  # nose
    (0, 255, 0),  # right_eye
    (0, 0, 255),  # left_eye
    (255, 0, 0),  # right_ear
    (218, 28, 91),
    (94, 34, 178),
    (212, 33, 72),
    (202, 84, 26),
    (215, 145, 28),
    (205, 189, 26),
    (144, 181, 22),
    (97, 189, 24),
    (49, 184, 27),
    (58, 143, 181),
    (57, 117, 182),
    (59, 43, 185),
    (53, 194, 132),
    (59, 193, 184)

]


def loop_through_people1(frame, keypoints_with_scores, person_label, edges):
    i = 0
    for person in keypoints_with_scores:
        draw_connections1(frame, person, person_label, edges, i)
        draw_keypoints1(frame, person)
        i = i + 1


def draw_connections1(frame, keypoints, person_label, edges, person_id):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [1, 1]))
    i = 0
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1 = shaped[p1]
        y2, x2, = shaped[p2]
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        if i == 0:
            cv2.putText(frame, str(person_label[person_id]), (int(x1 + 10), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)
            i = 1


def draw_keypoints1(frame, keypoints):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [1, 1]))
    i = 0
    for kp in shaped:
        ky, kx = kp
        cv2.circle(frame, (int(kx), int(ky)), 3, KP_COLORS[i], -1)
        i = i + 1


def calc_angles(p0, p1, p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)

    #     print(np.linalg.det([v0,v1]))
    #     print(np.dot(v0,v1))
    angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))

    return np.degrees(angle)


def calc_dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class Tracker(object):
    def __init__(self, dist_thresh=50, max_humans=6):
        self._dist_thresh = dist_thresh
        self._max_humans = max_humans

        self._dict_id2skeleton = {}
        self._cnt_humans = 0

    def track(self, curr_skels):
        #         curr_skels = self._sort_skeletons_by_dist_to_center(curr_skels) #########
        N = len(curr_skels)
        # Match skeletons between curr and prev
        if len(self._dict_id2skeleton) > 0:
            ids, prev_skels = map(list, zip(*self._dict_id2skeleton.items()))
            good_matches = self._match_features(prev_skels, curr_skels)

            self._dict_id2skeleton = {}
            is_matched = [False] * N
            for i2, i1 in good_matches.items():
                human_id = ids[i1]
                self._dict_id2skeleton[human_id] = np.array(curr_skels[i2])
                is_matched[i2] = True
            unmatched_idx = [i for i, matched in enumerate(
                is_matched) if not matched]
        else:
            good_matches = []
            unmatched_idx = range(N)

        # Add unmatched skeletons (which are new skeletons) to the list
        num_humans_to_add = min(len(unmatched_idx),
                                self._max_humans - len(good_matches))
        #         print("num:",num_humans_to_add)
        for i in range(num_humans_to_add):
            self._cnt_humans += 1
            self._dict_id2skeleton[self._cnt_humans] = np.array(
                curr_skels[unmatched_idx[i]])
        #         print(self._dict_id2skeleton)
        return self._dict_id2skeleton

    def _get_neck(self, skeleton):
        x, y = skeleton[2], skeleton[3]
        return x, y

    def _match_features(self, features1, features2):
        features1, features2 = np.array(features1), np.array(features2)

        cost = lambda x1, x2: np.linalg.norm(x1 - x2)

        def calc_dist(p1, p2):
            return (
                           (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

        def cost(sk1, sk2):

            # neck, shoulder, elbow, hip, knee
            #             joints = np.array([2, 3, 4, 5, 6, 7, 10, 11, 12,  ################################
            #                                13, 16, 14, 14, 14, 14, 14, 14, 14])

            #             joints = np.array([2, 3, 4, 5, 6, 7,14])
            joints = np.array([2, 3, 4, 5, 6, 7, 14, 8, 9, 10, 11])

            #             print("joints:",sk1[joints])
            sk1, sk2 = sk1[joints], sk2[joints]
            valid_idx = np.logical_and(sk1 != 0, sk2 != 0)
            sk1, sk2 = sk1[valid_idx], sk2[valid_idx]
            sum_dist, num_points = 0, int(len(sk1) / 2)
            if num_points == 0:
                return 99999
            else:
                for i in range(num_points):  # compute distance between each pair of joint
                    idx = i * 2
                    sum_dist += calc_dist(sk1[idx:idx + 2], sk2[idx:idx + 2])
                mean_dist = sum_dist / num_points
                mean_dist /= (1.0 + 0.05 * num_points)  # more points, the better
                return mean_dist

        # If f1i is matched to f2j and vice versa, the match is good.
        good_matches = {}
        n1, n2 = len(features1), len(features2)
        if n1 and n2:

            # dist_matrix[i][j] is the distance between features[i] and features[j]
            dist_matrix = [[cost(f1, f2) for f2 in features2]
                           for f1 in features1]
            dist_matrix = np.array(dist_matrix)

            # Find the match of features1[i]
            matches_f1_to_f2 = [dist_matrix[row, :].argmin()
                                for row in range(n1)]

            # Find the match of features2[i]
            matches_f2_to_f1 = [dist_matrix[:, col].argmin()
                                for col in range(n2)]

            for i1, i2 in enumerate(matches_f1_to_f2):
                if matches_f2_to_f1[i2] == i1 and dist_matrix[i1, i2] < self._dist_thresh:
                    good_matches[i2] = i1
            #                     print("goooooog")

            if 0:
                print("distance matrix:", dist_matrix)
                print("matches_f1_to_f2:", matches_f1_to_f2)
                print("matches_f1_to_f2:", matches_f2_to_f1)
                print("good_matches:", good_matches)

        return good_matches

HEAD = 0
LEFT_HIP = 12
RIGHT_HIP = 11
LEFT_ANKLE = 16
RIGHT_ANKLE = 15
LEFT_KNEE = 14
RIGHT_KNEE = 13
LEFT_SHOULDER = 6
RIGHT_SHOULDER = 5
LEFT_ELBOW = 8
RIGHT_ELBOW = 7
LEFT_WRIST = 10
RIGHT_WRIST = 9
########
FALL_FETURES = 0
FIGHT_FETURES = 1

w = 1
h = 1


def get_fetures_fight(lm, prev_points):
    head = lm[HEAD]
    lhip = lm[LEFT_HIP]
    rhip = lm[RIGHT_HIP]
    lankle = lm[LEFT_ANKLE]
    rankle = lm[RIGHT_ANKLE]
    lknee = lm[LEFT_KNEE]
    rknee = lm[RIGHT_KNEE]

    lshoulder = lm[LEFT_SHOULDER]
    rshoulder = lm[RIGHT_SHOULDER]
    lelbow = lm[LEFT_ELBOW]
    relbow = lm[RIGHT_ELBOW]
    lwrist = lm[LEFT_WRIST]
    rwrist = lm[RIGHT_WRIST]

    #     #all points in legs and arms
    l_legs = [[lshoulder[0] * w, lshoulder[1] * h], [lknee[0] * w, lknee[1] * h], [lhip[0] * w, lhip[1] * h],
              [lankle[0] * w, lankle[1] * h]]
    r_legs = [[rshoulder[0] * w, rshoulder[1] * h], [rknee[0] * w, rknee[1] * h], [rhip[0] * w, rhip[1] * h],
              [rankle[0] * w, rankle[1] * h]]

    l_arm = [[lshoulder[0] * w, lshoulder[1] * h], [lelbow[0] * w, lelbow[1] * h], [lwrist[0] * w, lwrist[1] * h]]
    r_arm = [[rshoulder[0] * w, rshoulder[1] * h], [relbow[0] * w, lelbow[1] * h], [rwrist[0] * w, rwrist[1] * h]]

    #     # for calc angles when leg moves
    l_sh_hip_knee = [[lshoulder[0] * w, lshoulder[1] * h], [lhip[0] * w, lhip[1] * h], [lknee[0] * w, lknee[1] * h]]
    r_sh_hip_knee = [[rshoulder[0] * w, rshoulder[1] * h], [rhip[0] * w, rhip[1] * h], [rknee[0] * w, rknee[1] * h]]

    #     ##**calc fetures**

    ##1)distance between left ankle and right knee
    ankle_dist = calc_dist([rankle[0] * w, rankle[1] * h], [lankle[0] * w, lankle[1] * h])

    # 2)calc angles between prev and current ankle
    left_ankle_angle = calc_angles(l_legs[3], l_legs[2], prev_points[2])
    prev_points[2] = l_legs[3]

    right_ankle_angle = calc_angles(r_legs[3], r_legs[2], prev_points[3])
    prev_points[3] = r_legs[3]

    ##3)wrist dist
    left_wrist_dist = calc_dist(l_arm[2], prev_points[0])
    right_wrist_dist = calc_dist(r_arm[2], prev_points[1])

    ##4)dist angles
    left_wrist_angle = calc_angles(l_arm[2], l_arm[0], prev_points[0])
    right_wrist_angle = calc_angles(r_arm[2], r_arm[0], prev_points[1])

    prev_points[0] = l_arm[2]
    prev_points[1] = r_arm[2]

    return left_wrist_dist, right_wrist_dist, left_wrist_angle, right_wrist_angle, ankle_dist, left_ankle_angle, right_ankle_angle, prev_points


w = 1
h = 1


def get_fetures_fall(lm, prevhead, frame):
    head = lm[HEAD]
    lhip = lm[LEFT_HIP]
    rhip = lm[RIGHT_HIP]
    lshoulder = lm[LEFT_SHOULDER]
    rshoulder = lm[RIGHT_SHOULDER]

    if (prevhead[0] == 0) & (prevhead[1] == 0):
        prevhead = [head[0] * w, head[1] * h]

    x = lshoulder[1] * h
    y = lshoulder[0] * w
    x1 = lhip[1] * h
    y1 = lhip[0] * w

    head_dist = calc_dist([head[1] * h, head[0] * w], prevhead)
    ratio = abs((y - y1) / (x - x1))
    prev_head = [head[1] * h, head[0] * w]

    return prev_head, ratio, head_dist, frame


class Fetures_Proc:

    def __init__(self, max_humans=6):
        self._max_humans = max_humans
        self._dict_skeleton_fall = {}
        self._dict_skeleton_fight = {}
        self._cnt_humans = 0

    def track_fetures(self, key, fetures, mode):

        if mode == FALL_FETURES:
            if key in self._dict_skeleton_fall:
                self._dict_skeleton_fall[key].append(fetures)
            else:
                self._dict_skeleton_fall[key] = [fetures]

        else:
            if key in self._dict_skeleton_fight:
                self._dict_skeleton_fight[key].append(fetures)
            else:
                self._dict_skeleton_fight[key] = [fetures]

    def get_frames_number(self, key, mode):
        if mode == FALL_FETURES:
            return len(self._dict_skeleton_fall[key])

        return len(self._dict_skeleton_fight[key])

    def reset_fetures(self, key, mode):
        if mode == FALL_FETURES:
            self._dict_skeleton_fall[key] = self._dict_skeleton_fall[key][0:15]
        else:
            self._dict_skeleton_fight[key] = self._dict_skeleton_fight[key][0:5]

    def get_feture_vector(self, key, mode):
        if mode == FALL_FETURES:
            return self._dict_skeleton_fall[key]

        return self._dict_skeleton_fight[key]

    def calc_dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def check_fight_distance(self, key, persons):
        keys = list(persons.keys())
        phead = persons[key][HEAD]
        keys.remove(key)
        for k in keys:
            dist = calc_dist(persons[k][HEAD], phead)
            if dist < 50:
                return True

        return False



#load falling model
fall_model = load_model('fall_model1.h5')
#load falling scalar
fall_scaller=joblib.load('fall_scaler.bin')
#load fight model
fight_model = load_model('allModelv2.h5')
#load fight scallar
fight_scaller = joblib.load('fight_scaler.bin')


class Fight_Fall_detector:
    def __init__(self, dst_thresh=60, con_thresh=0.20):
        self.fps_time = 0
        self.prev_head = [0, 0]
        self.prev_points = [[0, 0], [0, 0], [0, 0], [0, 0]]
        self.tracker = Tracker(dist_thresh=dst_thresh)
        self.feture_tracker = Fetures_Proc()
        self.conf_thresh = con_thresh
        # load falling model
        self.fall_model = load_model('fall_model1.h5')
        # load falling scalar
        self.fall_scaller = joblib.load('fall_scaler.bin')
        # load fight model
        self.fight_model = load_model('allModelv2.h5')
        # load fight scallar
        self.fight_scaller = joblib.load('fight_scaler.bin')

    def detect(self, frame):
        frame = cv2.resize(frame, (480, 359))

        with tf.device('/GPU:1'):
            img = frame.copy()
            img = tf.image.resize_with_pad(tf.expand_dims(img, axis=0), 352, 480)
            input_img = tf.cast(img, dtype=tf.int32)

            # Detection section
            results = movenet(input_img)
            keypoints_with_scores = results['output_0'].numpy()[:, :, :51].reshape((6, 17, 3))
            # clean bad selection
            persons = clean_unwanted_detections(frame, keypoints_with_scores, self.conf_thresh)

            if len(persons) > 0:
                ############################### Tracker
                persons1 = []
                for i in range(len(persons)):
                    persons1.append(np.delete(persons[i], np.s_[2:3], axis=1))

                pe = self.tracker.track(persons1[:][:16])
                loop_through_people1(frame, np.array(list(pe.values())), list(pe.keys()), EDGES)

                ###############################end of tracker

                # for every person in the frame get fetures and predict action
                for key in list(pe.keys()):
                    ################## Fight detection every 10 frames
                    left_wrist_dist, right_wrist_dist, left_wrist_angle, right_wrist_angle, \
                    ankle_dist, left_ankle_angle, right_ankle_angle, self.prev_points = get_fetures_fight(pe[key],
                                                                                                          self.prev_points)

                    self.feture_tracker.track_fetures(key, fetures=[left_wrist_dist, right_wrist_dist, left_wrist_angle,
                                                                    right_wrist_angle, \
                                                                    ankle_dist, left_ankle_angle, right_ankle_angle],
                                                      mode=FIGHT_FETURES)

                    frame_count = self.feture_tracker.get_frames_number(key, FIGHT_FETURES)

                    if frame_count == 10:

                        frames_fet = self.feture_tracker.get_feture_vector(key, FIGHT_FETURES)
                        frames_fet = np.asarray(frames_fet, dtype=np.float32)
                        frames_fet = np.reshape(frames_fet, (1, 10, 7))

                        self.feture_tracker.reset_fetures(key, FIGHT_FETURES)

                        # normalize
                        for i in range(frames_fet.shape[1]):
                            frames_fet[:, i, :] = self.fight_scaller[i].transform(frames_fet[:, i, :])

                        label = self.fight_model.predict(frames_fet)

                        if (label[0][0] > label[0][1]) & (label[0][0] > 0.80):
                            if self.feture_tracker.check_fight_distance(key, pe):
                                print(f"person {key} : Kick")
                        elif (label[0][2] > label[0][1]) & (label[0][2] > 0.85):
                            if self.feture_tracker.check_fight_distance(key, pe):
                                print(f"person {key} : punch")
                        else:
                            print(f"person {key} : normal")

                    ##################
                    ############################### fall detection
                    self.prev_head, ratio, head_dist, _ = get_fetures_fall(pe[key], self.prev_head, frame)

                    self.feture_tracker.track_fetures(key, fetures=[ratio, head_dist], mode=FALL_FETURES)
                    frame_count = self.feture_tracker.get_frames_number(key, FALL_FETURES)
                    if frame_count == 32:

                        frames_fet_fall_copy = self.feture_tracker.get_feture_vector(key, FALL_FETURES)
                        frames_fet_fall_copy = np.asarray(frames_fet_fall_copy, dtype=np.float32)
                        frames_fet_fall_copy = np.reshape(frames_fet_fall_copy, (1, 32, 2))

                        self.feture_tracker.reset_fetures(key, FALL_FETURES)

                        for i in range(frames_fet_fall_copy.shape[1]):
                            frames_fet_fall_copy[:, i, :] = self.fall_scaller[i].transform(
                                frames_fet_fall_copy[:, i, :])

                        label = self.fall_model.predict(frames_fet_fall_copy)

                        if (label[0][0] > label[0][1]) & (ratio < 1) & (label[0][0] > 0.80):
                            print(f"person {key} : fall")
                        else:
                            print(f"person {key} : normal_no_fall")

                ###############################end of fall

            cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - self.fps_time)), (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.fps_time = time.time()

            cv2.imshow('Movenet Multipose', frame)

ffmodel = Fight_Fall_detector()

def fall_fight_model():
    return ffmodel
