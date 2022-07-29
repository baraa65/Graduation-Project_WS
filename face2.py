import os
from os import listdir
from os.path import isfile, join
# import keras
# from matplotlib import pyplot
import tensorflow
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
# from mtcnn import MTCNN
import cv2

# from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input

from face_detector import Detector

import numpy as np
import time
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

detector = Detector()
# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

basemodel = tensorflow.keras.applications.ResNet50(weights='models/resnet50.h5', include_top=False, pooling="avg", input_shape=(224, 224, 3))
model = tensorflow.keras.models.Model(inputs=basemodel.input, outputs=basemodel.output)

def get_face(img, results, size):
    if len(results) == 0:
        return None

    x1, y1, x2, y2 = results[0]
    face = img[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return face_array


def extract_face(frame, required_size=(224, 224)):
    res = detector.processImage(frame)

    return get_face(frame, res, required_size)


def get_embedding(file):
    start_time = time.time()

    face = extract_face(file)

    print("--- %s seconds --- (extract face)" % (time.time() - start_time))

    if face is None:
        return None

    start_time = time.time()

    samples = asarray([face], 'float32')
    samples = preprocess_input(samples)
    yhat = model.predict(samples)
    print(yhat)

    print("--- %s seconds --- (model)" % (time.time() - start_time))

    return yhat


# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    start_time = time.time()

    score = cosine(known_embedding, candidate_embedding)

    print("--- %s seconds ---(cosine)" % (time.time() - start_time))

    if score <= thresh:
        print('>face is a Match (%.3f <= %.3f)' % (score, thresh))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (score, thresh))

    return score


currentPath = join(os.path.abspath(os.getcwd()), 'faces')

faces = [f for f in listdir(join(currentPath)) if isfile(join(currentPath, f))]

faces_dict = {}

for filename in faces:
    name = filename.split('.')[0]
    img = cv2.imread(join('faces', filename))
    faces_dict[name] = get_embedding(img)

# baraa = cv2.imread('baraa1.jpg')
# baraa = get_embedding(baraa)

# detector = MTCNN()


def match_face(img, emb):
    # results = detector.detect_faces(img)
    img_emb = get_embedding(img)

    # if len(results) > 0:
    #     box = results[0]['box']
    #     p1 = (box[0], box[1])
    #     p2 = (box[0] + box[2], box[1] + box[3])
    #     img = cv2.rectangle(img, p1, p2, (255, 10, 10))

    if img_emb is None:
        print('No Face !!')
        return None
    else:
        return is_match(img_emb, emb)


def match(img):
    for name, face in faces_dict.items():
        res = match_face(img, face)
        if res is None: continue

        if res <= 0.5:
            return name

    return None


def detect_faces(img, match):
    res = detector.processImage(img)

    if len(res) > 0:
        box = res[0]
        x1, y1, x2, y2 = box
        p1 = (x1, y1)
        p2 = (x2, y2)
        img = cv2.rectangle(img, p1, p2, (255, 10, 10))

        if match is not None:
            cv2.putText(img, match, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'Stranger', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img, len(res)

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
            joints = np.array([0, 1, 2])

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


face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
tracker = Tracker()
person_ids = []
def detect_faces_media(image, match):
    check_face = False
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
        for detection in results.detections:
            x1 = int(detection.location_data.relative_bounding_box.xmin*480)
            x2 = int(x1 + detection.location_data.relative_bounding_box.width*480)
            y1 = int(detection.location_data.relative_bounding_box.ymin*360)
            y2 = int(y1 + detection.location_data.relative_bounding_box.height*360)
            p1 = (x1, y1)
            p2 = (x2, y2)
            image = cv2.rectangle(image, p1, p2, (255, 10, 10))
            face = np.array([[[x1, y1], [x2, y2], [(x1 + x2) / 2, (y1 + y2) / 2]]])
            fa = tracker.track(face)
            new_ids = list(fa.keys())

            for id in new_ids:
                if id in person_ids:
                    break
                else:
                    person_ids.append(id)
                    check_face = True



        if match is not None:
            cv2.putText(image, match, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'Stranger :  '+str(list(fa.keys())[0]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    if results.detections is None:
        length = 0
    else:
        length = len(results.detections)


        # if results.detections:
        #     im = image[y1:y2, x1:x2]

    return image,check_face,length