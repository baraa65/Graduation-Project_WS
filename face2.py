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
import numpy as np
from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
preprocess_input = tensorflow.keras.applications.resnet50.preprocess_input

###
# from keras.models import load_model
# from skimage.transform import resize
# from imageio import imread


from face_detector import Detector

import time
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# import face_recognition as fr



detector = Detector()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# basemodel = tensorflow.keras.applications.ResNet50(weights='models/resnet50.h5', include_top=False, pooling="avg", input_shape=(224, 224, 3))
# model = tensorflow.keras.models.Model(inputs=basemodel.input, outputs=basemodel.output)

# face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# faceNet = load_model('models/facenet_keras.h5')
# cascade = cv2.CascadeClassifier('models/cascade.xml')

# def prewhiten(x):
#     if x.ndim == 4:
#         axis = (1, 2, 3)
#         size = x[0].size
#     elif x.ndim == 3:
#         axis = (0, 1, 2)
#         size = x.size
#     else:
#         raise ValueError('Dimension should be 3 or 4')
#
#     mean = np.mean(x, axis=axis, keepdims=True)
#     std = np.std(x, axis=axis, keepdims=True)
#     std_adj = np.maximum(std, 1.0/np.sqrt(size))
#     y = (x - mean) / std_adj
#     return y
#
# def l2_normalize(x, axis=-1, epsilon=1e-10):
#     output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
#     return output
#
#
# def load_and_align_images(filepaths, margin):
#     aligned_images = []
#     for filepath in filepaths:
#         start_time = time.time()
#         img = cv2.imread(filepath)
#
#         print("--- %s seconds --- (read)" % (time.time() - start_time))
#
#         start_time = time.time()
#
#         faces = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3)
#         (x, y, w, h) = faces[0]
#         cropped = img[y - margin // 2:y + h + margin // 2, x - margin // 2:x + w + margin // 2, :]
#         print("--- %s seconds --- (extract)" % (time.time() - start_time))
#
#         aligned = resize(cropped, (224, 224), mode='reflect')
#         aligned_images.append(aligned)
#
#     return np.array(aligned_images)
#
# def calc_embs(filepaths, margin=10, batch_size=1):
#     aligned_images = prewhiten(load_and_align_images(filepaths, margin))
#     pd = []
#     start_time = time.time()
#     for start in range(0, len(aligned_images), batch_size):
#         pd.append(model.predict_on_batch(aligned_images[start:start+batch_size]))
#
#     embs = l2_normalize(np.concatenate(pd))
#     print("--- %s seconds --- (emb)" % (time.time() - start_time))
#
#     return embs
#
# start_time = time.time()
#
# embs = calc_embs(['faces/baraa.jpg'])
# print(embs)
# print("--- %s seconds --- (total)" % (time.time() - start_time))
#

#####

def get_face_from_box(img, box, size=(224, 224)):
    p1, p2 = box
    x1, y1 = p1
    x2, y2 = p2
    if x1 < 0: x1 = 0
    if x2 < 0: x2 = 0
    if y1 < 0: y1 = 0
    if y2 < 0: y2 = 0
    print(f'x1: {x1},x2: {x2},y1: {y1},y2: {y2}')
    face = img[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)

    return face_array


def get_face(img, results, size):
    if len(results) == 0:
        return None

    x1, y1, x2, y2 = results[0]
    face = img[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return face_array


def extract_faces(frame, required_size=(224, 224)):
    detections = detector.processImage(frame)

    res = []

    for d in detections:
        face = get_face(frame, [d], required_size)
        x1, y1, x2, y2 = d
        box = ((x1, y1), (x2, y2))
        res.append((face, box))

    return res
    frame.flags.writeable = False
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    res = []

    if results.detections:
        for detection in results.detections:
            x1 = int(detection.location_data.relative_bounding_box.xmin * 480)
            x2 = int(x1 + detection.location_data.relative_bounding_box.width * 480)
            y1 = int(detection.location_data.relative_bounding_box.ymin * 360)
            y2 = int(y1 + detection.location_data.relative_bounding_box.height * 360)
            face = get_face(frame, [[x1, y1, x2, y2]], required_size)
            box = ((x1, y1), (x2, y2))
            res.append((face, box))

    return res


def get_embedding(file):
    start_time = time.time()

    faces = extract_faces(file)

    print("--- %s seconds --- (extract face)" % (time.time() - start_time))

    res = []

    for face, box in faces:
        if face is None:
            res.append(None)

        start_time = time.time()

        samples = asarray([face], 'float32')
        samples = preprocess_input(samples)
        yhat = model.predict(samples)
        # yhat = fr.face_encodings(face)

        # yhat = yhat[0] if len(yhat) > 0 else []

        # print(yhat, yhat2)

        print("--- %s seconds --- (model)" % (time.time() - start_time))

        res.append((yhat, box))

    return res

def get_emb_from_face(face):
    start_time = time.time()

    samples = asarray([face], 'float32')
    samples = preprocess_input(samples)
    yhat = model.predict(samples)

    print("--- %s seconds --- (model)" % (time.time() - start_time))
    return yhat


def is_match(known_embedding, candidate_embedding, thresh=0.5):
    score = cosine(known_embedding, candidate_embedding)

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
    print(name)
    faces_dict[name], box = get_embedding(img)[0]

def match_faces(img, faces):
    image = img
    embs = get_embedding(img)
    res = []

    for img_emb, box in embs:
        match = None
        min_score = 1

        for name, emb in faces:
            score = is_match(img_emb, emb)
            if score < 0.5 and score < min_score:
                match = name

        if match is None: match = 'Stranger'
        image = draw_box(image, box, match)
        res.append(match)

    return image, res

def only_boxes(img):
    image = img
    faces = extract_faces(img)

    for face, box in faces:
        image = draw_box(image, box, '')

    return image, []

def match(img):
    # return only_boxes(img)
    return match_faces(img, faces_dict.items())

def match_without_detection(img, boxes):
    faces = [(get_face_from_box(img, box), box) for box in boxes]
    embs = [(get_emb_from_face(face), box) for face, box in faces]

    image = img
    res = []

    for img_emb, box in embs:
        match = None
        min_score = 1

        for name, emb in faces_dict.items():
            score = is_match(img_emb, emb)
            if score < 0.5 and score < min_score:
                match = name

        if match is None: match = 'Stranger'
        image = draw_box(image, box, match)
        res.append(match)

    return image, res

def draw_box(img, box, name):
    p1, p2 = box
    img = cv2.rectangle(img, p1, p2, (255, 10, 10))
    if name != 'Stranger':
        cv2.putText(img, name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        cv2.putText(img, 'Stranger', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img

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

    boxes = []

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
            boxes.append((p1, p2))

            for id in new_ids:
                if id in person_ids:
                    break
                else:
                    person_ids.append(id)
                    check_face = True
                    print("=====================================================")



        if match is not None:
            cv2.putText(image, match[0], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(image, 'Stranger :  '+str(list(fa.keys())[0]), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    if results.detections is None:
        length = 0
    else:
        length = len(results.detections)


        # if results.detections:
        #     im = image[y1:y2, x1:x2]

    return image,check_face,length, boxes