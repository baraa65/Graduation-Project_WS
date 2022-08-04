import os
from os import listdir
from os.path import isfile, join
import tensorflow
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
import cv2
from keras_vggface.vggface import VGGFace
from tracker import Tracker

preprocess_input = tensorflow.keras.applications.resnet50.preprocess_input

from face_detector import Detector

import time
import numpy as np
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

detector = Detector()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
faces_dict = {}

def get_face_from_box(img, box, size=(224, 224)):
    p1, p2 = box
    x1, y1 = p1
    x2, y2 = p2
    if x1 < 0: x1 = 0
    if x2 < 0: x2 = 0
    if y1 < 0: y1 = 0
    if y2 < 0: y2 = 0
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


def getFacesFiles():
    currentPath = join(os.path.abspath(os.getcwd()), 'mysite', 'faces')
    faces = [f for f in listdir(join(currentPath)) if isfile(join(currentPath, f))]
    return faces

def getFacesFilesEmb(files):
    global faces_dict

    faces_dict = {}

    for filename in files:
        name = filename.split('.')[0]
        img = cv2.imread(join('mysite', 'faces', filename))
        faces_dict[name], box = get_embedding(img)[0]

def checkNewFaces():
    files = getFacesFiles()
    oldFaces = [key for key, value in faces_dict.items()]
    oldFaces.sort()
    newFaces = [filename.split('.')[0] for filename in files]
    newFaces.sort()

    if oldFaces != newFaces:
        getFacesFilesEmb(files)
        print('==================================== NEW FACES ============================================')
        print(newFaces)

def get_match(emb, faces):
    match = None
    min_score = 1

    for name, e in faces:
        score = is_match(emb, e)
        if score < 0.5 and score < min_score:
            match = name

    if match is None: match = 'Stranger'

    return match

def match_faces(img, faces):
    image = img
    embs = get_embedding(img)
    res = []

    for img_emb, box in embs:
        match = get_match(img_emb, faces)
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
    checkNewFaces()

    faces = [(get_face_from_box(img, box), box) for box in boxes]
    embs = [(get_emb_from_face(face), box) for face, box in faces]

    image = img
    res = []

    for img_emb, box in embs:
        match = get_match(img_emb, faces_dict.items())
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

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
tracker = Tracker()
person_ids = []

def detections_to_boxes(detections):
    boxes = []

    for detection in detections:
        x1 = int(detection.location_data.relative_bounding_box.xmin * 480)
        x2 = int(x1 + detection.location_data.relative_bounding_box.width * 480)
        y1 = int(detection.location_data.relative_bounding_box.ymin * 360)
        y2 = int(y1 + detection.location_data.relative_bounding_box.height * 360)
        p1 = (x1, y1)
        p2 = (x2, y2)
        boxes.append((p1, p2))

    return boxes

def sortKey(box):
    return box[0][0]

def get_face_matrix(box):
    p1, p2 = box
    x1, y1 = p1
    x2, y2 = p2
    return [[x1, y1], [x2, y2], [(x1 + x2) / 2, (y1 + y2) / 2]]

def detect_faces_media(image, matches):
    global person_ids
    check_face = False
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    boxes = []

    if not results.detections: person_ids = []

    if results.detections:
        boxes = detections_to_boxes(results.detections)
        boxes.sort(key=sortKey)

        faces = np.array([get_face_matrix(box) for box in boxes])
        fa = tracker.track(faces)
        new_ids = list(fa.keys())

        if len(person_ids) != len(new_ids):
            person_ids = new_ids
            check_face = True
        else:
            for id in new_ids:
                if not id in person_ids:
                    person_ids = new_ids
                    check_face = True
                    break

        if not check_face:
            for box, match in zip(boxes, matches):
                draw_box(image, box, match)

    if results.detections is None:
        length = 0
    else:
        length = len(results.detections)

    return image, check_face, length, boxes
