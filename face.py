import os
from os import listdir
from os.path import isfile, join
import keras
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn import MTCNN
import cv2

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input


def get_face(img, results, size):
    if len(results) == 0:
        return None

    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(size)
    face_array = asarray(image)
    return face_array


def extract_face(frame, required_size=(224, 224)):
    detector = MTCNN()
    results = detector.detect_faces(frame)
    print(results)

    return get_face(frame, results, required_size)


def get_embedding(file):
    face = extract_face(file)

    if face is None:
        return None

    samples = asarray([face], 'float32')
    samples = preprocess_input(samples)
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    yhat = model.predict(samples)

    return yhat


# determine if a candidate face is a match for a known face
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
    faces_dict[name] = get_embedding(img)

# baraa = cv2.imread('baraa1.jpg')
# baraa = get_embedding(baraa)

detector = MTCNN()


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
    results = detector.detect_faces(img)

    if len(results) > 0:
        box = results[0]['box']
        p1 = (box[0], box[1])
        p2 = (box[0] + box[2], box[1] + box[3])
        img = cv2.rectangle(img, p1, p2, (255, 10, 10))

        if match is not None:
            cv2.putText(img, match, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(img, 'Stranger', p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img, len(results)
