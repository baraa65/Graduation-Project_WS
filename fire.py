import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')


def is_fire(img):
    result = model(img)
    new_panda = result.pandas().xyxy[0]
    fire = False

    for i in (new_panda['name']):
        for j in new_panda['confidence']:
            if i == 'fire':
                if j > 0.3:
                    fire = True

    return fire
