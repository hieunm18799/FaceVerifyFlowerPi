from torch import save
from torchvision import transforms
from facenet_pytorch import MTCNN
import numpy as np
from termcolor import colored

import math
from fractions import Fraction

#________________________ VARIABLES ___________________________
PATIENCE = 3

#________________________ FUNCTION ___________________________
# Detect 1 face in image and save it to file
def detect_and_crop_faces(img, target=(112, 96)):
    ratio = Fraction(target[0], target[1])
    ratio = ratio.as_integer_ratio()
    # Define MTCNN module
    mtcnn = MTCNN(device='cpu', select_largest=True, post_process=False, min_face_size=96)

    boxes, _ = mtcnn.detect(img)

    if boxes is None:
        print("The image have 0 faces!")
        return False

    # print(boxes)
    for i, box in enumerate(boxes):
        box_w = box[2] - box[0]
        box_h = box[3] - box[1]
        if box_w / img.width >= 0.15 and box_h / img.height >= 0.233:
            if i > 0:
                print("The image have 2 faces!")
                return False
            mul = math.ceil(box_w / ratio[1])
            if mul < math.ceil(box_h / ratio[0]):
                mul = math.ceil(box_h / ratio[0])

            new_x1 = box[0] - (mul * ratio[1] - box_w) / 2
            new_x2 = new_x1 + mul * ratio[1]
            new_y1 = box[1] - (mul * ratio[0] - box_h) / 2
            new_y2 = new_y1 + mul * ratio[0]

            img = img.crop((new_x1, new_y1, new_x2, new_y2))

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(target)
                # transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[256.0, 256.0, 256.0]),
            ])
            original_image = transform(img)
    return original_image

# Cosine distance
def cal_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

def save_model(model, folder_path):
    save(model.state_dict(), folder_path)
    print(colored("Model saved to %s" % folder_path, "red"))

#________________________ CLASS ___________________________
class EarlyStopping:
    def __init__(self, patience=PATIENCE, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
    def early_stop(self, loss) -> bool:
        if self.best_loss - loss > self.min_delta:
            self.best_loss = loss
            self.counter = 0
        elif self.best_loss - loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False