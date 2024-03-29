import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_PTH import MobileFaceNet
from client import SAVED_CLIENT, BATCH_SIZE, FACE_DATASET

from typing import OrderedDict
import argparse

import pickle

#________________________ VARIABLES ___________________________
SAVED_FILE = "know_faces_embedding.pickle"
#________________________ START ___________________________
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Know-faces embedding")
    parser.add_argument(
        "--face_dataset",
        type=str,
        default=FACE_DATASET,
        help=f"Path to data's directory! (default: {FACE_DATASET})",
    )
    parser.add_argument(
        "--saved_file",
        type=str,
        default=SAVED_FILE,
        help=f"Location of saved embedding file! (default: {SAVED_FILE})",
    )
    args = parser.parse_args()

    device = torch.device('cpu')
    model = MobileFaceNet().to(device)
    state_dict = torch.load(SAVED_CLIENT + '/best_model.pth', map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[127.5, 127.5, 127.5], std=[128.0, 128.0, 128.0]),
    ])
    dataset = ImageFolder(root = args.face_dataset, transform = transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    class_names = dataset.classes
    known_faces_embeddings = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            embeddings = model(images).cpu().detach().numpy()

            for i, label in enumerate(labels):
                person_id = class_names[label.item()]

                if person_id not in known_faces_embeddings:
                    known_faces_embeddings[person_id] = []

                known_faces_embeddings[person_id].append(embeddings[i])

    # for person_id, embeddings_list in known_faces_embeddings.items():
    #     avg_embedding = torch.mean(torch.vstack([torch.tensor(e) for e in embeddings_list]), dim=0)
    #     known_faces_embeddings[person_id] = avg_embedding

    # torch.save(known_faces_embeddings, SAVED_CLIENT + args.saved_file)
    with open(SAVED_CLIENT + args.saved_file, 'wb') as handle:
        pickle.dump(known_faces_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)