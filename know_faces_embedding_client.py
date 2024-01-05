import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_PTH import MobileFaceNet
from client import SAVED_CLIENT

from typing import OrderedDict
import argparse

#________________________ VARIABLES ___________________________
FACE_DATASET = "/face_dataset"
SAVED_FILE = "/know_faces_embedding.pth"
#________________________ START ___________________________
if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Know-faces embedding")
    parser.add_argument(
        "--face_dataset",
        type=str,
        default=FACE_DATASET,
        help=f"Path to data's directory! (default: /face_dataset)",
    )
    parser.add_argument(
        "--saved_file",
        type=str,
        default=SAVED_FILE,
        help="Location of saved embedding file! (default: know_faces_embedding.pth)",
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
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    class_names = dataset.classes
    known_faces_embeddings = {}

    with torch.no_grad():
        for images, labels in dataloader:
            images.to(device)
            embeddings = model(images).cpu().detach().numpy().flatten()
            person_id = class_names[labels.item()]

            if person_id not in known_faces_embeddings:
                known_faces_embeddings[person_id] = []
            
            known_faces_embeddings[person_id].append(embeddings)

    # Calculate average embeddings
    for person_id, embeddings_list in known_faces_embeddings.items():
        avg_embedding = torch.mean(torch.vstack([torch.tensor(e) for e in embeddings_list]), dim=0)
        known_faces_embeddings[person_id] = avg_embedding

    # Step 4: Save Embeddings
    torch.save(known_faces_embeddings, SAVED_CLIENT + args.saved_file)