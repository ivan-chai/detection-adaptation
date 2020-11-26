import torch
from torchvision.transforms.functional import to_tensor

import numpy as np
import os

from copy import deepcopy

from collections import OrderedDict

import albumentations as A


difficulty_to_events = {
    "easy": ['Gymnastics', 'Handshaking', 'Waiter_Waitress', 'Press_Conference', 'Worker_Laborer',
             'Parachutist_Paratrooper', 'Sports_Coach_Trainer', 'Meeting', 'Aerobics', 'Row_Boat',
             'Dancing', 'Swimming', 'Family_Group', 'Balloonist', 'Dresses', 'Couple', 'Jockey',
             'Tennis', 'Spa', 'Surgeons'],
    "medium": ['Stock_Market', 'Hockey', 'Students_Schoolkids', 'Ice_Skating', 'Greeting', 'Football',
               'Running', 'people--driving--car', 'Soldier_Drilling', 'Photographers', 'Sports_Fan',
               'Group', 'Celebration_Or_Party', 'Soccer', 'Interview', 'Raid', 'Baseball', 'Soldier_Patrol',
               'Angler', 'Rescue'],
    "hard": ['Traffic', 'Festival', 'Parade', 'Demonstration', 'Ceremony', 'People_Marching', 'Basketball',
             'Shoppers', 'Matador_Bullfighter', 'Car_Accident', 'Election_Campain', 'Concerts', 'Award_Ceremony',
             'Picnic', 'Riot', 'Funeral', 'Cheering', 'Soldier_Firing', 'Car_Racing', 'Voter'],
}

events_difficulty = {}
for diff, events in difficulty_to_events.items():
    for event in events:
        events_difficulty[event] = diff


def read_annotations(root, split):
    assert(split in ["train","val","test"])
    annotations = []
    with open(os.path.join(root, split, "label.txt"), "r") as fp:
        line = fp.readline()
        while line:
            line = line.strip()
            if line[0] == "#":
                path = line[2:]
                annotations.append({"path": os.path.join(split, "images", path), "labels": []})
                line = fp.readline()
                continue
            if "--" in line:
                path = line
                annotations.append({"path": os.path.join(split, "images", path), "labels": []})
                line = fp.readline()
                continue

            numbers = list(map(lambda x: float(x), line.split()))

            if len(numbers) < 4:
                line = fp.readline()
                continue

            bbox = np.array(numbers[0:4])
            bbox = np.concatenate([bbox[:2], bbox[:2] + bbox[2:]])

            try:
                keypoints = np.array(numbers[4:19]).reshape(5,3)[:,:2]
                if (keypoints == -1.).all():
                    keypoints = None
            except:
                keypoints = None

            if (bbox[2] - bbox[0])*(bbox[3] - bbox[1]) > 0:
                annotations[-1]["labels"].append({"bbox": bbox, "keypoints": keypoints})
            line = fp.readline()

    return annotations


class WIDERFACEDataset(torch.utils.data.Dataset):
    """See http://shuoyang1213.me/WIDERFACE/. Label files with
    keypoints annotations is available at https://www.dropbox.com/s/7j70r3eeepe4r2g/retinaface_gt_v1.1.zip?dl=0.
    root
    ---train
    ------images
    ------label.txt
    ---test:
    ------images
    ------label.txt
    ---val:
    ------images
    ------label.txt

    A sample of this dataset has a structure {
        "image": Numpy array of shape :math:`(H, W, 3)`,
        "bboxes": Numpy array of shape :math:`(N, 4)` (XYXY format),
        "keypoints": Numpy array of shape :math:`(N, 5, 2)`,
        "has_keypoints": Numpy array of shape :math:`(N,)`,
        ...
    }

    Args:
        root: path to root folder of the dataset
        split: "train", "val" or "test"
        transform: transformation or list of transformations.
    """
    def __init__(self, root, split="train", transform=None):
        assert split in os.listdir(root), f"Folder {split} not found"
        assert "images" in os.listdir(os.path.join(root, split)), f"Folder {split}/images not found"
        assert "label.txt" in os.listdir(os.path.join(root, split)), f"File {split}/label.txt not found"
        self.annotations = read_annotations(root, split)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_annotation = deepcopy(self.annotations[idx])
        image = A.read_rgb_image(os.path.join(self.root, image_annotation["path"]))
        image_labels = image_annotation["labels"]

        bboxes = []
        keypoints = []
        has_keypoints = []
        for label in image_labels:
            bbox = label["bbox"]
            bboxes.append(bbox)
            pts = label["keypoints"]
            if pts is not None:
                keypoints.append(pts)
                has_keypoints.append(1)
            else:
                pts = (bbox[:2] + bbox[2:])/2
                pts = np.tile(pts[None,:], (5,1))
                keypoints.append(pts)
                has_keypoints.append(0)

        bboxes = np.array(bboxes) if len(bboxes)>0 else np.zeros((0,4))
        keypoints = np.array(keypoints) if len(keypoints)>0 else np.zeros((0,5,2))
        has_keypoints = np.array(has_keypoints) if len(has_keypoints)>0 else np.zeros((0,))
        has_keypoints = has_keypoints.astype(np.bool)

        event_name, file_name = image_annotation["path"].split("/")[-2:]
        event_name = '--'.join(event_name.split('--')[1:])
        file_name = ".".join(file_name.split(".")[:-1])

        sample = {
                "image": image,
                "event": event_name,
                "file": file_name,
                "bboxes": bboxes,
                "keypoints": keypoints,
                "has_keypoints": has_keypoints,
                "difficulty": events_difficulty.get(event_name),
        }
        if self.name:
            sample["dataset"] = self.name

        if self.transform is not None:
            to_transform = {key: sample[key] for key in ["image", "bboxes", "keypoints", "has_keypoints", "dataset"]}
            if isinstance(self.transform, list):
                for t in self.transform:
                    to_transform = t(**to_transform)
            else:
                to_transform = self.transform(**to_transform)
            sample.update(to_transform)

        return sample
