import torch
import numpy as np

from torchvision.transforms.functional import to_tensor

from tqdm import tqdm

from .widerface import WIDERFACEDataset

from dalib.detection import calculate_AP, calculate_PR

class WIDERFACEEvaluator:
    def __init__(self, dataset_root, detector, device="cuda:0"):
        detector = detector.eval().to(device)

        dataset = WIDERFACEDataset(dataset_root, split='val')

        pred_and_tar = {}

        for img, ann in tqdm(dataset, desc="Generating validation predictions"):
            event_name = ann["event"]
            if event_name not in pred_and_tar.keys(): pred_and_tar[event_name] = {}
            file_name = ann["file"]

            with torch.no_grad():
                pred, = detector(to_tensor(img)[None,...].to(device), threshold=0.01)
                for key in pred.keys():
                    if isinstance(pred[key], torch.Tensor):
                        pred[key] = pred[key].numpy()
            pred_and_tar[event_name][file_name] = {"scores": pred["scores"], "bboxes_pr": pred["bboxes"], "bboxes_gt": ann["bboxes"]}

        self.pred_and_tar = pred_and_tar

    def AP_by_difficulty(self, resolution=100, iou_threshold=0.5, min_face_height=10, max_face_height=np.inf):
        diff_to_events = {
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

        metrics = {}

        for diff in diff_to_events.keys():
            PR_table = []
            events = diff_to_events[diff]
            for event in events:
                pts = self.pred_and_tar[event]
                for pt in pts.values():
                    gt_heights = pt["bboxes_gt"][:,3] - pt["bboxes_gt"][:,1]
                    subsets = [np.where(np.logical_and(gt_heights > min_face_height, gt_heights < max_face_height))[0]]
                    if len(subsets[0]) == 0:
                        continue
                    PR_table.append(calculate_PR(pt["scores"], pt["bboxes_pr"], pt["bboxes_gt"], subsets=subsets, resolution=resolution, iou_threshold=iou_threshold)[0])
            PR_table = np.stack(PR_table).mean(axis=0)
            AP = calculate_AP(PR_table)
            metrics[diff] = {"PR": PR_table, "AP": AP}

        return metrics

    def AP_by_size(self, intervals=[10,50,300,np.inf], resolution=100, iou_threshold=0.5):
        PR_tables = []
        for pt_by_file in self.pred_and_tar.values():
            for pt in pt_by_file.values():
                gt_heights = pt["bboxes_gt"][:,3] - pt["bboxes_gt"][:,1]
                subsets = [np.where(np.logical_and(gt_heights >= low, gt_heights < high))[0] for low, high in zip(intervals[:-1], intervals[1:])]
                _PR_tables = calculate_PR(pt["scores"], pt["bboxes_pr"], pt["bboxes_gt"], subsets=subsets, resolution=resolution, iou_threshold=iou_threshold)
                PR_tables.append(_PR_tables)

        PR_tables = np.stack(PR_tables).mean(axis=0)
        AP = np.stack([calculate_AP(PR_table) for PR_table in PR_tables])

        return {"PR": PR_tables, "AP": AP}
