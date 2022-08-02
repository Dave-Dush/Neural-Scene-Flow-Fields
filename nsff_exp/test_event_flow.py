from event_flow import warp_network, FlowDataset
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import math

def test_warp(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device):

    test_dset = FlowDataset.FlowDataset(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args)
    test_loader = DataLoader(test_dset, batch_size=args.batch_size)

    warp_model = warp_network.Warp()
    warp_model = torch.nn.DataParallel(warp_model)
    warp_model_ckp = torch.load("/usr/stud/dave/Neural-Scene-Flow-Fields/nsff_exp/models/hopeful-rain-13.pth")
    warp_model.load_state_dict(warp_model_ckp["model_state_dict"])
    warp_model.to(device)

    warp_model.eval()

    with torch.no_grad():
        features, left_ev, right_ev, targets = test_dset[1]
        #print(features)
        img_before = features["before"]["rgb_image_tensor"].unsqueeze(0)
        img_after = features["after"]["rgb_image_tensor"].unsqueeze(0)

        rev_voxel = features["before"]["reversed_voxel_grid"].unsqueeze(0)
        after_voxel = features["after"]["voxel_grid"].unsqueeze(0)

        features = {
            "before": {"rgb_image_tensor": img_before, "reversed_voxel_grid": rev_voxel},
            "after": {"rgb_image_tensor": img_after, "voxel_grid": after_voxel},
        }
        warped_items = warp_model(features)
        # print(warped_items[2].shape, warped_items[4].shape)
        # for i, (features, left_ev, right_ev, targets) in enumerate(tqdm(test_loader)):

        #     #print(features)
        #     targets = targets.to(device)
            
        #     warped_items = warp_model(features)

        #     print(warped_items[2].shape)
        #     print(warped_items[4].shape)


def dirs_to_paths(seq_dirs):
    """
    Return path dictionary that contains left, middle and right structure
    """
    max_interpolations = len(seq_dirs) * 3

    event_roots = list()

    left_imgs = list()
    middle_imgs = list()
    right_imgs = list()
    matching_ts = list()

    for m_i in tqdm(range(0, max_interpolations, 3)):
        dir_idx = math.floor(m_i/3)

        _seq_dir = seq_dirs[dir_idx]
        _ts = list()

        with open(os.path.join(_seq_dir, "upsampled/imgs/timestamp.txt"), "r") as _f:
            _temp_ts = [float(line.strip()) for line in _f]
            _ts.extend(_temp_ts)

        t_start = _ts[0]
        t_end = _ts[-1]
        
        _dt = np.linspace(t_start, t_end, 7)

        _event_root = os.path.join(_seq_dir, "events/")
        event_roots.append(_event_root)

        for in_i in range(3):

            _left_t = _dt[(2 * in_i)]
            _middle_t = _dt[(2 * in_i) + 1]
            _right_t = _dt[(2 * in_i) + 2]

            left_match = (np.abs(_ts - _left_t).argmin())
            middle_match = (np.abs(_ts - _middle_t).argmin())
            right_match = (np.abs(_ts - _right_t).argmin()) 

            _ts_dict = {
                "left": _left_t,
                "middle": _middle_t,
                "right": _right_t,
            }
            matching_ts.append(_ts_dict)
            
            left_id = f"{left_match:08d}"
            middle_id = f"{middle_match:08d}"
            right_id = f"{right_match:08d}" 

            left_path = os.path.join(_seq_dir, f"upsampled/imgs/{left_id}.png") 
            left_imgs.append(left_path)

            middle_path = os.path.join(_seq_dir, f"upsampled/imgs/{middle_id}.png") 
            middle_imgs.append(middle_path)

            right_path = os.path.join(_seq_dir, f"upsampled/imgs/{right_id}.png")  
            right_imgs.append(right_path)
    
    return left_imgs, middle_imgs, right_imgs, event_roots, matching_ts

def config_parse():
    import configargparse

    parser = configargparse.ArgumentParser()

    parser.add_argument("--config", is_config_file=True)

    parser.add_argument("--equal_dir_txt", type=str)

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--width", type=int)

    parser.add_argument("--height", type=int)

    parser.add_argument("--epochs", type=int)

    parser.add_argument("--batch_size", type=int)

    parser.add_argument("--lr", type=float)

    parser.add_argument("--dataset_size", type=int)

    args = parser.parse_args()

    return args

if __name__ == "__main__":

    args = config_parse()
    
    equal_dir_txt = args.equal_dir_txt
    total_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dset_size = args.dataset_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Getting appropriate dirs")

    #get dir list from txt file using with and not np

    
    dir_list = ["/usr/stud/dave/storage/user/dave/timelens_vimeo/00002/0351"]

    left_imgs, middle_imgs, right_imgs, event_roots, matching_ts = dirs_to_paths(dir_list)
   
    test_warp(left_imgs, middle_imgs, right_imgs, event_roots, matching_ts, args, device)