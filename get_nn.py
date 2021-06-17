"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import torch
import yaml
from termcolor import colored
from utils.common_config import get_val_dataset, get_val_transformations, get_val_dataloader,\
                                get_model, get_test_dataset
from utils.evaluate_utils import get_predictions, get_predictions_sample, hungarian_evaluate
from utils.memory import MemoryBank 
from utils.utils import fill_memory_bank, single_image_inference
from PIL import Image
import pandas as pd
import numpy as np
import os

FLAGS = argparse.ArgumentParser(description='Evaluate models from the model zoo')
FLAGS.add_argument('--config_exp', help='Location of config file')
FLAGS.add_argument('--model', help='Location where model is saved')
FLAGS.add_argument('--save_grid', action='store_true', help='Location where model is saved')
args = FLAGS.parse_args()

PARALLEL_NOT_SET_FOR_SIMCLR = True

def main():
    
    # Read config file
    print(colored('Read config file {} ...'.format(args.config_exp), 'blue'))
    with open(args.config_exp, 'r') as stream:
        config = yaml.safe_load(stream)
    config['batch_size'] = 512 # To make sure we can evaluate on a single 1080ti
    print(config)

    # Get dataset
    print(colored('Get train dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset = get_test_dataset(config, transforms)
    # dataset = get_val_dataset(config, transforms)
    dataloader = get_val_dataloader(config, dataset)
    print('Number of samples: {}'.format(len(dataset)))

    # Get model
    print(colored('Get model ...', 'blue'))
    model = get_model(config)
    print(model)

    # Read model weights
    print(colored('Load model weights ...', 'blue'))
    state_dict = torch.load(args.model, map_location='cpu')

    if config['setup'] in ['simclr', 'moco', 'selflabel']:

        if PARALLEL_NOT_SET_FOR_SIMCLR:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            
            for k, v in state_dict.items():
                k = k.replace('module.', '')
                new_state_dict[k] = v

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)

    elif config['setup'] == 'scan':
        model.load_state_dict(state_dict['model'])

    else:
        raise NotImplementedError
        
    # CUDA
    model.cuda()

    # Perform evaluation
    if config['setup'] in ['simclr', 'moco']:
        print(colored('Perform evaluation of the pretext task (setup={}).'.format(config['setup']), 'blue'))
        print('Create Memory Bank')
        if config['setup'] == 'simclr': # Mine neighbors after MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'],
                                    config['num_classes'], config['criterion_kwargs']['temperature'])

        else: # Mine neighbors before MLP
            memory_bank = MemoryBank(len(dataset), config['model_kwargs']['features_dim'], 
                                    config['num_classes'], config['temperature'])
        memory_bank.cuda()

        print('Fill Memory Bank')
        fill_memory_bank(dataloader, model, memory_bank)

        import glob
        dir_path = os.path.join('samples', '*.jpg')
        IMG_NAMES = glob.glob(dir_path)
        # IMG_NAMES = [os.path.join('samples', '544878_23_4_ori.jpg')]

        for IMG_NAME in IMG_NAMES:
            output = single_image_inference(model, IMG_NAME)
            # print(memory_bank.features.shape)
            # print(output.shape)
            print('Mine the nearest neighbors')

            if args.save_grid:
                TOPK = 7
                indices = memory_bank.mine_nearest_neighbors_single_image(output, TOPK)
                IMG_NAME = IMG_NAME.replace('samples/', '').replace('.jpg', '')
                visualize_save_grid(indices[0], dataset, IMG_NAME)

    elif config['setup'] in ['scan', 'selflabel']:
        print(colored('Perform evaluation of the clustering model (setup={}).'.format(config['setup']), 'blue'))
        head = state_dict['head'] if config['setup'] == 'scan' else 0
        out = get_predictions_sample(config, dataloader, model)

        file = pd.DataFrame()
        # print(len(out[0]['predictions']))
        # print(len(out[0]['paths']))
        file['class'] = out[0]['predictions']
        file['path'] = out[0]['paths']
        file.to_csv('./cluster_info.csv')

        for (cls, pth) in zip(file['class'], file['path']):
            path = os.path.join('results', 'cluster_result', str(cls))
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

            image = Image.open(pth).convert("RGB")
            pth = pth.replace('data/samples/0/', '')
            # print(os.path.join(path, pth))
            # print(str(cls))
            image.save(os.path.join(path, pth))
    else:
        raise NotImplementedError

@torch.no_grad()
def get_prototypes(config, predictions, features, model, topk=10):
    import torch.nn.functional as F

    # Get topk most certain indices and pred labels
    print('Get topk')
    probs = predictions['probabilities']
    n_classes = probs.shape[1]
    dims = features.shape[1]
    max_probs, pred_labels = torch.max(probs, dim = 1)
    indices = torch.zeros((n_classes, topk))
    for pred_id in range(n_classes):
        probs_copy = max_probs.clone()
        mask_out = ~(pred_labels == pred_id)
        probs_copy[mask_out] = -1
        conf_vals, conf_idx = torch.topk(probs_copy, k = topk, largest = True, sorted = True)
        indices[pred_id, :] = conf_idx

    # Get corresponding features
    selected_features = torch.index_select(features, dim=0, index=indices.view(-1).long())
    selected_features = selected_features.unsqueeze(1).view(n_classes, -1, dims)

    # Get mean feature per class
    mean_features = torch.mean(selected_features, dim=1)

    # Get min distance wrt to mean
    diff_features = selected_features - mean_features.unsqueeze(1)
    diff_norm = torch.norm(diff_features, 2, dim=2)

    # Get final indices
    _, best_indices = torch.min(diff_norm, dim=1)
    one_hot = F.one_hot(best_indices.long(), indices.size(1)).byte()
    proto_indices = torch.masked_select(indices.view(-1), one_hot.view(-1))
    proto_indices = proto_indices.int().tolist()
    return proto_indices


def visualize_save_grid(indices, dataset, img_name):
    from torchvision.utils import save_image

    path = os.path.join('samples')
    # path = os.path.join(config['setup'], 'images')

    try:
        os.makedirs(path)
    except FileExistsError:  # directory already exists
        pass

    imgs = []
    for idx in indices:
        img = np.array(dataset.get_image(idx)).astype(np.uint8)
        # img = Image.fromarray(img)
        imgs.append(img)
    imgs = torch.tensor(imgs).permute(0, 3, 1, 2).float()

    save_image(imgs, nrow=8, padding=2, normalize=True, fp=os.path.join(path, f"{img_name}_result.jpg"))

    print(f"[INFO] {img_name} IMAGES SAVED!")


if __name__ == "__main__":
    main() 
