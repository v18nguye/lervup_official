import os
import json


def train_test(root, path):
    """Load user profile train, test

    :param root: string
        current working absolute path

    :param path: string
        relative path to saved train vs test data

    :return:
           train_data: dict
                training mini-batches
                    {ratio: {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}, ...}

            test_data: dict
                test data
                    {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    """
    train_test_info = json.load(
        open('/home/nguyen/Documents/intern20/Vis-Priva-Expos/privacy-competition/data/train_val_preds.json'))

    test_data = train_test_info['test']
    train_data = train_test_info['train']

    return train_data, test_data


def gt_user_expos(root, path):
    """Load crowd-sourcing user exposure

    :param root:
    :param path:

    :return:
        gt_usr_expo : dict
            ground-truth user exposure by situation
                 {situ1: {user1: avg_score, ...}, ...}
    """
    gt_usr_expo = json.load(open('/home/nguyen/Documents/intern20/Vis-Priva-Expos/privacy-competition/data/gt_ptrainval.json'))

    return gt_usr_expo


def vis_concepts(root, path):
    class_situ = json.load(
        open('/home/nguyen/Documents/intern20/Vis-Priva-Expos/privacy-competition/data/vis_concept_situ.json'))

    return class_situ
