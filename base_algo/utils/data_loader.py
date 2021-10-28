import os
import json

def load_train_test(path):
    """Load pre-processed data

    :param root: string
        current working absolute path
    :param path: string
        relative path to saved train vs test data

    :param load_txt: boolean
        if redefine train and test set by .txt lists

    :return:
           train_data: dict
                training mini-batches
                    {ratio: {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}, ...}

            test_data: dict
                test data
                    {user1: {photo1: {class1: [obj1, ...], ...}}, ...}, ...}

    """
    train_test_info = json.load(open(path))
    test_data = train_test_info['test']
    train_data = train_test_info['train']
    val_data = train_test_info['val']

    return train_data, test_data


def load_gt_user_expo(path):
    """Load crowd-sourcing user exposure

    :param root:
    :param path:

    :return:
        gt_usr_expo : dict
            ground-truth user exposure by situation
                 {situ1: {user1: avg_score, ...}, ...}
    """
    gt_usr_expo = json.load(open(path))

    return gt_usr_expo


def load_situs(path, denormalization = True):
    """Load object situation under a dictionary form

    :param root: string
    :param path: string
        path to situations
    :param denormalization: boolean

    :return:
        class_situs : dict
            situation and its crowd-sourcing class exposure corr
                {situ1: {class1: score, ...}, ...}
    """
    class_situs = {}
    situs = os.listdir(path)


    for situ in situs:
        situ_key = situ.split('.')[0]

        class_situs[situ_key] = {}
        with open(os.path.join(path, situ)) as fp:

            lines = fp.readlines()
            for line in lines:
                parts = line.split(' ')
                class_ = parts[0]

                if denormalization:
                    score = float(parts[1])*3
                else:
                    score = float(parts[1])

                class_situs[situ_key][class_] = score

    return class_situs