"""
This module load the trained model and reproduce its results

Use:
    python model_info.py --model_name out/best_bank_competition.pkl
"""

import json
import pickle
import _init_paths


def info():
    """

    :return:
    """
    overall_results = {}
    MODEL_NAMES = ['out/best_bank_competition.pkl', 'out/best_acc_competition.pkl', 'out/best_it_competition.pkl', 'out/best_wait_competition.pkl']
    for model_name in MODEL_NAMES:

        model = pickle.load(open(model_name, 'rb'))
        model.cfg.OUTPUT.VERBOSE = True
        model.set_seeds()
        results = model.inference()

        for user, score in results.items():
            if user not in overall_results:
                overall_results[user] = {}

            if model_name.split('_')[1] not in overall_results[user]:
                overall_results[user][model_name.split('_')[1]] = score

        json.dump(overall_results, open('./lervup_inference_ftv1.json', 'w'))


if __name__ == '__main__':
    info()