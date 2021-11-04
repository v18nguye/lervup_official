import os
import json


def activator(vis_concepts, situ_name, path_pre_vis_concepts, load_detectors):
    """Discover active detectors per situation

    Parameters
    ----------
    vis_concepts : dict
        object impact in a given situation
            {situ: {class1: score, ...}, }

    path_pre_vis_concepts: string
        path to pre-selected visual concepts given by the privacy base-line

    load_detectors : boolean
        load active detectors determined in the privacy baseline method

    Returns
    -------
        active_detectors : dict
            active detectors in a given situation, and its exposure corr
                {detector1: score1,...}

        opt_threds : dict
            optimal threshold for each activated detector (in the loaded detector case)
                {detector1: threshold1, ...}
    """
    active_detectors = {}
    opt_threds = {}

    new_vis_concepts = json.load(open('/home/nguyen/Documents/intern20/Vis-Priva-Expos/baseline_competition/tools/test/combined_opt_ths.json'))

    for class_, scores in new_vis_concepts[situ_name].items():
        active_detectors[class_] = scores[2]
        opt_threds[class_] = scores[1]

    return active_detectors, opt_threds
