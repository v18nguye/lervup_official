import os
import json
def activator(vis_concepts, situ_name, path_pre_vis_concepts, load_detectors):
    """Discover active detectors per situation

    Parameters
    ----------
    vis_concepts : dict
        object impact in a given situation
            {class1: score, ...}

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

    if not load_detectors:
        for class_, score in vis_concepts[situ_name].items():
                active_detectors[class_] = score

    else:
        sel_vis_concepts = json.load(open(path_pre_vis_concepts))
        detector_in_situ = sel_vis_concepts[situ_name]

        for object_, tau_thresh_score in detector_in_situ.items():
            active_detectors[object_] = tau_thresh_score[2] # visual concept score
            opt_threds[object_] = tau_thresh_score[1] # its threshold

    return active_detectors, opt_threds
