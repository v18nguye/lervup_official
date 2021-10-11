"""
The module analyses fine-tuning parameters


"""
import os
import pickle


def check(path, c, p):
    """
    Retrieve all values of the parameter and its best results.
    Test is verified for a situation + a detector.

    Parameters
    ----------
    path: string
        path to trained models
    c: string
        component
    p: string
        component's param

    Returns
    -------
        dict:
            {param_val1: best1, ...}

    """
    result = {}
    models = os.listdir(path)

    for model_name in models:
        m_path = os.path.join(path, model_name)
        model = pickle.load(open(m_path, 'rb'))
        model.set_seeds()
        model.test_vispel()
        m_result = model.test_result
        param_value = model.cfg[c][p]

        if param_value not in result:
            result[param_value] = m_result

        else:
            if m_result > result[param_value]:
                result[param_value] = m_result

    return result


def check_param_situ(path, c, p):
    """

    Check param values and its results in situations

    Parameters
    ----------
    path: string
        path to both situations + detectors
    c: string
        component
    p: string
        component's params

    Returns
    -------

        {param_val1: {situ1: result1, ...}, ...}
    """

    all_cases = os.listdir(path)
    mobi_results = {}
    rcnn_results = {}

    for situ_detector in all_cases:
        c_path = os.path.join(path, situ_detector)
        result = check(c_path, c, p)

        parts = situ_detector.split('_')

        if parts[1] == 'mobi':
            for key, value in result.items():
                if key not in mobi_results:
                    mobi_results[key] = {}
                mobi_results[key][parts[0]] = value

        if parts[1] == 'rcnn':
            for key, value in result.items():
                if key not in rcnn_results:
                    rcnn_results[key] = {}
                rcnn_results[key][parts[0]] = value

    return mobi_results, rcnn_results
