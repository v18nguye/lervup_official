def load_acronym(situ):
    """

    :return:
    """
    encoding= {'job_search_IT': 'IT', 'bank_credit': 'BANK', \
                         'job_search_waiter_waitress': 'WAIT', \
                         'accommodation_search': 'ACCOM', \
                         'life_insurance': 'LIFI'}

    return encoding[situ]

def situ_decoding(situ):
    """

    :return:
    """
    decoding = {'IT':'job_search_IT', 'BANK': 'bank_credit', \
                         'WAIT': 'job_search_waiter_waitress', \
                         'ACCOM': 'accommodation_search', \
                         'LIFI': 'life_insurance'}

    return decoding[situ]