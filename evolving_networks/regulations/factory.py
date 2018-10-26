class Factory(object):
    def __init__(self):
        pass

    def determine_mode(self, **kwargs):
        raise NotImplementedError()

    def __str__(self):
        raise NotImplementedError()
