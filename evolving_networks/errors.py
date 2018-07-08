class InvalidAggregationError(KeyError):
    pass


class InvalidActivationError(KeyError):
    pass


class InvalidConfigurationError(TypeError):
    pass


class InvalidSpecieAttribute(AttributeError):
    pass


class InvalidConfigurationFile(IOError):
    pass


class InvalidAttributeValueError(AttributeError):
    pass
