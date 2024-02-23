import json

def serialize_with_class_names(data):
    """
    Recursively serializes a data structure, replacing non-JSON serializable objects
    with their class names as strings. This function supports nested dictionaries
    and lists.

    Parameters
    ----------
    data : dict or list
        The data structure to process for JSON serialization. Can be a dictionary
        or a list containing nested dictionaries and lists.

    Returns
    -------
    dict or list
        A new data structure with non-serializable objects replaced by their
        class names in string format.

    Examples
    --------
    >>> class CustomClass:
    ...     pass
    >>> data = {
    ...     'int': 1,
    ...     'list': [1, 2, 3, CustomClass()],
    ...     'dict': {'key': 'value', 'nested': {'custom_obj': CustomClass()}},
    ... }
    >>> result = serialize_with_class_names(data)
    >>> print(result)
    {'int': 1, 'list': [1, 2, 3, 'CustomClass'], 'dict': {'key': 'value', 'nested': {'custom_obj': 'CustomClass'}}}
    """
    if isinstance(data, dict):
        return {k: serialize_with_class_names(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_with_class_names(item) for item in data]
    else:
        try:
            json.dumps(data)
            return data  # Data is serializable, return as is
        except TypeError:
            return data.__class__.__name__  # Replace with class name if not serializable
