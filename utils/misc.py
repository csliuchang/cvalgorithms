from collections import abc



def update_prefix_of_dict(_dict, old_prefix, new_prefix):
    if not isinstance(_dict, dict):
        return
    tmp = _dict
    for k, v in tmp.items():
        if isinstance(v, str) and v.startswith(old_prefix):
            _dict[k] = v.replace(old_prefix, new_prefix)
        else:
            if isinstance(v, dict):
                update_prefix_of_dict(_dict[k], old_prefix, new_prefix)
            elif isinstance(v, list):
                for _item in v:
                    update_prefix_of_dict(_item, old_prefix, new_prefix)
                    

def update_value_of_dict(_dict, old_value, new_value):
    if not isinstance(_dict, dict):
        return
    tmp = _dict
    for k, v in tmp.items():
        if isinstance(v, str) and v == old_value:
            _dict[k] = new_value
        else:
            if isinstance(v, dict):
                update_value_of_dict(_dict[k], old_value, new_value)
            elif isinstance(v, list):
                for _item in v:
                    update_value_of_dict(_item, old_value, new_value)


def repalce_kwargs_in_dict(_dict):
    if not isinstance(_dict, dict):
        return
    _items = _dict.copy().items()
    for k, v in _items:
        if 'kwargs' == k:
            _kwargs = _dict.pop('kwargs')
            _dict.update(_kwargs)
        else:
            if isinstance(v, dict):
                repalce_kwargs_in_dict(_dict[k])
            elif isinstance(v, list):
                for _item in v:
                    repalce_kwargs_in_dict(_item)


def is_tuple_of(seq, expected_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


class NiceRepr(object):
    """Inherit from this class and define ``__nice__`` to "nicely" print your
    objects.

    Defines ``__str__`` and ``__repr__`` in terms of ``__nice__`` function
    Classes that inherit from :class:`NiceRepr` should redefine ``__nice__``.
    If the inheriting class has a ``__len__``, method then the default
    ``__nice__`` method will return its length.

    Examples
    --------
    >>> class Foo(NiceRepr):
    ...    def __nice__(self):
    ...        return 'info'
    >>> foo = Foo()
    >>> assert str(foo) == '<Foo(info)>'
    >>> assert repr(foo).startswith('<Foo(info) at ')

    Examples
    --------
    >>> class Bar(NiceRepr):
    ...    pass
    >>> bar = Bar()
    >>> import pytest
    >>> with pytest.warns(None) as record:
    >>>     assert 'object at' in str(bar)
    >>>     assert 'object at' in repr(bar)

    Examples
    --------
    >>> class Baz(NiceRepr):
    ...    def __len__(self):
    ...        return 5
    >>> baz = Baz()
    >>> assert str(baz) == '<Baz(5)>'
    """

    def __nice__(self):
        """str: a "nice" summary string describing this module"""
        if hasattr(self, '__len__'):
            return str(len(self))
        else:
            raise NotImplementedError(
                f'Define the __nice__ method for {self.__class__!r}')

    def __repr__(self):
        """str: the string of the module"""
        try:
            nice = self.__nice__()
            classname = self.__class__.__name__
            return f'<{classname}({nice}) at {hex(id(self))}>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)

    def __str__(self):
        """str: the string of the module"""
        try:
            classname = self.__class__.__name__
            nice = self.__nice__()
            return f'<{classname}({nice})>'
        except NotImplementedError as ex:
            warnings.warn(str(ex), category=RuntimeWarning)
            return object.__repr__(self)


def is_method_overridden(method, base_class, derived_class):
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method


def is_list_of(seq, expected_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def is_str(x):
    """Whether the input is an string instance.

    Note: This method is deprecated since python 2 is no longer supported.
    """
    return isinstance(x, str)