from torch import nn
from functools import reduce


def adjust_args(args):
    """given a list of arguments param:args:
    if the first argument is not a list, make it into one.
    for all following args: if it is a single value,
        repeat it into a list of the same len as the first one.
        if it is already a list, assert it has the same len as the first one"""
    if args[0].__class__ not in (tuple, list):
        args[0] = [args[0]]
    l = len(args[0])

    for i, arg in enumerate(args[1:], start=1):
        if arg.__class__ not in (tuple, list):
            args[i] = [arg for j in range(l)]
        else:
            assert len(arg) == l, "wrong len for parameter {}".format(i)

    return args


# TODO maybe put this in an general params file
classes_to_init = [nn.Conv2d,
                   nn.Conv1d]

def init_weights(model, init_f, classes=classes_to_init):
    """initializes param:model's weights by applying param_init_f
    only initializes modules of the classes in classes_to_init
    """
    if (reduce(lambda x, y: x or y, [isinstance(model, c) for c in classes])):
        init_f(model.weight)
        model.weight.requires_grad = True
        if model.bias is not None:
            model.bias.data.zero_()
            model.bias.requires_grad = True