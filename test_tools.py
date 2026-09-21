from math import log

def tensor_mem_usage(tensor):
    return tensor.nelement() * tensor.element_size()

def model_mem_usage(model):
    psz = sum(p.nelement() * p.element_size() for p in model.parameters())
    bsz = sum(b.nelement() * b.element_size() for b in model.buffers())
    return psz + bsz

def mem_usage(obj):
    if obj.__class__ in (int, float):
        return obj
    if obj.__class__.__name__ == 'Tensor':
        return tensor_mem_usage(obj)
    if obj.__class__ in (list, tuple):
        return sum([mem_usage(o) for o in obj])
    return model_mem_usage(obj)

def mem_str(obj):
    units = ['B', 'KB', 'MB', 'GB']
    mem = mem_usage(obj)
    l = int(log(mem, 1024))
    if l == 0:
        return '{:.0f}{}'.format(mem / (1024 ** l), units[l])
    return '{:.2f}{}'.format(mem / (1024 ** l), units[l])

def show_mem(obj):
    print(mem_str(obj))

def shape_and_size(obj):
    if obj.__class__.__name__ == 'Tensor':
        print('{} - {}'.format(obj.shape, mem_str(obj)))
    elif obj.__class__ in (list, tuple):
        for o in obj:
            print('{} - {}'.format(list(o.shape), mem_str(o)))
        print('TOTAL: {}'.format(mem_str(obj)))
    else:
        print('Model size: {}'.format(mem_str(obj)))
