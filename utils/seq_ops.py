import numpy as np


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chop(L,min_overlap=511,max_len=1022):
    return L[max_len-min_overlap:-max_len+min_overlap]

def intervals(L,min_overlap=511,max_len=1022,parts=None):
    if parts is None: parts = []
    #print('L:',len(L))
    #print(len(parts))
    if len(L)<=max_len:
        if parts[-2][-1]-parts[-1][0]<min_overlap:
        #print('DIFF:',parts[-2][-1]-parts[-1][0])
            return parts+[np.arange(L[int(len(L)/2)]-int(max_len/2),L[int(len(L)/2)]+int(max_len/2)) ]
        else:
            return parts
    else:
        parts+=[L[:max_len],L[-max_len:]]
        L=chop(L,min_overlap,max_len)
        return intervals(L,min_overlap,max_len,parts=parts)

def get_intervals_and_weights(seq_len,min_overlap=511,max_len=1022,s=16):
    ints=intervals(np.arange(seq_len),min_overlap=min_overlap,max_len=max_len)
    ## sort intervals
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]

    a=int(np.round(min_overlap/2))
    t=np.arange(max_len)

    f=np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))
    f[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

    f0=np.ones(max_len)
    f0[max_len-a:] = 1 / (1 + np.exp((t[:a]-a/2)/s))

    fn=np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a]-a/2)/s))

    filt=[f0]+[f for i in ints[1:-1]]+[fn]
    M = np.zeros((len(ints),seq_len))
    for k,i in enumerate(ints):
        M[k,i] = filt[k]
    M_norm = M/M.sum(0)
    return (ints, M, M_norm)


def break_long_sequence(seq_length, model_window=1022):
    half_window = model_window // 2
    if seq_length <= model_window:
        return [[0, seq_length]]
    else:
        lst = []
        s = 0; e = 0
        while e < seq_length:
            e = min(seq_length, s + model_window)
            lst.append([s, e])
            s += half_window
        return lst
    
def get_interval(one_based_position, seq_length, model_window = 1022):
    half_window = model_window // 2
    if seq_length <= model_window:
        return [0, model_window]
    p = one_based_position - 1
    k = (p // half_window) * half_window    
    if k < half_window:
        return [0, model_window]
    elif k + half_window > seq_length:
        return [max(0, k - half_window), seq_length]
    else:
        if p - k < k + half_window - p:
            s = max(0, k - half_window)
            e = min(seq_length, k + half_window)
        else:
            s = k
            e = min(seq_length, k + model_window)
        return [s, e]