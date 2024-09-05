#Borrowed from Zhengzhong Tu

import numpy as np
import scipy.stats
import scipy.io
import math
from scipy import signal
from sklearn.cluster import KMeans


def vqpooling_pooling(chunk):
    chunk = np.asarray(chunk, dtype=np.float64)  # kmeans IndexError out of bound,
    # bug here:
    # https://stackoverflow.com/questions/41635426/index-n-is-out-of-bounds-for-axis-0-with-size-n-when-running-parallel-kmeans-w
    km = KMeans(n_clusters=2)
    km.fit(chunk.reshape(-1, 1))
    label0 = np.asarray(np.where(km.labels_ == 0), dtype=np.int)
    label1 = np.asarray(np.where(km.labels_ == 1), dtype=np.int)
    # print(label0, label1)
    chunk0 = chunk[label0[0]]
    chunk1 = chunk[label1[0]]
    # print(chunk0, chunk1)
    mu0 = np.mean(chunk0)
    mu1 = np.mean(chunk1)
    # print(mu0, mu1)
    w = (1.0 - min(mu0, mu1)/max(mu0, mu1)) ** 2.0
    # print(w)
    if mu0 <= mu1:
        s = np.divide(np.sum(chunk0) + w * np.sum(chunk1), \
            len(chunk0) + w * len(chunk1))
    else:
        s = np.divide(np.sum(chunk1) + w * np.sum(chunk0), \
            len(chunk1) + w * len(chunk0))
    return s


def hysteresis_pooling(chunk):
    '''parameters'''
    tau = 2 # 2-sec * 30 fps
    comb_alpha = 0.2 # weighting
    ''' function body '''
    chunk = np.asarray(chunk, dtype=np.float64)
    chunk_length = len(chunk)
    l = np.zeros(chunk_length)
    m = np.zeros(chunk_length)
    q = np.zeros(chunk_length)
    for t in range(chunk_length):
        ''' calculate l[t] - the memory component '''
        if t == 0: # corner case
            l[t] = chunk[t]
        else:
            # get previous frame indices
            idx_prev = slice(max(0, t-tau), max(0, t-1)+1)
            # print(idx_prev)
            # calculate min scores
            l[t] = min(chunk[idx_prev])
        # print("l[t]:", l[t])
        ''' compute m[t] - the current component '''
        if t == chunk_length - 1: # corner case
            m[t] = chunk[t]
        else:
            # get next frame indices
            idx_next = slice(t, min(t + tau, chunk_length))
            # print(idx_next)
            # sort ascend order
            v = np.sort(chunk[idx_next])
            # generated Gaussian weight
            win_len = len(v) * 2.0 - 1.0
            win_sigma = win_len / 6.0
            # print(win_len, win_sigma)
            gaussian_win = signal.gaussian(win_len, win_sigma)
            gaussian_half_win = gaussian_win[len(v)-1:]
            # normalize gaussian descend kernel
            gaussian_half_win = np.divide(gaussian_half_win, np.sum(gaussian_half_win))
            # print(gaussian_half_win)
            m[t] = sum([x * y for x, y in zip(v, gaussian_half_win)])
        # print("m[t]:", m[t])
    ''' combine l[t] and m[t] into one q[t] '''
    q = comb_alpha * l + (1.0 - comb_alpha) * m
    # print(q)
    # print(np.mean(q))
    return q, np.mean(q)


# chunk = [3,3 ... ] 200x1 -> pooling -> return value
def pooling(chunk, pooling_type):
    chunk = np.squeeze(chunk)
    # print(chunk)
    if pooling_type == "mean":
        return np.mean(chunk)
    elif pooling_type == "geometric":
        if chunk.size == 1:
            return chunk
        # harmonic mean is only defined for positive values
        chunk_positive = list(filter(lambda x: x > 0, chunk))
        chunk_positive = np.asarray(chunk_positive, dtype=np.float64)
        return scipy.stats.mstats.gmean(chunk_positive)
    elif pooling_type == "median":
        return np.median(chunk)
    elif pooling_type == "harmonic":
        if chunk.size == 1:
            return chunk
        # harmonic mean is only defined for positive values
        chunk_positive = list(filter(lambda x: x > 0, chunk))
        chunk_positive = np.asarray(chunk_positive, dtype=np.float64)
        # hm = n / (1/x1 + 1/x2 + ... + 1/xn)
        return len(chunk_positive) / np.sum(1.0 / chunk_positive)
    elif pooling_type == "minkowski":
        # p = 2
        chunk = np.asarray(chunk, dtype=np.float64)
        return np.sqrt(np.mean(chunk**2))
    elif pooling_type == "percentile":
        if chunk.size == 1:
            return chunk
        else:
            threshold = np.percentile(chunk, q=10)
            window = list(filter(lambda x: x < threshold, chunk))
            # m = np.mean(window)
            return np.mean(window) if window != [] else 0
    elif pooling_type == "up-perc":
        threshold = np.percentile(chunk, q=80)
        window = list(filter(lambda x: x > threshold, chunk))
        # m = np.mean(window)
        return np.mean(window) if window != [] else 0
    elif pooling_type == "vqpooling":
        if chunk.size == 1:
            return chunk
        else:
            return vqpooling_pooling(chunk)
    elif pooling_type == "variation":
        if chunk.size == 1:
            return 0
        else:
            chunk_grad = np.abs(np.gradient(chunk))
            # print(chunk_grad)
            threshold = np.percentile(chunk_grad, q=90)
            window = list(filter(lambda x: x > threshold, chunk_grad))
            # print(window)
            return np.mean(window) if window != [] else 0
    elif pooling_type == "recency_simple":
        L = 5
        return np.mean(chunk[:-L])
    elif pooling_type == "primacy":
        if chunk.size == 1:
            return chunk
        # fP(t) = exp(−αP ∗ t), 0 ≤ t ≤ L
        alpha = 0.01
        L = 6  # 2-sec * 30 fps
        weight_vec = np.zeros(L)
        for t in range(L):
            weight_vec[t] = math.exp(-alpha * t)
        # print(weight_vec)
        s = sum([x * y for x, y in zip(chunk[0:L], weight_vec)])
        s = np.divide(s, np.sum(weight_vec))
        return s
    elif pooling_type == "recency":
        if chunk.size == 1:
            return chunk
        # fR(t) = exp(−αR ∗ (L − t)), 0 ≤ t ≤ L
        alpha = 0.01
        L = 6
        weight_vec = np.zeros(L)
        for t in range(L):
            weight_vec[t] = math.exp(-alpha * (L - t))
        # print(weight_vec)
        s = sum([x * y for x, y in zip(chunk[-L:], weight_vec)])
        s = np.divide(s, np.sum(weight_vec))
        return s
    elif pooling_type == "hybrid":
        alpha_p = 0.01
        alpha_r = 0.01
        comb_alpha = 1.0
        weight_vec = np.zeros(len(chunk))
        for t in range(len(chunk)):
            weight_vec[t] = math.exp(-alpha_r * (len(chunk) - t)) +\
                comb_alpha * math.exp(-alpha_p * t)
        # print(weight_vec)
        s = sum([x * y for x, y in zip(chunk, weight_vec)])
        s = np.divide(s, np.sum(weight_vec))
        return s
    elif pooling_type == "hysteresis":
        if chunk.size == 1:
            return chunk
        else:
            q, q_mean = hysteresis_pooling(chunk)
            return q_mean
    elif pooling_type == "votingpool":
        if False: # v1
            Q = []
            Q.append(pooling(chunk, pooling_type="mean"))
            Q.append(pooling(chunk, pooling_type="harmonic"))
            Q.append(pooling(chunk, pooling_type="minkowski"))
            Q.append(pooling(chunk, pooling_type="percentile"))
            Q.append(pooling(chunk, pooling_type="vqpooling"))
            # Q.append(pooling(chunk, pooling_tyep="variation"))
            # Q.append(pooling(chunk, pooling_type="primacy"))
            # Q.append(pooling(chunk, pooling_type="recency"))
            Q.append(pooling(chunk, pooling_type="hysteresis"))
            return np.mean(Q)
        if True: # v2
            Q = []
            # Q.append(pooling(chunk, pooling_type="mean"))
            Q.append(pooling(chunk, pooling_type="harmonic"))
            # Q.append(pooling(chunk, pooling_type="geometric"))
            # Q.append(pooling(chunk, pooling_type="minkowski"))
            Q.append(pooling(chunk, pooling_type="percentile"))
            Q.append(pooling(chunk, pooling_type="vqpooling"))
            # Q.append(pooling(chunk, pooling_tyep="variation"))
            # Q.append(pooling(chunk, pooling_type="primacy"))
            # Q.append(pooling(chunk, pooling_type="recency"))
            Q.append(pooling(chunk, pooling_type="hysteresis"))
            Q = np.sort(np.asarray(Q, dtype=np.float64))
            win_len = len(Q) * 2.0 - 1.0
            win_sigma = win_len / 6.0
            # print(win_len, win_sigma)
            gaussian_win = signal.gaussian(win_len, win_sigma)
            gaussian_half_win = gaussian_win[len(Q)-1:]
            # normalize gaussian descend kernel
            gaussian_half_win = np.divide(gaussian_half_win, np.sum(gaussian_half_win))
            # print(gaussian_half_win)
            return sum([x * y for x, y in zip(Q, gaussian_half_win)])
    elif pooling_type == "epool":
        # Needs two pass of training
        Q = []
        Q.append(pooling(chunk, pooling_type="mean"))
        # Q.append(pooling(chunk, pooling_type="median"))
        # Q.append(pooling(chunk, pooling_type="harmonic"))
        Q.append(pooling(chunk, pooling_type="minkowski"))
        Q.append(pooling(chunk, pooling_type="percentile"))
        # Q.append(pooling(chunk, pooling_type="up-perc"))
        Q.append(pooling(chunk, pooling_type="vqpooling"))
        Q.append(pooling(chunk, pooling_type="variation"))
        # Q.append(pooling(chunk, pooling_type="primacy"))
        # Q.append(pooling(chunk, pooling_type="recency"))
        # Q.append(pooling(chunk, pooling_type="memory"))
        Q.append(pooling(chunk, pooling_type="hysteresis"))
        return Q
    elif pooling_type == "hyst-perc":
        q, _ = hysteresis_pooling(chunk)
        threshold = np.percentile(q, q=20)
        window = list(filter(lambda x: x < threshold, q))
        # m = np.mean(window)
        return np.mean(window) if window != [] else 0
    elif pooling_type == "hyst-up-perc":
        q, _ = hysteresis_pooling(chunk)
        threshold = np.percentile(q, q=90)
        window = list(filter(lambda x: x > threshold, q))
        # m = np.mean(window)
        return np.mean(window) if window != [] else 0
    elif pooling_type == "hyst-vqpool":
        q, _ = hysteresis_pooling(chunk)
        return vqpooling_pooling(q)
    elif pooling_type == "hyst-harmonic":
        q, _ = hysteresis_pooling(chunk)
        return pooling(q, pooling_type="harmonic")
    elif pooling_type == "hyst-geometric":
        q, _ = hysteresis_pooling(chunk)
        return pooling(q, pooling_type="geometric")
    elif pooling_type == "hyst-minkowski":
        q, _ = hysteresis_pooling(chunk)
        return pooling(q, pooling_type="minkowski")
    elif pooling_type == "hyst-hybrid":
        q, _ = hysteresis_pooling(chunk)
        return pooling(q, pooling_type="hybrid")
    else:
        raise Exception("Unknown pooling methods!")