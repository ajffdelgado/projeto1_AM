import numpy as np
import pandas as pd
from scipy.stats import scoreatpercentile
from sklearn.metrics import adjusted_rand_score
from random import sample
from functools import reduce
from operator import mul

import time

data = pd.read_csv('seg.csv', sep=",")
data = np.asarray(data)
X = data[:, 0:-1]
y_names = data[:, -1]
# view_shape = X[:, 0:6]
# view_rgb = X[:, 6:]
view_complete = X

labels = list(set(y_names))
classes_dict = {}

i = 0

t_s1 = []
t_s2 = []
t_s3 = []

for label in labels:
    classes_dict[label] = i
    i += 1

y_numbers = [classes_dict[y_names[i]] for i in range(y_names.shape[0])]


def normalize_data(rawpoints, high=1.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)

    rng = maxs - mins + 0.00000001

    return high - (((high - low) * (maxs - rawpoints)) / rng)


def compute_gamma(X):
    res = [X[l] - X[k] for l in range(len(X)) for k in range(l + 1, len(X))]
    res = [np.linalg.norm(r) for r in res]
    res = [r * r for r in res]

    # average of 10th and 90th percentiles
    sigma_2 = 0.5 * (scoreatpercentile(res, 10) + scoreatpercentile(res, 90))

    return 1. / sigma_2


def kernel_function(s, x_l, x_k, p):
    return np.exp(-0.5 * sum([(1 / s[i]) * (x_l[i] - x_k[i])
                             * (x_l[i] - x_k[i]) for i in range(p)]))


def kernel_c_means(X, n_clusters, gamma):
    # number of objects
    n = len(X)

    # number of features
    p = len(X[0])

    ## Initialization step

    # prototypes (random selection)
    g_indices = sample(range(n), n_clusters)
    g = [X[i] for i in g_indices]

    # centers of the best result from previous results
    # g = [[0.3795364043192934, 0.3681727851481405, 0.07073382373973885,
    #       0.00402763321023004, 0.05504151409211094, 0.0038551033999014365,
    #       0.1446596120857817, 0.13224435045688718 ,0.18418802505174003,
    #       0.11453608261492636, 0.702055821321919 ,0.3554298305915442,
    #       0.3521494094525531, 0.1844477154368661 ,0.44228416650158875,
    #       0.19337138121134181],
    #      [0.553139167799788, 0.12619817205290282 ,0.03181285339486201,
    #       0.0005705222005531411, 0.02709580627710643, 0.0005908827283926524,
    #       0.8939767073060303 ,0.8649694508193533 ,0.9434928804442836,
    #       0.8680717409977257 ,0.3484052250118648 ,0.5804920827260861,
    #       0.34883595179595434 ,0.9434928804442836 ,0.16771806614198298,
    #       0.12250615752588051],
    #      [0.509189380449152 ,0.1777557356692448 ,0.028346277974123257,
    #       0.0005105938182748351, 0.02630498511505035, 0.0005777043182389368,
    #       0.7406174355988876 ,0.679979686576522 ,0.8365782355476681,
    #       0.6962143748989461 ,0.1788685770252273 ,0.7668335162556471,
    #       0.2207172258283876 ,0.8365782355476681 ,0.26328427547193883,
    #       0.12790299480596057],
    #      [0.631948522910596 ,0.5151297483724596 ,0.10948586158418702,
    #       0.005852996175252289, 0.11896351495012807, 0.013719113266361684,
    #       0.2781315275050532 ,0.2588230461354997 ,0.32488241258548844,
    #       0.24678520110214264 ,0.6118648009759239 ,0.4216089289023741,
    #       0.33714021659289756 ,0.3250005010905273 ,0.294650039448744,
    #       0.16332277733107547],
    #      [0.4529565638690646 ,0.5658834408668962 ,0.0865966550311916,
    #       0.002859744397128692, 0.06674324291420097, 0.002127637032383047,
    #       0.3971234646744884 ,0.3689749644420364 ,0.4676244611465743,
    #       0.3489555712622391 ,0.51284736197632 ,0.5635799068950786,
    #       0.2088632060164993 ,0.4676244611465743, 0.29838661950760614,
    #       0.1674290590838581],
    #      [0.4802329886526328 ,0.45553525658720384, 0.03544555563658312,
    #       0.002000434881393091, 0.023078043475988224, 0.0017000567284905703,
    #       0.0263362341945705 ,0.020834256900060123 ,0.040542573247516124,
    #       0.016550229149884393, 0.7875525893875143 ,0.20608260723250904,
    #       0.5060751608862669 ,0.041543470485270656 ,0.6678937998252775,
    #       0.2195206753622774],
    #      [0.5140310511860255 ,0.8088963968719736 ,0.054475651774663555,
    #       0.001406610572214242, 0.04628977146905407, 0.0013988087498137984,
    #       0.10873536533330527 ,0.09135095968851728 ,0.09234759533333442,
    #       0.14263206181438182 ,0.679196258829785 ,0.07893182347743932,
    #       0.8213638898491268 ,0.13485718624921064 ,0.4145153358858919,
    #       0.8923330272503802]]

    # width hyperparameters
    s = [1. / gamma for i in range(p)]

    # assign objects to clusters
    res = [[2 * (1 - kernel_function(s, x, g[i], p)) for i in range(n_clusters)] for x in X]

    # cluster assignment
    y = np.array([np.argmin(r) for r in res])

    # partition of clusters
    P = [X[y == i] for i in sorted(set(y))]
    print(len(P))
    print(n_clusters)
    while True:
        ## Representation step
        print(len(P))
        print(n_clusters)
        start = time.time()

        for i in range(n_clusters):
            numerator = 0
            denominator = 0
            for x in P[i]:
                kf = kernel_function(s, x, g[i], p)
                numerator += kf * np.array(x)
                denominator += kf

            g[i] = numerator / denominator

        stop = time.time()
        t_s1.append(stop - start)
        start = time.time()

        ## Width hyperparameter computation step

        summ = [sum([sum([kernel_function(s, x, g[i], p) \
                         * (x[h] - g[i][h]) * (x[h] - g[i][h]) \
                          for x in P[i]]) for i in range(n_clusters)])
                for h in range(p)]
        prod = reduce(mul, summ)
        s = [summ[j] / (np.power(gamma, 1 / p) \
                        * np.power(prod, 1 / p)) for j in range(p)]

        stop = time.time()
        t_s2.append(stop - start)

        ## Cluster allocation step

        test = 0

        start = time.time()

        # compute winning clusters

        res = [[2 * (1 - kernel_function(s, x, g[i], p))
                for i in range(n_clusters)] for x in X]
        y_new = np.array([np.argmin(r) for r in res])

        if not np.array_equal(y_new, y):
            y = np.copy(y_new)
            print (y)
            print ("Len novo y: "+str(len(P)))
            # update partition
            P = [X[y == i] for i in sorted(set(y))]
            print(P)
            print ("Len novo P: "+str(len(P)))
            # continue while clusters change
            test = 1

        stop = time.time()
        t_s3.append(stop - start)

        if test == 0:
            break

    # compute objective function
    obj = sum([sum([2 * (1 - kernel_function(s, x, g[i], p)) for x in P[i]]) for i in range(n_clusters)])

    return g, s, [np.where(y == i) for i in range(n_clusters)], y, obj


norm_data, n_clusters, n_features = normalize_data(view_complete), len(labels), view_complete.shape[1]
gamma = compute_gamma(norm_data)

best_obj, best_ari, best_g, best_s, best_y, best_P = 999999, 0, [], [], [], []

for i in range(100):
    g_final, s_final, P_final, y_final, obj = kernel_c_means(norm_data, n_clusters, gamma)

    if obj < best_obj:
        best_obj, best_g, best_s, best_P, best_y = obj, np.copy(g_final), np.copy(s_final), np.copy(P_final), np.copy(
            y_final)
        best_ari = adjusted_rand_score(y_numbers, best_y)

t_s1 = np.array(t_s1)
t_s2 = np.array(t_s2)
t_s3 = np.array(t_s3)

print(np.average(t_s1), np.average(t_s2), np.average(t_s3))

with open('output.txt', 'w') as output_file:
    output_file.write("Centers:\n")
    output_file.write(str(best_g))
    output_file.write("\n\nWidth hyperparameters:\n")
    output_file.write(str(best_s))
    output_file.write("\n\nPartition:\n")
    output_file.write(str(best_P))
    output_file.write("\n\nARI:\n")
    output_file.write(str(best_ari))
