import sys, os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import numpy as np

genres = ['cellphone', 'home', 'electronics', 'watches', 'outdoor']

asps_map = {'service':0, 'price':1, 'appearance':2, 'other':3, 'functionality':4, 'quality':5, 'brand':6, 'usability':7}
jet = plt.get_cmap('jet'); cNorm  = colors.Normalize(vmin=0, vmax=7)
cjet = cmx.ScalarMappable(norm=cNorm, cmap=jet)

num_asp = 8

for g in genres:

    print 'processing', g

    fname_lasp = '../dataset/labels/{0}.asps'.format(g)
    fname_xofy = '../dataset/labels/{0}.xofy'.format(g)

    hss=[]; asps=[]; num=0
    with open(fname_xofy) as fin_x, open(fname_lasp) as fin_l:
        xofys = fin_x.readlines()
        lasps = fin_l.readlines()
        num = len(xofys)

        for i in range(num):
            xofy=xofys[i].strip(); lasp=lasps[i].strip().split()
            ts=xofy.split('/'); x=float(ts[0]); y=float(ts[1]); hss.append(x/y)
            idxs=[0]*num_asp
            for asp in lasp:
                idxs[asps_map[asp]] = 1
            asps.append(idxs)

    asps = np.array(asps)

    step_size = 100; xs=[x/1000.0 for x in range(int(step_size*0.5), 1000, step_size)]; bins=len(xs)
    hist = np.histogram(hss, bins=bins)[0]
    hist = np.divide(hist, num*1.0)

    hist_map = {}#'overall': ys}
    for asp in asps_map.keys():
        val = asps_map[asp]
        asp_c = asps[:,val]
        idxs = np.where(asp_c>0)[0]
        ss = [hss[idx] for idx in idxs]

        if len(ss) == 0:
            hist_map[asp] = []
            continue
        hist = np.histogram(ss, bins=bins)[0]
        hist = np.divide(hist, len(ss)*1.0)
        hist_map[asp] = hist

    plt.figure()
    i = 0
    for asp in hist_map.keys():
        ys = hist_map[asp]
        if len(ys) == len(xs):
            plt.plot(xs, ys, color=cjet.to_rgba(i), label=asp, linewidth=2)
        i += 1
    plt.legend(loc=0)
    plt.xlim([0,1])
    plt.title(g)
    plt.savefig('{0}_hs_dist.png'.format(g))

