#
# Obtain hyperspherical prototypes prior to network training.
#
# @inproceedings{mettes2016hyperspherical,
#  title={Hyperspherical Prototype Networks},
#  author={Mettes, Pascal and van der Pol, Elise and Snoek, Cees G M},
#  booktitle={Advances in Neural Information Processing Systems},
#  year={2019}
# }
#
import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import torch
from sklearn.preprocessing import normalize

#This code heavily adopted from https://proceedings.neurips.cc/paper/2019/file/02a32ad2669e6fe298e607fe7cc0e1a0-Paper.pdf
#Original Repo: https://github.com/psmmettes/hpn
#
def parse_args():
    parser = argparse.ArgumentParser(description="Hyperspherical prototypes")
    parser.add_argument('-c', dest="cluster", default=300, type=int)
    parser.add_argument('-d', dest="dims", default=128, type=int)
    parser.add_argument('-l', dest="learning_rate", default=0.1, type=float)
    parser.add_argument('-m', dest="momentum", default=0.9, type=float)
    parser.add_argument('-e', dest="epochs", default=500000, type=int, )
    parser.add_argument('-s', dest="seed", default=300, type=int)
    parser.add_argument('-r', dest="resdir", default="", type=str)
    parser.add_argument('-w', dest="wtvfile", default="", type=str)
    parser.add_argument('-n', dest="nn", default=2, type=int)
    args = parser.parse_args()
    return args


#
# Compute the loss related to the prototypes.
#
def norm_prots(prototypes):
    with torch.no_grad():
        w = prototypes.data.clone()
        w = nn.functional.normalize(w, dim=1, p=2)
        prototypes.copy_(w)
    return prototypes

def prototype_loss(prototypes):
    prototypes=norm_prots(prototypes)
    # Dot product of normalized prototypes is cosine similarity.
    product = torch.matmul(prototypes, prototypes.t()) + 1
    # Remove diagnonal from loss.
    product -= 2. * torch.diag(torch.diag(product))
    # Minimize maximum cosine similarity.
    loss = product.max(dim=1)[0]
    return loss.mean(), product.max()


#
# Compute the semantic relation loss.
#
def prototype_loss_sem(prototypes, triplets):
    product = torch.matmul(prototypes, prototypes.t()) + 1
    product -= 2. * torch.diag(torch.diag(product))
    loss1 = -product[triplets[:, 0], triplets[:, 1]]
    loss2 = product[triplets[:, 2], triplets[:, 3]]
    return loss1.mean() + loss2.mean(), product.max()


def plot_prot(fname,prots_data):
    X_embedded = TSNE(n_components=2).fit_transform(prots_data)
    plt.scatter(X_embedded[:,0],X_embedded[:,1])
    fig_path=fname+'.png'
    plt.savefig(fig_path)
    plt.clf()

#
# Main entry point of the script.
#
if __name__ == "__main__":
    # Parse user arguments.
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")
    kwargs = {'num_workers': 64, 'pin_memory': True}

    # Set seed.
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    use_wtv = False

    # Initialize prototypes.
    prototypes = torch.randn(args.cluster, args.dims)
    prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
    # optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
    #                       momentum=args.momentum)

    optimizer = optim.Adam([prototypes], lr=args.learning_rate)

    # Optimize for separation.
    for i in range(args.epochs):
        # Compute loss.
        loss1, sep = prototype_loss(prototypes)
        loss = loss1
        # Update.
        loss.backward()
        optimizer.step()
        # Renormalize prototypes.
        prototypes = nn.Parameter(F.normalize(prototypes, p=2, dim=1))
        optimizer = optim.SGD([prototypes], lr=args.learning_rate, \
                              momentum=args.momentum)
        print ("%03d/%d: %.4f\r" % (i, args.epochs, sep))
    # Store result.
    prototypes = norm_prots(prototypes)
    fname="fls/prototypes-%dd-%dc_%d" % (args.dims, args.cluster,args.epochs)
    np.save(args.resdir + fname+'.npy', prototypes.data.numpy())
    plot_prot(fname,prototypes.data)
