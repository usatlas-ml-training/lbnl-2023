import json
import os
import pickle as pkl
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve


def save_model(outpath, model_kwargs, kernel_sizes, fc_size, dropout, depth, overwrite, n_epochs, lr, batch_size, nearest):
    if not overwrite and os.path.isfile(f"{outpath}/best_epoch_weights.pth"):
        print(f"model in {outpath} already exists, please delete it")
        sys.exit(0)

    filelist = [f for f in os.listdir(outpath) if not f.endswith(".txt")]  # don't remove the logs.txt

    for f in filelist:
        try:
            os.remove(os.path.join(outpath, f))
        except Exception:
            shutil.rmtree(os.path.join(outpath, f))

    with open(f"{outpath}/model_kwargs.pkl", "wb") as f:  # dump model architecture
        pkl.dump(model_kwargs, f, protocol=pkl.HIGHEST_PROTOCOL)

    with open(f"{outpath}/hyperparameters.json", "w") as fp:  # dump hyperparameters
        json.dump(
            {
                "n_epochs": n_epochs,
                "lr": lr,
                "batch_size": batch_size,
                "nearest": nearest,
                "kernel_sizes": kernel_sizes,
                "fc_size": fc_size,
                "dropout": dropout,
                "depth": depth,
            },
            fp,
        )


def load_data(dataset_path, flag, n_files, quick):
    if quick:  # use only 1000 events for train
        print(f"--- loading only 1000 events for quick {flag}")
        data = torch.load(f"{dataset_path}/{flag}/processed/data_0.pt")[:1000]
    else:
        data = []
        for i in range(n_files):
            data += torch.load(f"{dataset_path}/{flag}/processed/data_{i}.pt")
            print(f"--- loaded file {i} for {flag}")

    return data


def make_roc(y_test, y_score, path):
    fpr, tpr, _ = roc_curve(y_test, y_score)

    fig, ax = plt.subplots()
    ax.plot(
        tpr,
        fpr,
        color="darkorange",
        lw=2,
        label=f"AUC = {round(auc(fpr, tpr)*100,2)}%",
    )
    plt.xlim([0.0, 1.0])
    plt.ylabel("False Positive Rate")
    plt.xlabel("True Positive Rate")
    plt.yscale("log")
    plt.legend(loc="lower right")
    plt.savefig(path)


def make_dr_Mij_plots(outpath, epoch, Top_N=5):
    """
    Computes and plots the deltaR and the invariant mass of the Top_N relevant edges for the model at a given epoch.

    Args:
        outpath (str): Path to load the Rscores pkl files.
        epoch (int): The epoch at which to load the model.
            (if -1: best trained model, if 0: untrained model, otherwise loads the corresponding epoch)
        Top_N (int): The number of relevant edges to look at.

    Returns:
        subjet_idx (np.array): NumPy array of shape ``[num_particles]`` with elements
                                representing which subjet the particle belongs to.

    """

    if epoch == -1:
        PATH = f"{outpath}/Rscores_best"
        legend_title = "Trained model"
        save_as = "best"
    elif epoch == 0:
        PATH = f"{outpath}/Rscores_untrained"
        legend_title = "Untrained model"
        save_as = "untrained"
    else:
        PATH = f"{outpath}/Rscores_epoch_{epoch}"
        legend_title = f"Model at epoch {epoch}"
        save_as = f"epoch_{epoch}"

    # load the jet information
    with open(f"{PATH}/batch_x.pkl", "rb") as handle:
        batch_x_list = pkl.load(handle)
    with open(f"{PATH}/batch_y.pkl", "rb") as handle:
        batch_y_list = pkl.load(handle)
    with open(f"{PATH}/batch_p4.pkl", "rb") as handle:
        batch_p4_list = pkl.load(handle)

    # load the edgeRscores and edge_index of each EdgeConv block
    with open(f"{PATH}/R_edges.pkl", "rb") as handle:
        R_edges_list = pkl.load(handle)
    with open(f"{PATH}/edge_index.pkl", "rb") as handle:
        edge_index_list = pkl.load(handle)

    dr_top, dr_qcd = [], []
    Mij_top, Mij_qcd = [], []

    for i in range(len(batch_y_list)):
        # define the jet information
        jet_label = batch_y_list[i]

        eta = batch_x_list[i][:, 0]
        phi = batch_x_list[i][:, 1]

        px = batch_p4_list[i][:, 0]
        py = batch_p4_list[i][:, 1]
        pz = batch_p4_list[i][:, 2]
        e = batch_p4_list[i][:, 3]

        # define the edgeRscores and the edge_index of the last EdgeConv block
        edge_Rscores = R_edges_list[i]["edge_conv_2"]
        edge_index = edge_index_list[i]["edge_conv_2"]

        def deltaR(eta1, eta2, phi1, phi2):
            return torch.sqrt(torch.square(eta2 - eta1) + torch.square(phi2 - phi1))

        # pick the Top_N edge Rscores, get the indices
        top_edges = torch.topk(edge_Rscores, Top_N).indices

        for edge in top_edges:
            particle_1 = edge_index[0][edge]
            particle_2 = edge_index[1][edge]

            px1 = px[particle_1]
            py1 = py[particle_1]
            pz1 = pz[particle_1]

            px2 = px[particle_2]
            py2 = py[particle_2]
            pz2 = pz[particle_2]

            eta1 = eta[particle_1]
            eta2 = eta[particle_2]

            phi1 = phi[particle_1]
            phi2 = phi[particle_2]

            e1 = e[particle_1]
            e2 = e[particle_2]

            M12 = torch.sqrt(
                torch.square(e1 + e2) - torch.square(px1 + px2) - torch.square(py1 + py2) - torch.square(pz1 + pz2)
            )

            if jet_label == 1:
                dr_top.append(deltaR(eta1, eta2, phi1, phi2).item())
                Mij_top.append(M12)
            else:
                dr_qcd.append(deltaR(eta1, eta2, phi1, phi2).item())
                Mij_qcd.append(M12)

    # save the results
    if not os.path.exists(f"{outpath}/dr_plots"):
        os.makedirs(f"{outpath}/dr_plots")
    with open(f"{outpath}/dr_plots/dr_{save_as}_top.pkl", "wb") as f:
        pkl.dump(dr_top, f)
    with open(f"{outpath}/dr_plots/dr_{save_as}_qcd.pkl", "wb") as f:
        pkl.dump(dr_qcd, f)

    if not os.path.exists(f"{outpath}/Mij_plots"):
        os.makedirs(f"{outpath}/Mij_plots")
    with open(f"{outpath}/Mij_plots/Mij_{save_as}_top.pkl", "wb") as f:
        pkl.dump(Mij_top, f)
    with open(f"{outpath}/Mij_plots/Mij_{save_as}_qcd.pkl", "wb") as f:
        pkl.dump(Mij_qcd, f)

    # make the dr plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        dr_top,
        bins=np.linspace(0, 1.5, 30),
        label="Top",
        color="Blue",
        histtype="step",
        linewidth=1,
        density=True,
    )
    ax.hist(
        dr_qcd,
        bins=np.linspace(0, 1.5, 30),
        label="QCD",
        color="Orange",
        histtype="step",
        linewidth=2,
        density=True,
    )
    ax.legend(title=legend_title)
    ax.set_yscale("log")
    ax.set_ylim(0, 20)
    ax.set_xlim(0, 1.6)
    ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_xlabel(r"$\Delta$R of the 5 most relevant edges")
    ax.set_ylabel("Normalized counts")
    fig.tight_layout()
    plt.savefig(f"{outpath}/dr_plots/dr_{save_as}.pdf")
    print(f"- saved the plot as {outpath}/dr_plots/dr_{save_as}.pdf")

    # make the Mij plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        Mij_top,
        bins=np.linspace(0, 100),
        label="Top",
        color="Blue",
        histtype="step",
        linewidth=1,
        density=True,
    )
    ax.hist(
        Mij_qcd,
        bins=np.linspace(0, 100),
        label="QCD",
        color="Orange",
        histtype="step",
        linewidth=2,
        density=True,
    )
    ax.legend(title=legend_title)
    ax.set_yscale("log")
    ax.set_ylim(0, 20)
    # ax.set_xlim(0, 100)
    # ax.set_xticks([0, 0.5, 1, 1.5])
    ax.set_xlabel("Invariant mass of the 5 most relevant edges")
    ax.set_ylabel("Normalized counts")
    fig.tight_layout()
    plt.savefig(f"{outpath}/Mij_plots/Mij_{save_as}.pdf")
    print(f"- saved the plot as {outpath}/Mij_plots/Mij_{save_as}.pdf")
