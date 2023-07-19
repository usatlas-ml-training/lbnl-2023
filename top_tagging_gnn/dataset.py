import os.path as osp
from glob import glob
import requests

import torch
from torch_geometric.data import Data, Dataset


class TopTaggingDataset(Dataset):
    """
    Initialize parameters of graph dataset
    Args:
        root (str): path
    """

    def __init__(self, root, mode, transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode  # train/val/test
        super().__init__(root, transform, pre_transform, pre_filter)

    def download(self):
        raw_file = f"{self.root}/raw/{self.mode}.h5"
        if not osp.exists(raw_file):
            print(f"Downloading {raw_file} from Zenodo")
            url = f"https://zenodo.org/record/2603256/files/{self.mode}.h5?download=1"
            r = requests.get(url)
            open(raw_file, "wb").write(r.content)

    @property
    def raw_file_names(self):
        return [f"{self.mode}.h5"]

    @property
    def processed_file_names(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return sorted([processed_path.replace(self.processed_dir, ".") for processed_path in proc_list])

    def prepare_ptfiles(self):
        """
        Loads the H5 file and splits it up into pt files for optimized PyTorch training/inference
        Transforms the input variables and stores only the following 7 features per particle
            1. ∆η difference in pseudorapidity between the particle and the jet axis
            2. ∆φ difference in azimuthal angle between the particle and the jet axis
            3. log pT logarithm of the particle's pT
            4. log E logarithm of the particle's energy
            5. log pT/pT_jet logarithm of the particle's pT relative to the jet pT
            6. log E/E_jet logarithm of the particle's energy relative to the jet energy
            7. ∆R
        """

        # https://github.com/jet-universe/particle_transformer/blob/main/utils/convert_top_datasets.py

        import awkward as ak
        import numpy as np
        import pandas as pd
        import vector

        vector.register_awkward()

        df = pd.read_hdf(f"{self.root}/raw/{self.mode}.h5", key="table")

        def _col_list(prefix, max_particles=200):
            return ["%s_%d" % (prefix, i) for i in range(max_particles)]

        _px = df[_col_list("PX")].values
        _py = df[_col_list("PY")].values
        _pz = df[_col_list("PZ")].values
        _e = df[_col_list("E")].values

        mask = _e > 0
        n_particles = np.sum(mask, axis=1)

        px = ak.unflatten(_px[mask], n_particles)
        py = ak.unflatten(_py[mask], n_particles)
        pz = ak.unflatten(_pz[mask], n_particles)
        energy = ak.unflatten(_e[mask], n_particles)

        p4 = ak.zip(
            {
                "px": px,
                "py": py,
                "pz": pz,
                "energy": energy,
            },
            with_name="Momentum4D",
        )

        jet_p4 = ak.sum(p4, axis=-1)

        # outputs
        v = {}
        v["label"] = df["is_signal_new"].values

        v["jet_pt"] = jet_p4.pt.to_numpy()
        v["jet_eta"] = jet_p4.eta.to_numpy()
        v["jet_phi"] = jet_p4.phi.to_numpy()
        v["jet_energy"] = jet_p4.energy.to_numpy()
        v["jet_mass"] = jet_p4.mass.to_numpy()
        v["jet_nparticles"] = n_particles

        v["part_px"] = px
        v["part_py"] = py
        v["part_pz"] = pz
        v["part_energy"] = energy

        v["part_deta"] = p4.deltaeta(jet_p4)
        v["part_dphi"] = p4.deltaphi(jet_p4)

        # https://github.com/jet-universe/particle_transformer/blob/main/data/TopLandscape/top_kin.yaml
        part_pt = np.hypot(v["part_px"], v["part_py"])
        part_pt_log = np.log(part_pt)
        part_e_log = np.log(v["part_energy"])
        part_logptrel = np.log(part_pt / v["jet_pt"])
        part_logerel = np.log(v["part_energy"] / v["jet_energy"])
        part_deltaR = np.hypot(v["part_deta"], v["part_dphi"])

        data = []
        c = 0
        for jet_index in range(len(df - 1)):
            data.append(
                Data(
                    x=torch.cat(
                        [
                            torch.from_numpy(v["part_deta"][jet_index].to_numpy().reshape(-1, 1)),
                            torch.from_numpy(v["part_dphi"][jet_index].to_numpy().reshape(-1, 1)),
                            torch.from_numpy(part_pt_log[jet_index].to_numpy().reshape(-1, 1)),
                            torch.from_numpy(part_e_log[jet_index].to_numpy().reshape(-1, 1)),
                            torch.from_numpy(part_logptrel[jet_index].to_numpy().reshape(-1, 1)),
                            torch.from_numpy(part_logerel[jet_index].to_numpy().reshape(-1, 1)),
                            torch.from_numpy(part_deltaR[jet_index].to_numpy().reshape(-1, 1)),
                        ],
                        axis=1,
                    ),
                    y=torch.tensor(v["label"][jet_index]).long(),
                )
            )

            if self.mode == "test":  # add (px,py,pz,E) info for lrp fastjet tests
                data[-1]["px"] = torch.from_numpy(v["part_px"][jet_index])
                data[-1]["py"] = torch.from_numpy(v["part_py"][jet_index])
                data[-1]["pz"] = torch.from_numpy(v["part_pz"][jet_index])
                data[-1]["E"] = torch.from_numpy(v["part_energy"][jet_index])

            if jet_index % 100000 == 0 and jet_index != 0:
                print(f"saving datafile data_{c}")
                torch.save(data, f"{self.processed_dir}/data_{c}.pt")
                c += 1
                data = []

    def len(self):
        proc_list = glob(osp.join(self.processed_dir, "*.pt"))
        return len(sorted([processed_path.replace(self.processed_dir, ".") for processed_path in proc_list]))

    def get(self, idx):
        data = torch.load(
            osp.join(self.processed_dir, "data_{}.pt".format(idx)),
            map_location="cpu",
        )
        return data

    def __getitem__(self, idx):
        return self.get(idx)


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Input data path")
    parser.add_argument("--mode", type=str, required=True, help="'train' or 'val' or 'test'?")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """
    To process train data: python dataset.py --dataset data/toptagging/ --mode train
    To process val data:   python dataset.py --dataset data/toptagging/ --mode val
    To process test data:  python dataset.py --dataset data/toptagging/ --mode test
    """

    args = parse_args()

    topdataset = TopTaggingDataset(root=f"{args.dataset}/{args.mode}", mode=args.mode)
    topdataset.prepare_ptfiles()
