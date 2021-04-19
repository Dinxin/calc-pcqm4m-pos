#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: marcuschen
## @File: basic.py
## @Created Time: Mon Apr  5 11:36:14 2021
## @Description:
import os 
import pdb
import torch 
import logging 
import numpy as np 
import pandas as pd 
import openbabel as ob 
import pybel 
from rdkit import Chem 
from rdkit.Chem import AllChem
from pybel import Outputfile
from argparse import ArgumentParser
import multiprocessing
from multiprocessing import Pool
import sys 
import os.path as osp
root = osp.dirname(osp.abspath(__file__))
print(root)
sys.path.insert(0, root)
from utils import calc_dist_min, load_smiles_list, OpenBabelCalculator, RDKitCalculator, calc_dist

logging.basicConfig(format="%(asctime)s, %(message)s", level=logging.INFO)

conversion = ob.OBConversion()
conversion.SetInAndOutFormats("smi", "mdl")
smi_conversion = ob.OBConversion()
smi_conversion.SetInAndOutFormats("smi", "can")


def parse_args():
    parser = ArgumentParser(description="Basic statistical analysis for quantum chemistry dataset.")
    parser.add_argument("--dataset", type=str, default="qm9", choices=["qm9", "pcqm4m"], help="dataset name (default: qm9)")
    parser.add_argument("--subset_ratio", type=float, default=None, help="subset ratio (default: None for full dataset)")
    parser.add_argument("--start_idx", type=int, default=0, help="start index for dataset to be analyzed (default: 0)")
    parser.add_argument("--forcefield", type=str, default="mmff94", help="algorithm used for generating spatial information (default: mmff94)")
    parser.add_argument("--op", type=str, default="statis", choices=["statis", "check", "contrast", "add"], help="op (default: statis)")
    parser.add_argument("--max_size", type=int, default=-1, help="maximum size of dataset (default: -1 for all)")
    parser.add_argument("--threshold", type=float, default=0.9, help="threshold for minimum interatomic distance in a molecule (default: 0.9)")
    parser.add_argument("--max_steps", type=int, default=10000, help="maximum number of steps to calculate atomic coordinates (default: 10000)")
    parser.add_argument("--retry_times", type=int, default=3, help="maximum number of retries (default: 3)")
    parser.add_argument("--mp", action="store_true", help="whether to use multiprocessing (default: False)")
    parser.add_argument("--nprocs", type=int, default=8, help="number of procesors (default: 8)")
    parser.add_argument("--removehs", action="store_true", help="whether to remove hydrogen atoms (default: False)")
    args = parser.parse_args()
    return args 


# helper functions
def parse_block(block, N):
    pos = block.split("\n")[4: 4 + N]
    pos = [[float(x) for x in line.split()[:3]] for line in pos]
    pos = torch.tensor(pos, dtype=torch.float)
    return pos 


def optimize(smiles):
    ob_mol = ob.OBMol()
    conversion.ReadString(ob_mol, smiles)
    steps = 50 
    w_bond_list = []
    wo_bond_list = []
    py_mol_list = []
    pos_list = []
    while steps < args.max_steps:
        ob_calc = OpenBabelCalculator(ob_mol, forcefield=args.forcefield, steps=steps)
        curr_min_dist_w_bond = ob_calc.get_min_dist(with_bond=True)
        curr_min_dist_wo_bond = ob_calc.get_min_dist(with_bond=False)
        py_mol = ob_calc.get_pymol()
        pos = ob_calc.get_pos()
        w_bond_list.append(curr_min_dist_w_bond) 
        wo_bond_list.append(curr_min_dist_wo_bond)
        py_mol_list.append(py_mol)
        pos_list.append(pos)
        steps *= 2
    max_idx = wo_bond_list.index(max(wo_bond_list))
    max_py_mol = py_mol_list[max_idx]
    max_pos = pos_list[max_idx]
    return max_py_mol, max_pos, w_bond_list[max_idx], wo_bond_list[max_idx]


def proc_one_smiles(smiles):
    ob_mol = ob.OBMol() 
    conversion.ReadString(ob_mol, smiles)
    ob_calc = OpenBabelCalculator(ob_mol, forcefield=args.forcefield, removehs=args.removehs) 
    py_mol = ob_calc.get_pymol()
    pos = ob_calc.get_pos()

    curr_min_dist_w_bond = ob_calc.get_min_dist(with_bond=True)
    curr_min_dist_wo_bond = ob_calc.get_min_dist(with_bond=False)

    if curr_min_dist_w_bond is None or curr_min_dist_wo_bond is None:
        pos = pos.numpy()
        return pos, None, None, smiles 

    if curr_min_dist_wo_bond < args.threshold:
        logging.info("Invalid: Minimum interatomic distance without bonds: {:.4f}, Smiles: {} ...".format(curr_min_dist_wo_bond, smiles))
        retry = 0
        curr_min_dist_wo_bond_ = curr_min_dist_wo_bond
        while curr_min_dist_wo_bond_ < args.threshold and retry <= args.retry_times:
            py_mol_, pos_, curr_min_dist_w_bond_, curr_min_dist_wo_bond_ = optimize(smiles)
            retry += 1

        py_mol = py_mol_
        pos = pos_
        curr_min_dist_w_bond = curr_min_dist_w_bond_ 
        curr_min_dist_wo_bond = curr_min_dist_wo_bond_
        logging.info("  Valid: Minimum interatomic distance without bonds: {:.4f}, Smiles: {} ...".format(curr_min_dist_wo_bond, smiles))

    pos = pos.numpy()
    return pos, curr_min_dist_w_bond, curr_min_dist_wo_bond, smiles


def statis_qm9_mp(filepath):
    pool = multiprocessing.Pool(args.nprocs)
    logging.info("Generating smiles ...")
    supplier = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=False)
    smiles_list = [Chem.MolToSmiles(mol) for mol in supplier]
    logging.info("Generate done ...")

    fail_count = 0
    smiles2idx = {smiles: idx for idx, smiles in enumerate(smiles_list)}
    smiles2pos = {}
    wo_bond = 100 
    w_bond = 100 
    for pos, curr_w_bond, curr_wo_bond, smiles in pool.imap(proc_one_smiles, smiles_list):
        if smiles2idx[smiles] % 1000 == 0:
            logging.info("Processing idx: {}, smiles: {}, fail count so far: {} ...".format(
                smiles2idx[smiles], smiles, fail_count))
        if curr_wo_bond < args.threshold:
            fail_count += 1
        assert isinstance(pos, np.ndarray)
        smiles2pos[smiles] = pos
        if curr_wo_bond < wo_bond: 
            wo_bond = curr_wo_bond 
        if curr_w_bond < w_bond:
            w_bond = curr_w_bond 
        if smiles2idx[smiles] % 1000 == 0:
            logging.info("minimum interatomic distance with    bonds so far: {:.4f} ...".format(w_bond)) 
            logging.info("minimum interatomic distance without bonds so far: {:.4f} ...".format(wo_bond))
    
    new_smiles2pos = {}
    for smiles, pos in smiles2pos.items():
        new_smiles2pos[smiles] = torch.from_numpy(pos) 
    smiles2pos = new_smiles2pos 
    del new_smiles2pos 
    out_filepath = "./qm9_pos.pt" 
    torch.save(smiles2pos, out_filepath)


def statis_qm9(filepath):

    supplier = Chem.SDMolSupplier(filepath, removeHs=False, sanitize=False)

    w_bond = 100
    wo_bond = 100 
    smiles2pos = {}
    for i, mol in enumerate(supplier):
        if i < args.start_idx: 
            continue
        try:
            smiles = Chem.MolToSmiles(mol)
        except:
            logging.info("Processing mol {}, parse smiles fail ...".format(i))
            smiles = None

        if i % 1000 == 0:
            logging.info("Processing mol {}, smiles: {} ...".format(i, smiles))
        
        # ob_mol: OpenBabel's molecule object 
        ob_mol = ob.OBMol() 
        conversion.ReadString(ob_mol, smiles) 
        ob_calc = OpenBabelCalculator(ob_mol, forcefield=args.forcefield, removehs=args.removehs) 
        py_mol = ob_calc.get_pymol()
        pos = ob_calc.get_pos()

        # sanitize
        if mol is None:
            logging.info("idx: {}, smiles: {} cannot be parsed ...".format(i, can_smiles))
            continue
        N = mol.GetNumAtoms()
        try:
            assert N == pos.size(0)
        except:
            pdb.set_trace()
            tmp = 1

        curr_min_dist_w_bond = ob_calc.get_min_dist(with_bond=True)
        curr_min_dist_wo_bond = ob_calc.get_min_dist(with_bond=False)

        if curr_min_dist_wo_bond < args.threshold:
            logging.info("Invalid: Minimum interatomic distance without bonds: {:.4f}, Smiles: {} ...".format(curr_min_dist_wo_bond, smiles))
            retry = 0
            curr_min_dist_wo_bond_ = curr_min_dist_wo_bond
            while curr_min_dist_wo_bond_ < args.threshold and retry <= args.retry_times:
                py_mol_, pos_, curr_min_dist_w_bond_, curr_min_dist_wo_bond_ = optimize(smiles)
                retry += 1

            py_mol = py_mol_
            pos = pos_
            curr_min_dist_w_bond = curr_min_dist_w_bond_ 
            curr_min_dist_wo_bond = curr_min_dist_wo_bond_
            logging.info("  Valid: Minimum interatomic distance without bonds: {:.4f}, Smiles: {} ...".format(curr_min_dist_wo_bond, smiles))
        
        min_dist = calc_dist(py_mol, pos, with_bond=False)
        try:
            assert min_dist == curr_min_dist_wo_bond
        except: 
            logging.info("Fail smiles: {} ...".format(smiles))

        smiles2pos[smiles] = pos

        if curr_min_dist_w_bond < w_bond: 
            w_bond = curr_min_dist_w_bond 
        if curr_min_dist_wo_bond < wo_bond:
            wo_bond = curr_min_dist_wo_bond

        if i % 1000 == 0:
            logging.info("minimum interatomic distance with    bonds so far: {:.4f} ...".format(w_bond)) 
            logging.info("minimum interatomic distance without bonds so far: {:.4f} ...".format(wo_bond))

        if args.max_size >= 0 and i >= args.max_size:
            break 
    
    out_filepath = "./qm9_pos.pt"
    torch.save(smiles2pos, out_filepath)


def proc_pcqm4m_mp(smiles_list):
    w_bond = 100 
    wo_bond = 100 
    smiles2pos = {}
    fail_count = 0 
    fail_smiles = []
    smiles2idx = {smiles: idx for idx, smiles in enumerate(smiles_list)}
    pool = Pool(args.nprocs)
    for pos, curr_w_bond, curr_wo_bond, smiles in pool.imap(proc_one_smiles, smiles_list):
        if smiles2idx[smiles] % 1000 == 0:
            logging.info("Processing idx: {}, smiles: {}, fail count so far: {} ...".format(
                smiles2idx[smiles], smiles, fail_count)) 
        if curr_w_bond is None or curr_wo_bond is None:
            fail_count += 1 
            fail_smiles.append(smiles) 
            assert isinstance(pos, np.ndarray) 
            smiles2pos[smiles] = pos 
            if smiles2idx[smiles] % 1000 == 0:
                logging.info("Minimum interatomic distance with    bonds so far: {:.4f} ...".format(w_bond)) 
                logging.info("Minimum interatomic distance without bonds so far: {:.4f} ...".format(wo_bond))
            continue 
        if curr_wo_bond < args.threshold:
            fail_count += 1 
            fail_smiles.append(smiles)
        assert isinstance(pos, np.ndarray) 
        smiles2pos[smiles] = pos 
        if curr_wo_bond < wo_bond:
            wo_bond = curr_wo_bond 
        if curr_w_bond < w_bond:
            w_bond = curr_w_bond 
        if smiles2idx[smiles] % 1000 == 0:
            logging.info("Minimum interatomic distance with    bonds so far: {:.4f} ...".format(w_bond)) 
            logging.info("Minimum interatomic distance without bonds so far: {:.4f} ...".format(wo_bond))

    new_smiles2pos = {}
    for smiles, pos in smiles2pos.items():
        new_smiles2pos[smiles] = torch.from_numpy(pos) 
    smiles2pos = new_smiles2pos 
    del new_smiles2pos 

    return smiles2pos


def proc_pcqm4m_sp(smiles_list):
    w_bond = 100 
    wo_bond = 100 
    smiles2pos = {}
    fail_count = 0 
    fail_smiles = []

    for i, smiles in enumerate(smiles_list):
        if i % 1000 == 0:
            logging.info("Processing idx: {}, smiles: {}, fail count so far: {} ...".format(i, smiles, fail_count))

        # ob_mol: OpenBabel's molecule object 
        ob_mol = ob.OBMol() 
        conversion.ReadString(ob_mol, smiles) 
        ob_calc = OpenBabelCalculator(ob_mol, forcefield=args.forcefield, removehs=args.removehs) 
        curr_min_dist_w_bond = ob_calc.get_min_dist(with_bond=True)
        curr_min_dist_wo_bond = ob_calc.get_min_dist(with_bond=False)
        py_mol = ob_calc.get_pymol()
        pos = ob_calc.get_pos()
        pdb.set_trace()
        if curr_min_dist_w_bond is None or curr_min_dist_wo_bond is None:
            fail_count += 1
            fail_smiles.append(smiles) 
            smiles2pos[smiles] = pos 
            if i % 1000 == 0:
                logging.info("minimum interatomic distance with    bonds so far: {:.4f} ...".format(w_bond)) 
                logging.info("minimum interatomic distance without bonds so far: {:.4f} ...".format(wo_bond))
            continue

        if curr_min_dist_wo_bond < args.threshold: 
            logging.info("Invalid: Minimum interatomic distance without bonds: {:.4f}, Smiles: {} ...".format(curr_min_dist_wo_bond, smiles)) 
            retry = 0 
            curr_min_dist_wo_bond_ = curr_min_dist_wo_bond 
            while curr_min_dist_wo_bond_ < args.threshold and retry <= args.retry_times:
                py_mol_, pos_, curr_min_dist_w_bond_, curr_min_dist_wo_bond_ = optimize(smiles) 
                retry += 1 

            py_mol = py_mol_ 
            pos = pos_ 
            curr_min_dist_w_bond = curr_min_dist_w_bond_ 
            curr_min_dist_wo_bond = curr_min_dist_wo_bond_ 
            logging.info("  Valid: Minimum interatomic distance with    bonds: {:.4f}, Smiles: {} ...".format(curr_min_dist_wo_bond, smiles)) 

        min_dist = calc_dist(py_mol, pos, with_bond=False)
        try:
            assert min_dist == curr_min_dist_wo_bond 
        except:
            logging.info("Fail smiles: {} ...".format(smiles))
            fail_count += 1
            fail_smiles.append(smiles)

        smiles2pos[smiles] = pos 

        if curr_min_dist_w_bond < w_bond:
            w_bond = curr_min_dist_w_bond 
        if curr_min_dist_wo_bond < wo_bond:
            wo_bond = curr_min_dist_wo_bond 

        if i % 1000 == 0:
            logging.info("minimum interatomic distance with    bonds so far: {:.4f} ...".format(w_bond)) 
            logging.info("minimum interatomic distance without bonds so far: {:.4f} ...".format(wo_bond))

        if args.max_size >= 0 and i >= args.max_size:
            break 

    return smiles2pos


def statis_pcqm4m(filepath, subset_ratio=None, start_idx=0):
    logging.info("Loading smiles list ...")
    smiles_list = load_smiles_list(os.path.join("../dataset", "pcqm4m_kddcup2021", "raw", "data.csv")) 
    smiles_list = smiles_list[start_idx:]
    if subset_ratio is not None:
        smiles_list = smiles_list[start_idx: start_idx + int(subset_ratio * len(smiles_list))]
    logging.info("Number of smiles: {} ...".format(len(smiles_list)))

    smiles2pos = proc_pcqm4m_sp(smiles_list)

    out_filepath = "./pcqm4m_pos_sp.pt" 
    torch.save(smiles2pos, out_filepath)

    logging.info("Fail smiles are listed as follows: ")
    for idx, smiles in enumerate(fail_smiles):
        print(idx, smiles)


def statis_pcqm4m_mp(filepath, subset_ratio=None, start_idx=0):
    pool = multiprocessing.Pool(args.nprocs)
    logging.info("Loading smiles list ...") 
    # 路径需要修改
    smiles_list = load_smiles_list(os.path.join("../../dataset", "pcqm4m_kddcup2021", "raw", "data.csv"))
    smiles_list = smiles_list[start_idx:]
    if subset_ratio is not None:
        smiles_list = smiles_list[start_idx: start_idx + int(subset_ratio * len(smiles_list))]
    if args.max_size >= 0:
        smiles_list = smiles_list[start_idx: start_idx + args.max_size]
    logging.info("Number of smiles: {} ...".format(len(smiles_list)))
    
    smiles2pos = proc_pcqm4m_mp(smiles_list)

    out_filepath = "./pcqm4m_pos_mp.pt"
    torch.save(smiles2pos, out_filepath)

    logging.info("Fail smiles are listed as follows: ")
    for idx, smiles in enumerate(fail_smiles):
        print(idx, smiles)


def supplement_pcqm4m():
    logging.info("Loading smiles2pos ...")
    filepath = "./pcqm4m_pos_mp.pt"
    smiles2pos = torch.load(filepath)

    logging.info("Loading smiles list ...") 
    smiles_list = load_smiles_list(os.path.join("../dataset", "pcqm4m_kddcup2021", "raw", "data.csv"))
    smiles_list_with_pos = list(smiles2pos.keys())

    if len(smiles2pos) < len(smiles_list):
        smiles_list_wo_pos = list(set(smiles_list) - set(smiles_list_with_pos))
        pdb.set_trace()
        if args.mp:
            smiles2pos_wo_pos = proc_pcqm4m_mp(smiles_list_wo_pos)
        else:
            smiles2pos_wo_pos = proc_pcqm4m_sp(smiles_list_wo_pos)
        smiles2pos.update(smiles2pos_wo_pos)
        assert len(smiles2pos) == len(smiles_list)


def check_dist_min():
    logging.info("Loading ...")
    filepath = "../dataset/pcqm4m_kddcup2021/sdf/pybel_pos_mmff94.pt"
    smiles2pos = torch.load(filepath) 
    logging.info("Loaded ...")
    
    checked_smiles = [
            "O=C1C=NC2C(=N1)C(=C(C=C2)F)C", 
            "NC1=c2c(N=N1)cc(=C)n(c2=O)C1CC1", 
            "NC1=c2c(N=N1)cc(=C)n(c2=O)C(C)C", 
            "CC1=CCCC(=C(CC1)C)C", 
            "COC(=O)C1CC(=C2C1C(C)CCC=C2)C", 
            "O=C1C(=C2C=CCC=C2C=Cc2c1cccc2)C", 
            "CNC1=CC2=NC(=CC(=O)C2C=C1)NC", 
            "O=C1N=C2N(C1)C(=O)C1C(=N2)C(=CS1)C", 
            "OCC1=c2cc(Cl)ccc2=NC(=O)C1=C", 
            "COC(=O)C1=C(NCCC(=CC1)C)C(=O)OC", 
            ]
    
    best_dist_min = 100
    best_smiles = None 
    for idx, smiles in enumerate(checked_smiles):
        pos = smiles2pos[smiles]
        if idx % 1000 == 0:
            logging.info("Processing idx: {}, smiles: {} ...".format(idx, smiles))
        mol = ob.OBMol() 
        conversion.ReadString(mol, smiles) 
        mol.AddHydrogens() 
        N = mol.NumAtoms() 
        assert N == pos.size(0)
        dist_min = calc_dist_min(mol, pos) 
        dist_min_ = calc_dist_min(mol, pos, ignore_bond=True)
        logging.info("minimum interatomic distance with    bonds: {:.4f} ...".format(dist_min)) 
        logging.info("minimum interatomic distance without bonds: {:.4f} ...".format(dist_min_))
        if  dist_min is not None and dist_min < best_dist_min:
            best_dist_min = dist_min
            best_smiles = smiles 
            logging.info("Best dist min so far: {}, smiles is: {} ...".format(best_dist_min, best_smiles))
    logging.info("Best dist min: {}, smiles is: {} ...".format(best_dist_min, best_smiles))


def main():
    if args.dataset == "qm9":
        qm9_filepath = "../dataset/QM9/raw/gdb9.sdf"
        if args.mp:
            statis_qm9_mp(qm9_filepath)
        else:
            statis_qm9(qm9_filepath)
    else:
        pcqm4m_filepath = "../dataset/pcqm4m_kddcup2021/raw/data.csv"
        if args.mp:
            statis_pcqm4m_mp(pcqm4m_filepath, subset_ratio=args.subset_ratio, start_idx=args.start_idx)
        else:
            statis_pcqm4m(pcqm4m_filepath, subset_ratio=args.subset_ratio, start_idx=args.start_idx)


if __name__ == "__main__":
    args = parse_args()
    if args.op == "check":
        check_dist_min()
    elif args.op == "statis":
        main()
    elif args.op == "contrast":
        contrast_in_qm9()
    elif args.op == "add":
        supplement_pcqm4m()
        
