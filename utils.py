#! /usr/bin/python
# -*- coding: utf-8 -*-
## @Author: marcuschen
## @File: utils.py
## @Created Time: Mon Apr  5 16:09:46 2021
## @Description:
import torch 
import pybel
import pandas as pd 
import openbabel as ob 


def calc_dist(mol, pos, with_bond=False):
    min_dist = 100
    if with_bond:
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            dist = (pos[begin] - pos[end]).pow(2).sum().sqrt().item()
            if dist < min_dist:
                min_dist = dist 
    else:
        n = pos.size(0)
        for i in range(n - 1):
            for j in range(i + 1, n):
                dist = (pos[i] - pos[j]).pow(2).sum().sqrt().item()
                if dist < min_dist: 
                    min_dist = dist 
    return min_dist


def calc_dist_min(mol, pos, ignore_bond=False):
    if not ignore_bond:
        bonds = ob.OBMolBondIter(mol)
        dist_list = []
        for bond in bonds:
            begin_atom_idx = bond.GetBeginAtomIdx() - 1 # zero-indexed 
            end_atom_idx = bond.GetEndAtomIdx() - 1 # zero-indexed 
            #begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
            #assert torch.all(pos[begin_atom_idx] == torch.tensor([begin_atom.x(), begin_atom.y(), begin_atom.z()], dtype=torch.float)).item() == True 
            #assert torch.all(pos[end_atom_idx] == torch.tensor([end_atom.x(), end_atom.y(), end_atom.z()], dtype=torch.float)).item() == True
            dist = calc_dist(pos[begin_atom_idx], pos[end_atom_idx]) 
            dist_list.append(dist.item())
        if len(dist_list) > 0:
            return min(dist_list)
        else:
            return None
    else:
        dist_list = [] 
        n = pos.size(0)
        for i in range(n - 1):
            for j in range(i + 1, n):
                assert i != j 
                dist = calc_dist(pos[i], pos[j]) 
                dist_list.append(dist)
        if len(dist_list) > 0:
            return min(dist_list) 
        else:
            return None 


def load_smiles_list(filepath):
    df = pd.read_csv(filepath, index_col=False) 
    smiles_list = df.smiles.tolist() 
    return smiles_list 


class OpenBabelCalculator(object):
    def __init__(self, mol, forcefield, steps=50, with_coordinates=False, removehs=False):
        super(OpenBabelCalculator, self).__init__() 
        self.mol = mol 
        self.steps = steps 
        self.removehs = removehs
        self.py_mol = pybel.Molecule(mol) 
        if not with_coordinates:
            self.pos = self.__calc_pos(forcefield, steps)
        else:
            if removehs:
                self.py_mol.removeh()
            self.pos = torch.tensor([atom.coords for atom in self.py_mol.atoms], dtype=torch.float)
        self.dist_list_w_bond = self.__calc_dist(with_bond=True)
        self.dist_list_wo_bond = self.__calc_dist(with_bond=False)

    def __calc_pos(self, forcefield, steps):
        self.py_mol.make3D(forcefield, steps)
        if self.removehs:
            self.py_mol.removeh()
        pos = torch.tensor([atom.coords for atom in self.py_mol.atoms], dtype=torch.float)
        assert pos.size(0) == len(self.py_mol.atoms)
        return pos 

    def __calc_dist(self, with_bond=True):
        dist_list = []
        if with_bond:
            bonds = ob.OBMolBondIter(self.mol)
            for bond in bonds:
                begin_atom_idx = bond.GetBeginAtomIdx() - 1 
                end_atom_idx = bond.GetEndAtomIdx() - 1
                dist = (self.pos[begin_atom_idx] - self.pos[end_atom_idx]).pow(2).sum(dim=-1).sqrt().item()
                dist_list.append(dist)
        else:
            pos = self.pos
            n = self.pos.size(0)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    dist = (pos[i] - pos[j]).pow(2).sum().sqrt().item()
                    dist_list.append(dist) 
            assert len(dist_list) == n * (n - 1) / 2
        return dist_list

    def get_dist_list(self, with_bond=True): 
        if with_bond:
            return self.dist_list_w_bond 
        else:
            return self.dist_list_wo_bond 

    def get_min_dist(self, with_bond=True): 
        try:
            return min(self.get_dist_list(with_bond=with_bond))
        except ValueError as e:
            return None

    def get_pymol(self):
        return self.py_mol

    def get_pos(self):
        return self.pos


class RDKitCalculator(object):
    def __init__(self, mol, item_text=None):
        self.mol = mol 
        self.pos = self.__calc_pos(item_text=item_text)
        self.dist_list_w_bond = self.__calc_dist(with_bond=True)
        self.dist_list_wo_bond = self.__calc_dist(with_bond=False)

    def __calc_pos(self, item_text=None):
        if item_text is None:
            AllChem.EmbedMolecule(mol, randomSeed=0xf00d) 
            block = Chem.MolToMolBlock(mol) 
            N = mol.GetNumAtoms()
            pos = parse_block(block, N) 
            self.mol = mol 
        else:
            N = self.mol.GetNumAtoms()
            pos = item_text.split('\n')[4:4 + N]
            pos = [[float(x) for x in line.split()[:3]] for line in pos]
            pos = torch.tensor(pos, dtype=torch.float)
        return pos 

    def __calc_dist(self, with_bond=True):
        dist_list = []
        if with_bond:
            for bond in self.mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                dist = (self.pos[start] - self.pos[end]).pow(2).sum(dim=-1).sqrt().item()
                dist_list.append(dist) 
        else:
            n = self.pos.size(0) 
            for i in range(n - 1):
                for j in range(i + 1, n):
                    dist = (self.pos[i] - self.pos[j]).pow(2).sum().sqrt().item()
                    dist_list.append(dist)
            assert len(dist_list) == n * (n - 1) / 2 
        return dist_list 

    def get_dist_list(self, with_bond=True):
        if with_bond:
            return self.dist_list_w_bond
        else:
            return self.dist_list_wo_bond

    def get_min_dist(self, with_bond=True):
        return min(self.get_dist_list(with_bond=with_bond))

    def get_mol(self):
        return self.mol

