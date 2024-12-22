from __future__ import print_function, division

import torch
import torch.nn as nn
from model.cgcnn import CrystalGraphConvNet, ConvLayer

class PSCGNet(nn.Module):
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False):
        super(PSCGNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([
            nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_f_len) for _ in range(n_conv)]) for nbr_f_len in nbr_fea_len
        ])
        self.conv_to_fc = nn.Linear(atom_fea_len*len(nbr_fea_len), h_fea_len)
        # self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)

        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
        atom_fea0 = self.embedding(atom_fea)
        crys_feas = []
        for idx, (nbr_f, nbr_f_idx) in enumerate(zip(nbr_fea, nbr_fea_idx)):
            atom_fea = atom_fea0
            for i, layer in enumerate(self.convs[idx]):
                atom_fea = layer(atom_fea, nbr_f, nbr_f_idx)
            crys_fea = self.pooling(atom_fea, crystal_atom_idx)
            crys_feas.append(self.conv_to_fc_softplus(crys_fea))

        crys_feas = torch.cat(crys_feas, dim=1)        
        # atom_feas.append(atom_fea)
        # crys_fea = torch.cat(atom_feas, dim=1)
        crys_fea = self.conv_to_fc(crys_feas)
        crys_fea = self.conv_to_fc_softplus(crys_fea)
        if self.classification:
            crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            out = self.logsoftmax(out)
        return crys_fea, out

    def pooling(self, atom_fea, crystal_atom_idx):
        assert sum([len(idx_map) for idx_map in crystal_atom_idx]) ==\
            atom_fea.data.shape[0]
        summed_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                      for idx_map in crystal_atom_idx]
        return torch.cat(summed_fea, dim=0)