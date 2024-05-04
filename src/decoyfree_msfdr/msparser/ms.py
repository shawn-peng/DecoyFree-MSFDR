import json
import multiprocessing
import hdf5storage
import pandas as pd
import numpy as np
from collections import defaultdict


class MS_Dataset:
    def __init__(self, df, scorefield='EValue'):
        print(df.shape)
        self.mat = self.extract_smat(df).to_numpy().T
        print(self.mat.shape)

    @staticmethod
    def extract_smat(
            tab: pd.DataFrame,
            score_column='EValue',
            logscale=True,
            negscore=True,
            spec_ref_column=('#SpecFile', 'SpecID'),
            pep_column='Peptide',
            noisotope=True,
            nodup=True,
            keep_diff_xl_pos=False):
        spec_matches = defaultdict(list)
        spec_peptides = defaultdict(set)
        for i, row in tab.iterrows():
            # specid = row['spectrum_reference']
            if isinstance(spec_ref_column, (tuple, list)):
                specid = ' '.join(map(lambda c: row[c], spec_ref_column))
            else:
                specid = row[spec_ref_column]
            s = row[score_column]
            if not s:
                continue
            if noisotope and spec_matches[specid] and spec_matches[specid][-1] == s:
                continue
            if len(spec_matches[specid]) >= 2:
                continue
            # pep = (row['sequence'], row['sequence_beta'])
            pep = row[pep_column]
            if nodup and pep in spec_peptides[specid]:
                continue
            if not keep_diff_xl_pos:
                spec_peptides[specid].add(pep)
            s = row[score_column]
            if logscale:
                s = np.log(s)
            if negscore:
                s = -s
            spec_matches[specid].append(s)
        for l in spec_matches.values():
            if len(l) == 1:
                l.append(0)
                # l.append(np.nan)
        smat = pd.DataFrame(spec_matches).transpose().rename(columns={0: 's1', 1: 's2'})
        print(smat)
        return smat
