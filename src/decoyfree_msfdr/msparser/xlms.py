import json
import multiprocessing
import hdf5storage
import pandas as pd
import numpy as np
from collections import defaultdict


class XLMS_Dataset:
    def __init__(self, df, nodup=True, scorefield='OpenPepXL:score', logscale=False, negscore=False,
                 spec_ref_column='spectrum_reference',
                 noisotope=True):
        # def __init__(self, df):
        df = df.query('xl_type=="cross-link"')
        print(df.shape)
        self.mat = self.extract_smat(df, score_column=scorefield, logscale=logscale, negscore=negscore,
                                     spec_ref_column=spec_ref_column, noisotope=noisotope, nodup=nodup).to_numpy().T
        print(self.mat.shape)

    @staticmethod
    def extract_smat(
            tab: pd.DataFrame,
            score_column='OpenPepXL:score',
            logscale=False,
            negscore=False,
            spec_ref_column='spectrum_reference',
            rank_column='xl_rank',
            noisotope=False,
            nodup=True,
            keep_diff_xl_pos=False):
        spec_matches = defaultdict(list)
        spec_peptides = defaultdict(set)
        for i, row in tab.iterrows():
            specid = row[spec_ref_column]
            # rank = row['xl_rank']
            s = row[score_column]
            if not s:
                continue
            if noisotope and spec_matches[specid] and spec_matches[specid][-1] == s:
                continue
            if len(spec_matches[specid]) >= 2:
                continue
            pep = (row['sequence'], row['sequence_beta'])
            if nodup and pep in spec_peptides[specid]:
                continue
            if not keep_diff_xl_pos:
                spec_peptides[specid].add(pep)
            if logscale:
                s += 20
                s = np.log(s)
                if np.isnan(s):
                    raise ValueError('There are invalid values in PSM score. Log scale can not be applied')
            if negscore:
                s = -s
            spec_matches[specid].append(s)
        for l in spec_matches.values():
            if len(l) == 1:
                l.append(0)
                # l.append(np.nan)
        return pd.DataFrame(spec_matches).transpose().rename(columns={0: 's1', 1: 's2'})
