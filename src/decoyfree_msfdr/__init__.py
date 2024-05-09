import argparse
import glob
import json
import multiprocessing
import pandas as pd
import numpy as np

from .msparser import idxml
from .msparser.xlms import XLMS_Dataset
from .msparser.ms import MS_Dataset
from . import xlms_launcher
from . import ms_launcher


def add_common_args(parser):

    parser.add_argument('-f', '--input-file')
    parser.add_argument('-d', '--input-dir')
    parser.add_argument('-t', '--input-type', default='idXML', help='format of input files, supported formats are csv,tsv,idXML')
    parser.add_argument('-m', '--eval-model', help='existing model to be evaluated')
    parser.add_argument('--score-field', help='the field name holding PSM scores')
    parser.add_argument('--threads', type=int, default=1, help='number of threads for parsing files and run algorithm')


def decoyfree_msfdr():
    """decoyfree-msfdr [-f|-d] [-t csv|tsv|idXML] [--score-field] []
    """
    parser = argparse.ArgumentParser(
        prog='decoyfree-msfdr',
        description='Estimate FDR in MS searching results using decoy-free approach',
        epilog='')

    add_common_args(parser)

    args, _ = parser.parse_known_args()

    if not args.input_file and not args.input_dir:
        print('no input file')
        return

    if args.input_file and args.input_dir:
        print('Error: input file and input dir both are provided, only one can be used')
        return -1

    df = collect_search_result(args)
    dataset = MS_Dataset(df)

    if not args.eval_model:
        print(dataset.mat)
        ms_launcher.run_dataset(dataset)
    else:
        model = json.load(open(args.eval_model))
        model_perf = ms_launcher.eval_model(dataset, model)
        print(model_perf)


def decoyfree_xlmsfdr():
    """decoyfree-msfdr [-f|-d] [-t csv|tsv|idXML] [--threads] [--score-field] []
    """
    parser = argparse.ArgumentParser(
        prog='decoyfree-xlmsfdr',
        description='Estimate FDR in MS searching results using decoy-free approach',
        epilog='')

    add_common_args(parser)

    args, _ = parser.parse_known_args()

    if not args.input_file and not args.input_dir:
        print('no input file')
        return

    if args.input_file and args.input_dir:
        print('Error: input file and input dir both are provided, only one can be used')
        return -1

    if args.input_type != 'idXML':
        print('Error: input type only support idXML for XLMS search')
        return -1

    df = collect_search_result(args)
    dataset = XLMS_Dataset(df)

    if not args.eval_model:
        print(dataset.mat)
        xlms_launcher.run_dataset(dataset)
    else:
        model = json.load(open(args.eval_model))
        model_perf = xlms_launcher.eval_model(dataset, model)
        print(model_perf)


def collect_search_result(args):
    if args.input_file:
        files = [args.input_file]
    else:
        files = glob.glob(f'{args.input_dir}/*.{args.input_type}')

    print(files)
    if args.threads == 1:
        dfs = []
        for f in files:
            dfs.append(collect_file(f, args))
    else:
        with multiprocessing.Pool(args.threads) as pool:
            dfs = pool.starmap(collect_file, [(f, args) for f in files])

    df = pd.concat(dfs)
    print(df.shape)
    return df


def collect_file(f, args):
    if args.input_type == 'idXML':
        return idxml.DataFrame(f)
    elif args.input_type == 'tsv':
        return pd.read_csv(f, delimiter='\t')
    elif args.input_type == 'csv':
        return pd.read_csv(f)



