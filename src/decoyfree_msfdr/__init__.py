import argparse
import glob
import json
import multiprocessing
import os

tolerance = 1e-8
max_iteration = 5000
show_plotting = False
plot_interval = 5
gaussian_model = False
model_samples = 2
ic2_comp = True
init_strategy = 'random'
mu_strategy = 'gaussian'
random_size = 2
parallel = True
inner_parallel = False
num_workers = 1
if init_strategy == 'random' and parallel and inner_parallel:
    show_plotting = False

if os.path.exists('on_server'):
    show_plotting = False

out_dir = 'results/'
constraints = 'no_constraint'


def str_bool(sbool):
    if sbool in ('True', 'true', 'TRUE'):
        return True
    elif sbool in ('False', 'false', 'FALSE'):
        return False
    else:
        raise ValueError('Invalid bool argument')


def add_common_args(parser):
    parser.add_argument('-f', '--input-file')
    parser.add_argument('-d', '--input-dir')
    parser.add_argument('-t', '--input-type', default='idXML',
                        help='format of input files, supported formats are csv,tsv,idXML')
    parser.add_argument('-m', '--eval-model', help='existing model to be evaluated')
    parser.add_argument('--threads', type=int, default=1, help='number of threads')

    parser.add_argument('-c', '--constraints', default=constraints, help='Choices of constraints to be used')
    parser.add_argument('-s', '--model_samples', type=int, default=model_samples,
                        help='Number of samples/top scores to be used in modeling')
    parser.add_argument('-r', '--random_size', type=int, default=random_size,
                        help='Number of random starts per skewness setting')  # num restarts
    # parser.add_argument('-q', '--part', type=int, default=-1)
    parser.add_argument('--tolerance', type=float, default=tolerance,
                        help='Threshold of the change of the point-wise log-likelihood\
                         for the EM algorithm to determine the convergence')
    parser.add_argument('--show_plotting', action='store_true', default=show_plotting,
                        help='Show plotting while fitting the model')
    parser.add_argument('--out_dir', default=out_dir, help='The place to save results')


def decoyfree_msfdr():
    """decoyfree-msfdr [-f file|-d dir] [-t csv|tsv|idXML] [--score-field] [] ...
    """
    parser = argparse.ArgumentParser(
        prog='decoyfree-msfdr',
        description='Estimate FDR in MS searching results using decoy-free approach',
        epilog='')

    add_common_args(parser)
    parser.add_argument('--score_field', default='EValue', help='The field stores the score of PSM')
    parser.add_argument('--log_scale', type=str_bool, default=True, help='Whether to model on the log scale of the data')
    parser.add_argument('--neg_score', type=str_bool, default=True,
                        help='Whether to take negative of the score. In our model, higher score means better. \
                        On log scale, this is done after taking log')
    parser.add_argument('--spec_ref_fields', default='#SpecFile,SpecID',
                        help='Comma separated fields to identify a spectrum uniquely')

    args, _ = parser.parse_known_args()

    if not args.input_file and not args.input_dir:
        print('no input file')
        return

    if args.input_file and args.input_dir:
        print('Error: input file and input dir both are provided, only one can be used')
        return -1

    from .msparser.ms import MS_Dataset

    df = collect_search_result(args)
    dataset = MS_Dataset(df, scorefield=args.score_field, logscale=args.log_scale, negscore=args.neg_score,
                         spec_ref_column=args.spec_ref_fields.split(','))

    from . import ms_launcher
    if not args.eval_model:
        print(dataset.mat)
        ms_launcher.run_dataset(dataset)
    else:
        model = json.load(open(args.eval_model))
        model_perf = ms_launcher.eval_model(dataset, model, args.out_dir)
        print(model_perf)


def decoyfree_xlmsfdr():
    """decoyfree-msfdr [-f|-d] [-t csv|tsv|idXML] [--threads] [--score-field] [] ...
    """
    parser = argparse.ArgumentParser(
        prog='decoyfree-xlmsfdr',
        description='Estimate FDR in XLMS searching results using decoy-free approach',
        epilog='')

    add_common_args(parser)
    parser.add_argument('--score_field', default='OpenPepXL:score', help='The field stores the score of PSM')
    parser.add_argument('--log_scale', type=str_bool, default=False,
                        help='Whether to model on the log scale of the data')
    parser.add_argument('--neg_score', type=str_bool, default=False,
                        help='Whether to take negative of the score. In our model, higher score means better. \
                        On log-scale, this is done after taking log')

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

    from .msparser.xlms import XLMS_Dataset
    from . import xlms_launcher

    df = collect_search_result(args)
    dataset = XLMS_Dataset(df, scorefield=args.score_field, logscale=args.log_scale, negscore=args.neg_score)

    if not args.eval_model:
        print(dataset.mat)
        xlms_launcher.run_dataset(dataset)
    else:
        model = json.load(open(args.eval_model))
        model_perf = xlms_launcher.eval_model(dataset, model, args.out_dir)
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

    import pandas as pd
    df = pd.concat(dfs)
    print(df.shape)
    return df


def collect_file(f, args):
    import pandas as pd
    from .msparser import idxml
    if args.input_type == 'idXML':
        return idxml.DataFrame(f)
    elif args.input_type == 'tsv':
        return pd.read_csv(f, delimiter='\t')
    elif args.input_type == 'csv':
        return pd.read_csv(f)
