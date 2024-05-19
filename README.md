# Decoy-Free FDR Estimation for Mass-Spectrometry

A package to estimate FDR in mass-spectrometry searching results using decoy-free approach

## Install

```commandline
pip install decoyfree-msfdr
```

## Usage

Regular MS/MS search FDR estimation

```commandline
decoyfree-msfdr [-h] [-f INPUT_FILE] [-d INPUT_DIR] [-t INPUT_TYPE] [-m EVAL_MODEL]
                [--score-field SCORE_FIELD] [--threads THREADS] [-c CONSTRAINTS] [-s MODEL_SAMPLES]
                [-r RANDOM_SIZE] [--tolerance TOLERANCE] [--show_plotting] [--out_dir OUT_DIR]
```

Cross-linked peptide MS/MS search FDR estimation

```commandline
decoyfree-xlmsfdr [-h] [-f INPUT_FILE] [-d INPUT_DIR] [-t INPUT_TYPE] [-m EVAL_MODEL]
                  [--score-field SCORE_FIELD] [--threads THREADS] [-c CONSTRAINTS] [-s MODEL_SAMPLES]
                  [-r RANDOM_SIZE] [--tolerance TOLERANCE] [--show_plotting] [--out_dir OUT_DIR]
```

```
options:
  -h, --help            show this help message and exit
  -f INPUT_FILE, --input-file INPUT_FILE
  -d INPUT_DIR, --input-dir INPUT_DIR
  -t INPUT_TYPE, --input-type INPUT_TYPE
                        format of input files, supported formats are csv,tsv,idXML
  -m EVAL_MODEL, --eval-model EVAL_MODEL
                        existing model to be evaluated
  --score-field SCORE_FIELD
                        the field name holding PSM scores
  --threads THREADS     number of threads
  -c CONSTRAINTS, --constraints CONSTRAINTS
                        Choices of constraints to be used
  -s MODEL_SAMPLES, --model_samples MODEL_SAMPLES
                        Number of samples/top scores to be used in modeling
  -r RANDOM_SIZE, --random_size RANDOM_SIZE
                        Number of random starts per skewness setting
  --tolerance TOLERANCE
                        Threshold of the change of the point-wise log-likelihood for the EM algorithm to determine the
                        convergence
  --show_plotting       Show plotting while fitting the model
  --out_dir OUT_DIR     The place to save results
```

### Options

| Option              | Argument                               | Default             | Description                                                                                                    |
|---------------------|----------------------------------------|---------------------|----------------------------------------------------------------------------------------------------------------|
| -f, --input-file    | Path                                   | N/A                 | Path to the search result file                                                                                 |
| -d, --input-dir     | Path                                   | N/A                 | Path to the directory holding search result files                                                              |
| -t, --input-type    | csv, tsv, idXML                        | idXML               | Search result format                                                                                           |
| -m, --eval-model    | Path                                   | N/A                 | Path to an existing model to be evaluated                                                                      |
| -c, --constraints   | no_constraint,<br/>unweighted_pdf_mode | unweighted_pdf_mode | Choices of constraints to be used                                                                              |
| -s                  | 1, 2                                   | 2                   | Number of samples/top scores to be used in modeling                                                            |
| --threads           | Integer                                | 1                   | Number of threads                                                                                              |
| -r, --random_starts | Integer                                | 2                   | Number of random starts per skewness setting                                                                   |
| --tolerance         | Float                                  | 1e-8                | Threshold of the change of the point-wise log-likelihood<br/>for the EM algorithm to determine the convergence |
| --show_plotting     | Bool                                   | False               | Show plotting while fitting the model                                                                          |
| --out_dir           | Path                                   | ./results           | The place to save results                                                                                      |

## Examples

### Regular MS/MS search with MSGF+ engine

Suppose the MS/MS search result with MSGF+ software is saved in .tsv format, data/sample.tsv.
You can run the FDR estimation algorithm with the following command,

```commandline
decoyfree-msfdr -f data/sample.tsv -t tsv --out_dir results/sample --threads 10
```

### Multiple search results

If you have multiple search results from MSGF+ saved in the 'data/sample/' directory, and you want to use them all
together to build a single model, do the following,

```commandline
decoyfree-msfdr -d data/sample/ -t tsv --out_dir results/sample --threads 10
```

**Note**: this will search the directory for all the files with .tsv extension. If you specified other formats, it will
search for the files with the corresponding extension.

### MS/MS search results with other engines

If you are using another search engine, please specify the following information,

| Option            | Default            | Description                                                                                                                 |
|-------------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------|
| --score_field     | EValue             | The score field used to model the data                                                                                      |
| --log_scale       | True               | Whether to model on the log scale of the data                                                                               |
| --neg_score       | True               | Whether to take negative of the score. In our model, higher score means better. On log-scale, this is done after taking log |
| --spec_ref_fields | "#SpecFile,SpecID" | Comma separated fields to identify a spectrum uniquely                                                                      |

### XL-MS/MS search with OpenPepXLLF engine

Suppose the MS/MS search result with MSGF+ software is saved in .idXML format, data/sample.idXML.
You can run the FDR estimation algorithm with the following command,

```commandline
decoyfree-xlmsfdr -f data/sample.idXML -t idXML --out_dir results/sample --threads 10
```

**Note**: Currently, idXML is the only format we support. Please let us know if you need to use another format in
[issues](https://github.com/shawn-peng/DecoyFree-MSFDR/issues). I'll add support to that.

### XL-MS/MS search results with other engines

If you are using another search engine, please specify the following information,

| Option        | Default           | Description                                                                        |
|---------------|-------------------|------------------------------------------------------------------------------------|
| --score_field | "OpenPepXL:score" | The score field used to model the data                                             |
| --log_scale   | False             | Whether to model on the log scale of the data                                      |
| --neg_score   | False             | Whether to take negative of the score, on log-scale, this is done after taking log |





