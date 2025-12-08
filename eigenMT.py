"""
This script allows eigenMT to be used to correct QTL summary statistics by doing multiple testing correction for the true number of tests per feature

authors: François Aguet, Kaur Alasoo, joed3, Roy Oelen

"""

##############LIBRARIES##############

from __future__ import print_function
import os
import sys
import fileinput
import argparse
import numpy as np
import pandas as pd
import scipy.linalg as splin
import gc
import gzip
from sklearn import covariance
from bgen_reader import read_bgen
import h5py
import tempfile
import json
# for making checksums
import hashlib
# for warnings
import warnings

##############FUNCTIONS##############

def create_hash_file(input_file, algorithm="sha256"):
    """
    Creates a hash of the specified file using the given algorithm and writes it
    to a new file with the same name but an extension matching the algorithm.

    Args:
        input_file (str): Path to the input file for which the hash should be created.
        algorithm (str): Hash algorithm to use (e.g., 'md5', 'sha1', 'sha256', 'sha512').

    Returns:
        int: Returns 0 on success, 1 on failure.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the specified algorithm is not supported by hashlib.
        IOError: If there is an error reading the input file or writing the output file.
    """
    try:
        # Validate algorithm
        if algorithm.lower() not in hashlib.algorithms_available:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        with open(input_file, "rb") as f:
            # Python 3.11+ shortcut if available
            if callable(getattr(hashlib, 'file_digest', None)):
                digest = hashlib.file_digest(f, algorithm.lower())
            else:
                digest = hashlib.new(algorithm.lower())
                while chunk := f.read(8192):
                    digest.update(chunk)

        # Output file path with .<algorithm> extension
        output_hash_loc = f"{input_file}.{algorithm.lower()}"

        with open(output_hash_loc, "w") as out_file:
            out_file.write(digest.hexdigest())

        return 0

    except Exception as e:
        print(f"Exception occurred upon {algorithm} file creation: {e}")
        return 1


def open_file(filename):
    """
    Open a file, which may be gzipped, and return a file object.
    
    This function checks if the file is gzipped by reading the first few bytes of the file header.
    If the file is gzipped, it opens the file using gzip; otherwise, it opens the file normally.

    Parameters:
    filename (str): Path to the file to be opened.

    Returns:
    file object: A file object for reading the file.
    """
    with open(filename, 'rb') as file_connection:
        file_header = file_connection.readline()
    if file_header.startswith(b"\x1f\x8b\x08"):
        opener = gzip.open(filename, 'rt')
    else:
        opener = open(filename)
    return opener

def make_genpos_dict(POS_fh, CHROM=None):
    """
    Read SNPs and their positions from a file and create a dictionary.

    If CHROM is supplied, only SNPs on that chromosome are returned. If CHROM is None,
    SNPs from all chromosomes are returned.

    Parameters:
    POS_fh (str or file-like object): File handle or path to the file containing SNP positions.
    CHROM (str, optional): Chromosome identifier to filter SNPs by chromosome. If None, keep all chromosomes.

    Returns:
    dict: Dictionary with SNP IDs as keys and their positions as float values.
    """

    # create dictionary of variant as keys, and chromosomal positions as values
    pos_dict = {}
    # open with the supplied filehandle
    with open_file(POS_fh) as POS:
        # move the cursor past the header
        POS.readline()  # skip header
        # read each line
        for line in POS:
            # split by whitespace, and remove the trailing newline
            line = line.rstrip().split()
            # if CHROM is not supplied, accept all chromosomes; otherwise only keep matches
            if CHROM is None or line[1] == CHROM:
                # add to the dictionary the variant as key, and the chromosomal position as the value
                pos_dict[line[0]] = float(line[2])
    return pos_dict

def make_phepos_dict(POS_fh, CHROM=None):
    """
    Read phenotypes (probes, genes, peaks) with their start and end positions from a file and create a dictionary.

    If CHROM is supplied, only phenotypes on that chromosome are returned. If CHROM is None, all phenotypes
    (from all chromosomes) are returned.

    Parameters:
    POS_fh (str or file-like object): File handle or path to the file containing phenotype positions.
    CHROM (str, optional): Chromosome identifier to filter phenotypes by chromosome. If None, keep all chromosomes.

    Returns:
    dict: Dictionary with phenotype IDs as keys and their start and end positions as numpy arrays (float64).
    """

    # create dictionary of phenotypes as keys, and a list of the start and stop as the values
    pos_dict = {}
    # open with supplied filehandle
    with open_file(POS_fh) as POS:
        # move the cursor past the header
        POS.readline()  # skip header
        # read each line
        for line in POS:
            # split by whitespace, removing the trailing newline
            line = line.rstrip().split()
            # if CHROM is not supplied, accept all chromosomes; otherwise only keep matches
            if CHROM is None or line[1] == CHROM:
                # grab the start and stop (columns 2 and 3)
                pos_array = np.array(line[2:4], dtype=np.float64)
                # add to the dictionary the phenotype as key, and the list of start and stop as values
                pos_dict[line[0]] = pos_array
    return pos_dict

def load_tensorqtl_output(tensorqtl_parquet, group_size_s=None):
    """
    Read tensorQTL output from a Parquet file and process it for eigenMT analysis.

    Parameters:
    tensorqtl_parquet (str): Path to the tensorQTL output Parquet file.
    group_size_s (pd.Series, optional): Series mapping gene IDs to group sizes for p-value adjustment.

    Returns:
    pd.DataFrame: Processed DataFrame with necessary columns for eigenMT analysis.
    """
    
    # read the parquest file
    df = pd.read_parquet(tensorqtl_parquet)
    # use the gene ID
    if 'gene_id' not in df:
        # check if the phenotype id column contains the genotype:phenotype combination
        if ':' in df['phenotype_id'].iloc[0]:
            # and then get the phenotype ID by taking the second element of each genotype:phenotype combination
            df['gene_id'] = df['phenotype_id'].apply(lambda x: x.rsplit(':',1)[1] if ':' in x else x)
        # rename the column phenotype_id to gene_id
        else:
            df.rename(columns={'phenotype_id':'gene_id'}, inplace=True)
    # eigenMT requires a 'p-value' column (see make_test_dict); first column must be variant, second gene/phenotype
    df = df[['variant_id', 'gene_id']+[i for i in df.columns if i not in ['variant_id', 'gene_id']]]
    # select p-value column
    if 'pval_nominal' in df.columns:
        df['p-value'] = df['pval_nominal'].copy()
    elif 'pval_gi' in df.columns:  # interaction model
        df['p-value'] = df['pval_gi'].copy()
    # if the 'genes' have been grouped, take the smallest p value for the entire group
    if group_size_s is not None:
        print('  * adjusting p-values by phenotype group size')
        df['p-value'] = np.minimum(df['p-value']*df['gene_id'].map(group_size_s), 1.0)
    return df

    
def make_gen_dict_matrixqtl(GEN_fh, pos_dict, sample_ids=None):
    """
    Read genotype matrix from MatrixEQTL and create a dictionary.
    
    Parameters:
    GEN_fh (str or file-like object): File handle or path to the file containing genotype data.
    pos_dict (dict): Dictionary with SNP IDs as keys and their positions as values.
    sample_ids (list, optional): List of sample IDs to filter the genotype data.

    Returns:
    dict: Dictionary with SNP positions as keys and genotypes as values.
    """
    
    # create dictionary of chromosomal positions as keys and the genotypes for that variant as a numpy array
    gen_dict = {}
    # read using the supplied file handle
    with open_file(GEN_fh) as GEN:
        # read the first line, which is the header
        header = GEN.readline().rstrip().split()
        # if a sample ID list was provided, get the indices of the samples that we are keeping
        if sample_ids is not None:
            ix = [header[1:].index(i) for i in sample_ids]
        # check each line, which is a variant
        for line in GEN: #Go through each line of the genotype matrix and add line to gen_dict
            # remove the newline at the end, and split by whitespace
            line = line.rstrip().split()
            # the variant is the first item in the line
            snp = pos_dict[line[0]]
            # the genotypes of the samples are the rest of the line
            genos = np.array(line[1:])
            # if we had a sample ID filter, use the indices we got before, to only keep the genotypes of those samples
            if sample_ids is not None:
                genos = genos[ix]
            # if we have genotype entires entered as NA, set these to be -1
            genos[genos == 'NA'] = -1  # no effect if already -1
            # convert to float values
            genos = np.float64(genos)
            # set the variants we had marked as -1, so the unknown ones, to be the mean of the variants that we do know the values of
            genos[genos == -1] = np.mean(genos[genos != -1])
            # finally put in the dictionary for each variant position as key, the genotypes for that variant
            gen_dict[snp] = genos
    return gen_dict  # pos->genotypes


def get_genotype_data_bgen(bgen_loc):
    """
    Read a BGEN file using bgen_reader and return (bim, fam, bed, bgen_object).

    The returned `bim` is a pandas.DataFrame with at least columns: ['snp','chrom','pos','allele_ids','nalleles','i']
    The returned `fam` is a pandas.DataFrame with sample IDs in column 'iid' (and placeholder PLINK columns).
    `bed` is returned as None (placeholder).
    `bgen` is the raw object returned by `read_bgen` for downstream per-variant access.
    """
    # read the bgen file using bgen_reader
    bgen = read_bgen(bgen_loc, verbose=False)
    # initialize bed-like
    bed = None

    # normalize samples -> pandas Index/Series
    samples = None
    # try to get samples
    try:
        samples_obj = bgen['samples']
        # if dask-like
        if hasattr(samples_obj, 'compute'):
            # if dask format, we need to call compute to get the actual values
            samples = samples_obj.compute()
        else:
            samples = samples_obj
    except Exception:
        # try other attribute names
        samples = None
    # if we were able to extract the samples
    if samples is None:
        # generate a fam-like with no samples
        fam = pd.DataFrame(columns=['fid', 'iid', 'father', 'mother', 'sex', 'phenotype'])
    else:
        # samples may be an Index, ndarray or Series
        sample_list = list(samples)
        # create a PLINK-like fam with placeholders
        fam = pd.DataFrame({
            'fid': [str(s).split(':')[0] for s in sample_list],
            'iid': [str(s) for s in sample_list],
            'father': [0]*len(sample_list),
            'mother': [0]*len(sample_list),
            'sex': [0]*len(sample_list),
            'phenotype': [-9]*len(sample_list)
        })

    # normalize variants
    try:
        # extract the variants
        vars_obj = bgen['variants']
        # again, if dask-like, we need to call compute to get the actual values
        if hasattr(vars_obj, 'compute'):
            vars_df = vars_obj.compute()
        else:
            vars_df = vars_obj
    except Exception:
        vars_df = None

    if vars_df is None:
        # if there is no info we could grab, we create an empty bim
        bim = pd.DataFrame(columns=['snp', 'chrom', 'pos', 'allele_ids', 'nalleles', 'i'])
        return bim, fam, bed, bgen

    # make sure the variant table is in dataframe format
    vars_df = pd.DataFrame(vars_df)

    # set the rsid as the snp column
    if 'rsid' in vars_df.columns:
        vars_df['snp'] = vars_df['rsid'].astype(str)
    # or the id if rsid is not present
    elif 'id' in vars_df.columns:
        vars_df['snp'] = vars_df['id'].astype(str)
    # or if neither, use the position and chromosome
    else:
        # compose a unique id
        chrom_col = vars_df.columns[0] if 'chrom' not in vars_df.columns else 'chrom'
        pos_col = 'pos' if 'pos' in vars_df.columns else (vars_df.columns[1] if vars_df.shape[1] > 1 else None)
        if pos_col is not None:
            vars_df['snp'] = vars_df.apply(lambda r: f"{r.get('chrom', '')}:{r.get('pos', '')}", axis=1).astype(str)
        # if even the position is not there, then just use the index
        else:
            vars_df['snp'] = vars_df.index.astype(str)

    # grab the columns we need
    # chromosome
    if 'chrom' not in vars_df.columns:
        # try other common names
        for cand in ['chromosome', 'contig']:
            if cand in vars_df.columns:
                vars_df['chrom'] = vars_df[cand]
                break
    # position
    if 'pos' not in vars_df.columns:
        for cand in ['position', 'bp', 'snp_position']:
            if cand in vars_df.columns:
                vars_df['pos'] = vars_df[cand]
                break

    # allele ids
    if 'allele_ids' not in vars_df.columns and 'alleles' in vars_df.columns:
        vars_df['allele_ids'] = vars_df['alleles']

    # number of alleles
    if 'nalleles' not in vars_df.columns:
        if 'allele_ids' in vars_df.columns:
            vars_df['nalleles'] = vars_df['allele_ids'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
        else:
            vars_df['nalleles'] = 2

    # add integer index mapping to genotype array positions
    vars_df = vars_df.reset_index(drop=True)
    vars_df['i'] = range(len(vars_df))

    # create bim with required columns
    bim = pd.DataFrame({
        'snp': vars_df['snp'].astype(str),
        'chrom': vars_df['chrom'].astype(str) if 'chrom' in vars_df.columns else ['']*len(vars_df),
        'pos': vars_df['pos'].astype(int) if 'pos' in vars_df.columns else [0]*len(vars_df),
        'allele_ids': vars_df['allele_ids'] if 'allele_ids' in vars_df.columns else ['']*len(vars_df),
        'nalleles': vars_df['nalleles'],
        'i': vars_df['i']
    })

    # set index name
    bim.index = bim['snp'].astype(str)
    bim.index.name = 'candidate'

    return bim, fam, bed, bgen


def bgen_to_genotypes(bim, fam, bgen, CHROM, minimumProbabilityStep=0.1, genpos_dict=None):
    """
    Convert a bgen object (as returned by read_bgen) into a dictionary mapping
    genomic position -> dosage numpy array (samples in fam order).

    This implementation is defensive about various bgen_reader return types and
    attempts to handle common probability layouts.
    """
    gen_dict = {}

    # normalize chromosome representation
    chrom_str = str(CHROM) if CHROM is not None else ''
    chrom_str = chrom_str.replace('chr', '')

    # ensure required columns in bim
    if 'i' not in bim.columns:
        return gen_dict

    snp_idxs = list(bim['i'].values)
    snp_names = list(bim['snp'].values)
    chroms = list(bim['chrom'].astype(str).values) if 'chrom' in bim.columns else [''] * len(bim)
    positions = list(bim['pos'].values) if 'pos' in bim.columns else [None] * len(bim)
    allele_ids_list = list(bim['allele_ids'].values) if 'allele_ids' in bim.columns else [None] * len(bim)

    # select SNPs on requested chromosome
    sel_indices = [i for i, c in enumerate(chroms) if str(c).replace('chr', '') == chrom_str]
    sel = [(snp_idxs[i], snp_names[i], positions[i]) for i in sel_indices]

    # optionally filter by genpos_dict (which maps variant ID -> position)
    if genpos_dict is not None:
        sel = [(idx, name, pos) for (idx, name, pos) in sel if name in genpos_dict]

    for sel_i, (snp_idx, snp_name, snp_pos) in enumerate(sel):
        try:
            allele_ids = allele_ids_list[sel_indices[sel_i]] if allele_ids_list is not None and len(allele_ids_list) > 0 else None
        except Exception:
            allele_ids = None
        chrom_val = chroms[sel_indices[sel_i]] if chroms is not None and len(chroms) > 0 else None
        # snp_name, snp_pos, allele_ids, chrom_val now available for building chr:pos:ref:alt keys
        try:
            geno = bgen['genotype'][snp_idx]
            if hasattr(geno, 'compute'):
                geno = geno.compute()
        except Exception:
            # couldn't access this variant; skip
            continue

        # geno expected to have keys/attributes: 'probs', 'ploidy', optionally 'phased'
        probs = None
        ploidy = None
        phased = False
        if isinstance(geno, dict):
            probs = geno.get('probs', None)
            ploidy = geno.get('ploidy', None)
            phased = bool(geno.get('phased', False))
        else:
            probs = getattr(geno, 'probs', None)
            ploidy = getattr(geno, 'ploidy', None)
            phased = bool(getattr(geno, 'phased', False))

        if probs is None:
            continue

        probs = np.asarray(probs)

        # skip non-diploid if ploidy info present
        if ploidy is not None:
            try:
                if np.any(np.array(ploidy) != 2):
                    continue
            except Exception:
                pass

        # compute dosage from probs
        try:
            if probs.ndim == 2 and probs.shape[1] >= 3:
                # assume columns [P0,P1,P2]
                p1 = probs[:, 1]
                p2 = probs[:, 2]
                dosage = p1 + 2.0 * p2
                max_prob = np.max(probs[:, :3], axis=1)
                na_mask = max_prob < ((1.0 / 3.0) + minimumProbabilityStep)
            else:
                # fallback: use available probs, mark missing if max prob is low
                max_prob = np.max(probs, axis=1)
                # cannot compute expected dosage reliably without at least 3 cols
                continue
        except Exception:
            continue

        dosage = np.asarray(dosage, dtype=float)
        dosage[na_mask] = np.nan

        if np.all(np.isnan(dosage)):
            # nothing to do
            continue

        mean_val = np.nanmean(dosage)
        if np.isnan(mean_val):
            continue
        dosage[np.isnan(dosage)] = mean_val

        key = snp_pos if snp_pos is not None else snp_name
        try:
            key_i = int(key)
        except Exception:
            key_i = key
        gen_dict[key_i] = dosage
        # also map by SNP name (rsid) when available
        try:
            gen_dict[str(snp_name)] = dosage
        except Exception:
            pass
        # also map by chrom:pos:ref:alt when allele ids and chrom/pos present
        try:
            if allele_ids is not None and snp_pos is not None and chrom_val is not None:
                parts = str(allele_ids).split(',')
                if len(parts) >= 2:
                    ref = parts[0]
                    alt = parts[1]
                    varid = f"{str(chrom_val).replace('chr','')}:{int(snp_pos)}:{ref}:{alt}"
                    gen_dict[varid] = dosage
        except Exception:
            pass
        # also map by SNP name (rsid) for easier matching when variant IDs use rsids
        try:
            gen_dict[str(snp_name)] = dosage
        except Exception:
            pass

    return gen_dict


def get_genotype_data_plink1(plink_prefix):
    """
    Read PLINK1 files (.bed/.bim/.fam) and return normalized `bim` and `fam` DataFrames.

    Parameters:
    plink_prefix (str): Path prefix to the PLINK files (without extension).

    Returns:
    tuple: (bim, fam) where `bim` is a DataFrame with columns ['snp','chrom','pos','allele_ids','nalleles','i']
           and `fam` is a DataFrame with columns ['fid','iid','father','mother','sex','phenotype']
    """
    import math

    bim_path = plink_prefix + '.bim'
    fam_path = plink_prefix + '.fam'

    # read bim
    try:
        bim_df = pd.read_csv(bim_path, sep='\t', header=None, dtype=str)
    except Exception as e:
        raise IOError('Could not read BIM file {}: {}'.format(bim_path, e))

    # BIM expected to have at least 6 columns: chrom, snp, cm, pos, a1, a2
    if bim_df.shape[1] < 6:
        raise IOError('BIM file {} does not have expected 6 columns'.format(bim_path))

    bim_df = bim_df.iloc[:, :6]
    bim_df.columns = ['chrom', 'snp', 'cm', 'pos', 'a1', 'a2']

    bim = pd.DataFrame({
        'snp': bim_df['snp'].astype(str),
        'chrom': bim_df['chrom'].astype(str),
        'pos': pd.to_numeric(bim_df['pos'], errors='coerce').fillna(0).astype(int),
        'allele_ids': (bim_df['a1'].astype(str) + ',' + bim_df['a2'].astype(str)),
        'nalleles': 2,
        'i': range(len(bim_df))
    })
    bim.index = bim['snp'].astype(str)
    bim.index.name = 'candidate'

    # read fam
    try:
        fam_df = pd.read_csv(fam_path, sep=r'\s+', header=None, dtype=str)
    except Exception as e:
        raise IOError('Could not read FAM file {}: {}'.format(fam_path, e))

    # FAM should have 6 columns
    if fam_df.shape[1] < 6:
        # pad with placeholders
        cols = fam_df.shape[1]
        for i in range(cols, 6):
            fam_df[i] = 0

    fam_df = fam_df.iloc[:, :6]
    fam_df.columns = ['fid', 'iid', 'father', 'mother', 'sex', 'phenotype']

    fam = fam_df.copy()

    return bim, fam


def plink_to_genotypes(bim, fam, bed_path, CHROM=None, sample_ids=None, genpos_dict=None):
    """
    Convert PLINK1 .bed/.bim/.fam into gen_dict mapping position->dosage array (samples in fam order).

    This implementation supports SNP-major .bed files (the default PLINK mode). It decodes two-bit
    codes per sample and maps them to dosages: 0->0, 1->1, 2->2, 3->missing (NaN). Missing values
    are imputed with the per-variant mean.

    Parameters:
    bim (pd.DataFrame): DataFrame produced by get_genotype_data_plink1
    fam (pd.DataFrame): DataFrame produced by get_genotype_data_plink1
    bed_path (str): Path to the .bed file
    CHROM (str): optional chromosome filter
    sample_ids (list): optional sample subset to keep (IID column values)
    genpos_dict (dict): optional mapping variant_id->position to further filter variants

    Returns:
    dict: mapping of variant position (int) or name -> numpy array of dosages
    """
    gen_dict = {}

    # normalize sample selection
    n_samples = len(fam)
    fam_iids = list(fam['iid'].astype(str))
    if sample_ids is not None:
        ix = [fam_iids.index(s) for s in sample_ids if s in fam_iids]
    else:
        ix = None

    # determine which SNPs to read (and their order) from bim
    chrom_filter = None
    if CHROM is not None:
        chrom_filter = str(CHROM).replace('chr', '')

    # prepare selected indices; include allele ids and chrom so we can build chr:pos:ref:alt keys
    selected_rows = []
    for idx, row in bim.iterrows():
        chrom = str(row['chrom']).replace('chr', '')
        snp_name = str(row['snp'])
        pos = row['pos']
        allele_ids = row['allele_ids'] if 'allele_ids' in row.index else None
        if chrom_filter is not None and chrom != chrom_filter:
            continue
        if genpos_dict is not None and snp_name not in genpos_dict:
            continue
        selected_rows.append((int(row['i']), snp_name, pos, chrom, allele_ids))

    # open bed file and verify header
    try:
        fh = open(bed_path, 'rb')
    except Exception as e:
        raise IOError('Could not open BED file {}: {}'.format(bed_path, e))

    header = fh.read(3)
    if len(header) < 3:
        fh.close()
        raise IOError('BED file {} is too short'.format(bed_path))

    # plink magic: first two bytes 0x6c 0x1b, third byte indicates mode (1 == SNP-major)
    if header[0] != 0x6c or header[1] != 0x1b:
        fh.close()
        raise IOError('Not a PLINK BED file: {}'.format(bed_path))

    mode = header[2]
    if mode != 1:
        fh.close()
        raise IOError('Only SNP-major BED files are supported (mode byte != 1). Found mode={}'.format(mode))

    # bytes per SNP
    bytes_per_snp = int(np.ceil(n_samples / 4.0))

    # iterate through SNPs in file order; PLINK stores in same order as BIM
    # We'll step through the bed file reading bytes_per_snp for each SNP,
    # and only keep those SNPs that are in selected_rows (by index i)
    # Build a mapping from SNP index -> (name,pos) for quick lookup
    sel_map = {i: (name, pos, chrom, allele_ids) for (i, name, pos, chrom, allele_ids) in selected_rows}

    # iterate SNPs
    snp_idx = 0
    # read sequentially
    try:
        while snp_idx < len(bim):
            chunk = fh.read(bytes_per_snp)
            if not chunk:
                break
            if snp_idx in sel_map:
                name, pos, chrom_val, allele_ids = sel_map[snp_idx]
                # decode chunk into per-sample two-bit codes
                codes = np.empty(n_samples, dtype=np.uint8)
                codes.fill(3)  # default to missing
                out_i = 0
                for byte in chunk:
                    for k in range(4):
                        if out_i >= n_samples:
                            break
                        code = (byte >> (2 * k)) & 0x3
                        codes[out_i] = code
                        out_i += 1

                # map codes to dosage: 0->0,1->1,2->2,3->nan
                dosage = np.empty(n_samples, dtype=float)
                dosage[:] = np.nan
                # mapping: code 0 => 0, code 1 => 1, code 2 => 2
                mask0 = codes == 0
                mask1 = codes == 1
                mask2 = codes == 2
                dosage[mask0] = 0.0
                dosage[mask1] = 1.0
                dosage[mask2] = 2.0

                # subset samples if requested
                if ix is not None:
                    dosage = dosage[ix]

                # impute missing with mean
                if np.all(np.isnan(dosage)):
                    # skip variants with all missing
                    pass
                else:
                    mean_val = np.nanmean(dosage)
                    dosage[np.isnan(dosage)] = mean_val
                    key = pos if (pos is not None and not pd.isna(pos) and pos != 0) else name
                    try:
                        key_i = int(key)
                    except Exception:
                        key_i = key
                    # store dosage under integer position (if available) and under SNP name for rsid matching
                    gen_dict[key_i] = dosage
                    try:
                        gen_dict[str(name)] = dosage
                    except Exception:
                        pass
                    # also map by chrom:pos:ref:alt when possible
                    try:
                        if allele_ids is not None and pos is not None and chrom_val is not None:
                            parts = str(allele_ids).split(',')
                            if len(parts) >= 2:
                                ref = parts[0]
                                alt = parts[1]
                                varid = f"{str(chrom_val).replace('chr','')}:{int(pos)}:{ref}:{alt}"
                                gen_dict[varid] = dosage
                    except Exception:
                        pass

            snp_idx += 1
    finally:
        fh.close()

    return gen_dict

def make_test_dict_matrixqtl(QTL_fh, gen_dict, genpos_dict, phepos_dict, cis_dist=None, pvalue_column=None, genchrom_dict=None, CHROM=None):
    """
    Create a dictionary of SNP-gene tests from a QTL file and return the file header.
    
    Parameters:
    QTL_fh (str or file-like object): File handle or path to the QTL file.
    gen_dict (dict): Dictionary with SNP positions as keys and genotypes as values.
    genpos_dict (dict): Dictionary with SNP IDs as keys and their positions as values.
    phepos_dict (dict): Dictionary with phenotype IDs as keys and their start and end positions as values.
    cis_dist (float): Maximum distance for SNPs to be considered in cis with the phenotype. If none is applied, all SNP-gene pairs in the summary stats will be considered.
    pvalue_column (str, optional): Name of the column containing p-values. If not provided, the function will search for common p-value column names.

    Returns:
    tuple: A dictionary with gene IDs as keys and a dictionary of test results as values, and the header of the QTL file.
    """
    
    # read the QTL filehandle that was supplied
    QTL = open_file(QTL_fh)
    # read the header of the file
    header = QTL.readline().rstrip().split()
    # check if the p value column was supplied
    if pvalue_column is not None:
        # and if it is actually present in the file
        if  pvalue_column in header:
            # get the index of that column
            pvalIndex = header.index(pvalue_column)
        # if not, then exit
        else:
            sys.exit(''.join(['Cannot find supplied p-value column in the tests file:', pvalue_column]))
    # find the column with the p-value based on some possibilities
    elif 'p-value' in header:
        # get the index of that column
        pvalIndex = header.index('p-value')
    elif 'p.value' in header:
        # get the index of that column
        pvalIndex = header.index('p.value')
    elif 'pvalue' in header:
        # get the index of that column
        pvalIndex = header.index('pvalue')
    else:
        sys.exit('Cannot find the p-value column in the tests file.')

    # create a dictionary that has each feature/phenotype/gene as a key, as as the value another dictionary with:
    # all the variants for this phenotype, 
    # the variant with the lowest p-value, 
    # the p-value of the variant with the lowest p-value, 
    # and the line of the QTL output of the variant wih the lowest p-value
    test_dict = {}

    # check each line in the QTL output
    for line in QTL:
        # remove trailing newline, and split by whitespace
        line = line.rstrip().split()
        # check if the variant, in the first column, is one for which we have the genotype position
        if line[0] in genpos_dict:
            # optional chromosome filter (requires genchrom_dict)
            if CHROM is not None and genchrom_dict is not None:
                vchrom = genchrom_dict.get(line[0])
                if vchrom is None or str(vchrom).replace('chr','') != str(CHROM).replace('chr',''):
                    continue
            # extract the position of the variant
            snp = genpos_dict[line[0]]
            # the phenotype is the second column in the file
            gene = line[1]
            # check if we have the variant in the genotype dict, based on the position and the variant based on their position
            if snp in gen_dict and gene in phepos_dict:
                # extract the chromosomal positon of the phenotype
                phepos = phepos_dict[gene]
                # check the absolute distance of the variant to the flanks of the phenotype, and take the closest, so the smallest value
                distance = min(abs(phepos - snp))
                # check if we are filtering by distance, and if this distance is within the cis window
                if cis_dist is None or distance <= cis_dist:
                    # if it is within the cis window, extract the p-value for this variant-feature
                    pval = line[pvalIndex]
                    # convert to a float
                    pval = float(pval)
                    # check if this is the first time we encounter this gne
                    if gene not in test_dict:
                        # if so, add it to the dictionary
                        test_dict[gene] = {'snps' : [snp], 'best_snp' : snp, 'pval' : pval, 'line' : '\t'.join(line)}
                    else:
                        # if not, then check if the variant is more significant that the current best hit
                        if pval < test_dict[gene]['pval']:
                            # if so, update the parameters describing the best hit
                            test_dict[gene]['best_snp'] = snp
                            test_dict[gene]['pval'] = pval
                            test_dict[gene]['line'] = '\t'.join(line)
                        # and add the location of this variant to the list of variants tested for this feature
                        test_dict[gene]['snps'].append(snp)

    # close the filehandle
    QTL.close()
    # return the dictionary and the header of the file
    return test_dict, "\t".join(header)

def make_test_dict_tensorqtl(QTL_fh, genpos_dict, cis_dist=None, group_size_s=None, genchrom_dict=None, CHROM=None):
    """
    Create a dictionary of SNP-gene tests from a tensorQTL file and return the file header.
    
    Parameters:
    QTL_fh (str or file-like object): Parquet file with variant-gene pair associations.
    genpos_dict (dict): Dictionary with SNP IDs as keys and their positions as values.
    phepos_dict (dict): Dictionary with phenotype IDs as keys and their start and end positions as values.
    cis_dist (float): Maximum distance for SNPs to be considered in cis with the phenotype. If none is applied, all SNP-gene pairs in the summary stats will be considered.
    pvalue_column (str, optional): Name of the column containing p-values. If not provided, the function will search for common p-value column names.

    Returns:
    tuple: A dictionary with gene IDs as keys and a dictionary of test results as values, and the header of the QTL file.
    """
    
    # load the tensorQTL output
    qtl_df = load_tensorqtl_output(QTL_fh, group_size_s=group_size_s)
    # chromosome filter if requested
    if CHROM is not None:
        chrom_str = str(CHROM).replace('chr','')
        if 'chrom' in qtl_df.columns:
            qtl_df = qtl_df[qtl_df['chrom'].astype(str).str.replace('chr','') == chrom_str]
        else:
            # try to parse chromosome from variant_id (prefix before ':')
            qtl_df = qtl_df[qtl_df['variant_id'].astype(str).apply(lambda x: str(x).split(':')[0].replace('chr','') == chrom_str)]
    # filter so that the variant-feature pairs are within the given cis distance
    if cis_dist is not None:
        qtl_df = qtl_df[qtl_df['tss_distance'].abs()<=cis_dist]
    # group the results for each feature
    gdf = qtl_df.groupby('gene_id')
    
    # create a dictionary that has each feature/phenotype/gene as a key, as as the value another dictionary with:
    # all the variants for this phenotype, 
    # the variant with the lowest p-value, 
    # the p-value of the variant with the lowest p-value, 
    # and the line of the QTL output of the variant wih the lowest p-value
    test_dict = {}
    # check each phenotype in each group
    for gene_id,g in gdf:
        # grab the smalles p-value position, and then as a dictionary the values it has based on the header
        g0 = g.loc[g['p-value'].idxmin()]
        # create the entry for this phenotype
        test_dict[gene_id] = {
            # getting all the variants
            'snps': [genpos_dict[i] for i in g['variant_id']],  # variant positions
            # take the variant at the position of the lowest P-value that we determined before
            'best_snp':genpos_dict[g0['variant_id']],
            # add that p-value
            'pval':g0['p-value'],
            # add the line of that lowest p-value
            'line':'\t'.join([i if isinstance(i, str) else '{:.6g}'.format(i)  for i in g0.values])
        }
    # return the dictionary and the header of the file
    return test_dict, '\t'.join(qtl_df.columns)

def make_test_dict_external(QTL_fh, gen_dict, genpos_dict, phepos_dict, cis_dist=None, pvalue_column=None, variant_col='variant_id', feature_col='feature_id', genchrom_dict=None, CHROM=None):
    """
    Create a dictionary of SNP-gene tests from a QTL file, assuming the genotype matrix and position file
    are separate from those used in the Matrix-eQTL run. This function is used with the external option
    to allow calculation of the effective number of tests using a different, preferably larger, genotype sample.
    
    Parameters:
    QTL_fh (str or file-like object): File handle or path to the QTL file.
    gen_dict (dict): Dictionary with SNP positions as keys and genotypes as values.
    genpos_dict (dict): Dictionary with SNP IDs as keys and their positions as values.
    phepos_dict (dict): Dictionary with phenotype IDs as keys and their start and end positions as values.
    cis_dist (float): Maximum distance for SNPs to be considered in cis with the phenotype. If none is applied, all snp-gene pairs in the summary stats will be considered.
    pvalue_column (str, optional): Name of the column containing p-values. If not provided, the function will search for common p-value column names.
    variant_index_col (float, optional): Index of the column that contains the variant identifier (first column is the default). 
    feature_index_col (float, optional): Index of the column that contains the feature identifier (second column is the default). 

    Returns:
    tuple: A dictionary with gene IDs as keys and a dictionary of test results as values, and the header of the QTL file.
    """
    
    # read the QTL file
    QTL = open_file(QTL_fh)
    # split the header based on whitespace
    header = QTL.readline().rstrip().split()
    # resolve variant and feature columns (accept either int index or string column name)
    def _resolve_col(col, header, default_name):
        if isinstance(col, int):
            return col
        if isinstance(col, str):
            if col in header:
                return header.index(col)
            else:
                sys.exit(''.join(["Cannot find supplied column in the tests file: ", col]))
        # fallback
        return header.index(default_name) if default_name in header else 0

    variant_index_col = _resolve_col(variant_col, header, 'variant_id')
    feature_index_col = _resolve_col(feature_col, header, 'feature_id')
    # check if the p value column was supplied
    if pvalue_column is not None:
        # and if it is actually present in the file
        if  pvalue_column in header:
            # get the index of that column
            pvalIndex = header.index(pvalue_column)
        # if not, then exit
        else:
            sys.exit(''.join(['Cannot find supplied p-value column in the tests file:', pvalue_column]))
    # find the column with the p-value based on some possibilities
    elif 'p-value' in header:
        # get the index of that column
        pvalIndex = header.index('p-value')
    elif 'p.value' in header:
        # get the index of that column
        pvalIndex = header.index('p.value')
    elif 'pvalue' in header:
        # get the index of that column
        pvalIndex = header.index('pvalue')
    else:
        sys.exit('Cannot find the p-value column in the tests file.')
    
    # create a dictionary that has each feature/phenotype/gene as a key, as as the value another dictionary with:
    # all the variants for this phenotype, 
    # the variant with the lowest p-value, 
    # the p-value of the variant with the lowest p-value, 
    # and the line of the QTL output of the variant wih the lowest p-value
    test_dict = {}

    # read each line in the QTL file
    for line in QTL:
        # split based on whitespace after removing the newline
        line = line.rstrip().split()
        # check if the first column, the variant, is in the dictionary of genomic positions
        if line[variant_index_col] in genpos_dict:
            # chromosome filter if requested
            if CHROM is not None and genchrom_dict is not None:
                vchrom = genchrom_dict.get(line[variant_index_col])
                if vchrom is None or str(vchrom).replace('chr','') != str(CHROM).replace('chr',''):
                    continue
            # get the position of the variant
            snp = genpos_dict[line[variant_index_col]]
            # get the phenotype from the line, in the second column
            gene = line[feature_index_col]
            # check if we have genotype data for this variant, and the position of this phenotype/gene/feature
            if snp in gen_dict and gene in phepos_dict:
                # get the position of the feature
                phepos = phepos_dict[gene]
                # check the distance of the variant to the flanks of the feature, and take the smallest value
                distance = min(abs(phepos - snp))
                # check if we are subsetting based on cis distance, and if we are in cis distance
                if cis_dist is not None and distance <= cis_dist:
                    # get the p-value for this variant and feature assocation
                    pval = line[pvalIndex]
                    # convert to float
                    pval = float(pval)
                    # check if this was the first time we handled this feature
                    if gene not in test_dict:
                        # if so, add the entry
                        test_dict[gene] = {'best_snp' : snp, 'pval' : pval, 'line' : '\t'.join(line)}
                    else:
                        # otherwise, check if this variant was more significant than the current most significant p-value for this feature
                        if pval < test_dict[gene]['pval']:
                            # if so, update for the best variant info to have this variants information
                            test_dict[gene]['best_snp'] = snp
                            test_dict[gene]['pval'] = pval
                            test_dict[gene]['line'] = '\t'.join(line)

    # close filehandle
    QTL.close()
    # get the position of each variant in a numpy array
    snps = np.array(genpos_dict.values())
    # check each phenotype again
    for gene in test_dict:
        # get the genomic position of the phenotype
        phepos = phepos_dict[gene]
        # Calculate distances to phenotype start and end positions
        is_in_cis_start = abs(snps - phepos[0]) <= cis_dist
        is_in_cis_end = abs(snps - phepos[1]) <= cis_dist
        # get all of the variants that were in the cis window of this feature
        test_dict[gene]['snps'] = snps[is_in_cis_start | is_in_cis_end]
    return test_dict, "\t".join(header)


def get_variants_from_qtl_file(QTL_fh, variant_col='variant_id', genchrom_dict=None, CHROM=None):
    """
    Extract unique variant identifiers from a QTL summary file.

    Supports plain-text QTL files (whitespace-separated) and Parquet files (tensorQTL output).

    Parameters:
    QTL_fh (str): Path to the QTL file (.txt/.gz/.parquet).
    variant_index_col (int): Column index (0-based) for the variant identifier in text files.

    Returns:
    list: Sorted list of unique variant identifiers (strings).
    """
    # Parquet (tensorQTL) case
    if isinstance(QTL_fh, str) and QTL_fh.endswith('.parquet'):
        try:
            df = pd.read_parquet(QTL_fh)
            # if variant_col is a string, try to use that column name
            if isinstance(variant_col, str) and variant_col in df.columns:
                vals = pd.unique(df[variant_col]).tolist()
                # apply chromosome/genchrom filtering if requested
                if CHROM is not None:
                    if genchrom_dict is not None:
                        vals = [v for v in vals if v in genchrom_dict and str(genchrom_dict[v]).replace('chr', '') == str(CHROM).replace('chr', '')]
                    else:
                        # try to find a chrom column for variants in dataframe
                        chrom_cols = [c for c in df.columns if 'chrom' in c.lower() or 'chr' in c.lower()]
                        if chrom_cols:
                            chrom_col = chrom_cols[0]
                            # map variant->chrom using dataframe
                            variant_to_chrom = dict(zip(df[variant_col].astype(str), df[chrom_col].astype(str)))
                            vals = [v for v in vals if v in variant_to_chrom and str(variant_to_chrom[v]).replace('chr', '') == str(CHROM).replace('chr', '')]
                return sorted(vals)
            # if variant_col is an int, use positional column
            if isinstance(variant_col, int) and variant_col < len(df.columns):
                colname = df.columns[variant_col]
                vals = pd.unique(df[colname]).tolist()
                if CHROM is not None:
                    if genchrom_dict is not None:
                        vals = [v for v in vals if v in genchrom_dict and str(genchrom_dict[v]).replace('chr', '') == str(CHROM).replace('chr', '')]
                return sorted(vals)
            # fallback: try common column name
            if 'variant_id' in df.columns:
                return sorted(pd.unique(df['variant_id']).tolist())
            return []
        except Exception:
            return []

    # Plain text (possibly gzipped)
    variants = set()
    fh = open_file(QTL_fh)
    header = fh.readline().rstrip().split()

    # resolve variant column index for text file
    if isinstance(variant_col, int):
        variant_index = variant_col
    else:
        if variant_col in header:
            variant_index = header.index(variant_col)
        else:
            # fallback to first column
            variant_index = 0

    # try to find a chromosome column in the header for text QTL files
    chrom_index = None
    for colname in ('snp_chromosome', 'snp_chrom', 'variant_chromosome', 'variant_chrom', 'chrom'):
        if colname in header:
            chrom_index = header.index(colname)
            break

    for line in fh:
        parts = line.rstrip().split()
        if len(parts) <= variant_index:
            continue
        v = parts[variant_index]
        # apply chrom/genchrom filtering if requested
        if CHROM is not None:
            ok = True
            if genchrom_dict is not None:
                if v not in genchrom_dict or str(genchrom_dict[v]).replace('chr', '') != str(CHROM).replace('chr', ''):
                    ok = False
            elif chrom_index is not None and len(parts) > chrom_index:
                if str(parts[chrom_index]).replace('chr', '') != str(CHROM).replace('chr', ''):
                    ok = False
            if not ok:
                continue
        variants.add(v)
    fh.close()
    return sorted(list(variants))


def get_variants_from_limix_h5(QTL_h5_path, genchrom_dict=None, CHROM=None):
    """
    Extract unique variant identifiers from a LIMIX H5 chunked QTL output file.

    Parameters:
    QTL_h5_path (str): Path to the LIMIX H5 file.

    Returns:
    list: Sorted list of unique variant identifiers (strings).
    """
    vs = set()
    try:
        h5fh = h5py.File(QTL_h5_path, 'r')
        for feature in h5fh.keys():
            if 'snp_id' in h5fh[feature].keys():
                arr = h5fh[feature]['snp_id']
                # if CHROM requested, check if per-feature snp_chromosome dataset exists
                have_snp_chrom = 'snp_chromosome' in h5fh[feature]
                for i, v in enumerate(arr):
                    # decode bytes to str if necessary
                    try:
                        vid = v.decode('utf-8')
                    except Exception:
                        vid = str(v)
                    if CHROM is not None:
                        ok = True
                        if have_snp_chrom:
                            try:
                                sc = h5fh[feature]['snp_chromosome'][i]
                                if isinstance(sc, (bytes, bytearray)):
                                    sc = sc.decode('utf-8')
                                if str(sc).replace('chr', '') != str(CHROM).replace('chr', ''):
                                    ok = False
                            except Exception:
                                pass
                        elif genchrom_dict is not None:
                            if vid not in genchrom_dict or str(genchrom_dict[vid]).replace('chr', '') != str(CHROM).replace('chr', ''):
                                ok = False
                        if not ok:
                            continue
                    vs.add(vid)
        h5fh.close()
    except Exception:
        # On error, return empty list
        return []
    return sorted(vs)


def get_phepos_from_limix_file(QTL_fh, CHROM=None):
    """
    Extract phenotype positions from a LIMIX text QTL file.

    Parameters:
    QTL_fh (str): Path to the LIMIX text output file (possibly gzipped).
    CHROM (str): optional chromosome filter for phenotypes (matches feature_chromosome column if present).

    Returns:
    dict: mapping feature_id -> numpy.array([start, end], dtype=float)
    """
    phepos = {}
    fh = open_file(QTL_fh)
    header = fh.readline().rstrip().split()
    # required columns
    if 'feature_id' not in header or ('feature_start' not in header and 'feature_start' not in header):
        # try common alternatives
        if 'phenotype_id' in header and 'feature_start' in header:
            pass
    # find indexes
    try:
        feature_idx = header.index('feature_id')
    except ValueError:
        # try phenotype_id
        feature_idx = header.index('phenotype_id') if 'phenotype_id' in header else None
    # start/end
    start_idx = header.index('feature_start') if 'feature_start' in header else (header.index('start') if 'start' in header else None)
    end_idx = header.index('feature_end') if 'feature_end' in header else (header.index('end') if 'end' in header else None)
    chrom_idx = header.index('feature_chromosome') if 'feature_chromosome' in header else (header.index('chrom') if 'chrom' in header else None)

    if feature_idx is None or start_idx is None or end_idx is None:
        fh.close()
        return {}

    for line in fh:
        parts = line.rstrip().split()
        if len(parts) <= max(feature_idx, start_idx, end_idx):
            continue
        feat = parts[feature_idx]
        try:
            s = float(parts[start_idx])
            e = float(parts[end_idx])
        except Exception:
            continue
        if CHROM is not None and chrom_idx is not None:
            if str(parts[chrom_idx]).replace('chr', '') != str(CHROM).replace('chr', ''):
                continue
        phepos[feat] = np.array([s, e], dtype=np.float64)
    fh.close()
    return phepos


def get_phepos_from_limix_h5(QTL_h5_path, CHROM=None):
    """
    Extract phenotype positions from a LIMIX H5 chunked QTL output file.

    Parameters:
    QTL_h5_path (str): Path to the LIMIX H5 file.
    CHROM (str): optional chromosome filter (matches 'feature_chromosome' dataset or attribute if present).

    Returns:
    dict: mapping feature_id -> numpy.array([start, end], dtype=float)
    """
    phepos = {}
    try:
        h5fh = h5py.File(QTL_h5_path, 'r')
    except Exception:
        return {}

    for feature in h5fh.keys():
        grp = h5fh[feature]
        # attempt datasets first
        s = None
        e = None
        chrom = None
        if 'feature_start' in grp:
            s = grp['feature_start'][()]
        if 'feature_end' in grp:
            e = grp['feature_end'][()]
        if 'feature_chromosome' in grp:
            chrom = grp['feature_chromosome'][()]
        # try attributes
        if s is None and 'feature_start' in grp.attrs:
            s = grp.attrs.get('feature_start')
        if e is None and 'feature_end' in grp.attrs:
            e = grp.attrs.get('feature_end')
        if chrom is None and 'feature_chromosome' in grp.attrs:
            chrom = grp.attrs.get('feature_chromosome')

        # decode bytes if needed
        try:
            if isinstance(s, (bytes, bytearray)):
                s = float(s.decode('utf-8'))
            if isinstance(e, (bytes, bytearray)):
                e = float(e.decode('utf-8'))
            if isinstance(chrom, (bytes, bytearray)):
                chrom = chrom.decode('utf-8')
        except Exception:
            pass

        if s is None or e is None:
            continue
        if CHROM is not None and chrom is not None:
            if str(chrom).replace('chr', '') != str(CHROM).replace('chr', ''):
                continue
        phepos[feature] = np.array([float(s), float(e)], dtype=np.float64)

    h5fh.close()
    return phepos

def make_test_dict_limix(QTL_fh, cis_dist=None, genchrom_dict=None, CHROM=None):
    """
    Processes QTL data to create dictionaries of genomic and phenotypic positions, and a dictionary of test results.

    Parameters:
    -----------
    QTL_fh : str
        File handle for the QTL data file.
    cis_dist : int, optional
        Maximum distance for cis-acting variants (default is None).

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - genpos_dict : dict
            Dictionary with variant identifiers as keys and their genomic positions as values.
        - phepos_dict : dict
            Dictionary with feature identifiers as keys and their start and end positions as values.
        - test_dict : dict
            Dictionary with feature identifiers as keys and dictionaries of test results as values.
        - header : str
            The header line of the QTL file.

    Notes:
    ------
    - The function reads the QTL file and identifies the p-value column.
    - It creates dictionaries for genomic positions, phenotypic positions, and test results.
    - Test results include all variants for each feature, the variant with the lowest p-value, and the corresponding line from the QTL file.
    - Only variants within the specified cis distance are considered.
    """
    
    # for the limix output, the feature column is actually the first column
    #feature_index_col = 0
    # and the variant is the second one
    #variant_index_col = 1
    # and the p-value columns is this
    pvalue_column = 'p_value'
    # read the QTL filehandle that was supplied
    QTL = open_file(QTL_fh)
    # read the header of the file
    header = QTL.readline().rstrip().split()
    # check if the p value column was supplied
    if pvalue_column is not None:
        # and if it is actually present in the file
        if  pvalue_column in header:
            # get the index of that column
            pvalIndex = header.index(pvalue_column)
        # if not, then exit
        else:
            sys.exit(''.join(['Cannot find supplied p-value column in the tests file:', pvalue_column]))
    # find the column with the p-value based on some possibilities
    elif 'p-value' in header:
        # get the index of that column
        pvalIndex = header.index('p-value')
    elif 'p.value' in header:
        # get the index of that column
        pvalIndex = header.index('p.value')
    elif 'pvalue' in header:
        # get the index of that column
        pvalIndex = header.index('pvalue')
    elif 'p_value' in header:
        # get the index of that column
        pvalIndex = header.index('p_value')
    else:
        sys.exit('Cannot find the p-value column in the tests file.')

    # create a dictionary that has each feature/phenotype/gene as a key, as as the value another dictionary with:
    # all the variants for this phenotype, 
    # the variant with the lowest p-value, 
    # the p-value of the variant with the lowest p-value, 
    # and the line of the QTL output of the variant wih the lowest p-value
    test_dict = {}
    # we also need two other dictionaries
    genpos_dict = {}
    phepos_dict = {}

    # get indices of each column (if present)
    feature_id_index = header.index('feature_id') if 'feature_id' in header else None
    snp_id_index = header.index('snp_id') if 'snp_id' in header else None
    feature_start_index = header.index('feature_start') if 'feature_start' in header else None
    feature_end_index = header.index('feature_end') if 'feature_end' in header else None
    snp_position_index = header.index('snp_position') if 'snp_position' in header else None
    snp_chromosome_index = header.index('snp_chromosome') if 'snp_chromosome' in header else None
    
    # check each line in the QTL output
    for line in QTL:
        # line looks like this: 
        # feature_id,snp_id,p_value,beta,beta_se,empirical_feature_p_value,feature_chromosome,feature_start,feature_end,ENSG,biotype,n_samples,n_e_samples,snp_chromosome,snp_position,assessed_allele,call_rate,maf,hwe_p
        # remove trailing newline, and split by whitespace
        line = line.rstrip().split()
        # grab the values (defensive: some columns may be missing)
        try:
            variant = line[snp_id_index] if snp_id_index is not None else None
            feature = line[feature_id_index] if feature_id_index is not None else None
            p_value = line[pvalIndex]
        except Exception:
            continue
        # parse feature start/end and variant position as numeric values; skip malformed lines
        try:
            feature_pos = np.array([float(line[feature_start_index]), float(line[feature_end_index])], dtype=np.float64) if feature_start_index is not None and feature_end_index is not None else None
        except Exception:
            continue
        try:
            var_pos = float(line[snp_position_index]) if snp_position_index is not None else None
        except Exception:
            continue
        # coerce variant position to an int when appropriate and add positions
        if var_pos is not None:
            try:
                # if var_pos is an integer value, store as int for consistency with PLINK keys
                if float(var_pos).is_integer():
                    var_pos_key = int(float(var_pos))
                else:
                    var_pos_key = float(var_pos)
            except Exception:
                var_pos_key = var_pos
            if variant is not None:
                genpos_dict[variant] = var_pos_key
        if feature is not None and feature_pos is not None:
            phepos_dict[feature] = feature_pos
        # apply CHROM filtering if requested; try snp_chromosome field first, then genchrom_dict if available
        if CHROM is not None:
            keep = True
            if snp_chromosome_index is not None and len(line) > snp_chromosome_index:
                try:
                    sc = str(line[snp_chromosome_index]).replace('chr', '')
                    if sc != str(CHROM).replace('chr', ''):
                        keep = False
                except Exception:
                    pass
            elif genchrom_dict is not None and variant is not None:
                if variant in genchrom_dict:
                    try:
                        sc = str(genchrom_dict[variant]).replace('chr', '')
                        if sc != str(CHROM).replace('chr', ''):
                            keep = False
                    except Exception:
                        pass
            if not keep:
                continue
        # check the absolute distance of the variant to the flanks of the phenotype, and take the closest
        if feature_pos is None or var_pos is None:
            continue
        distance = float(np.min(np.abs(feature_pos - var_pos)))
        # check if we are filtering by distance, and if this distance is within the cis window
        if cis_dist is None or distance <= cis_dist:
            # convert to a float
            pval = float(p_value)
            # check if this is the first time we encounter this gne
            if feature not in test_dict:
                # if so, add it to the dictionary
                test_dict[feature] = {'snps' : [var_pos_key], 'best_snp' : var_pos_key, 'pval' : pval, 'line' : '\t'.join(line)}
            else:
                # if not, then check if the variant is more significant that the current best hit
                if pval < test_dict[feature]['pval']:
                    # if so, update the parameters describing the best hit
                    test_dict[feature]['best_snp'] = var_pos_key
                    test_dict[feature]['pval'] = pval
                    test_dict[feature]['line'] = '\t'.join(line)
                # and add the location of this variant to the list of variants tested for this feature
                test_dict[feature]['snps'].append(var_pos_key)
    QTL.close()
    return genpos_dict, phepos_dict, test_dict, "\t".join(header)
        
def make_test_dict_limix_h5(QTL_h5_path, genpos_dict, phepos_dict, cis_dist=None, genchrom_dict=None, CHROM=None):
    """
    Processes QTL data from an HDF5 file to create a dictionary of test results.

    Parameters:
    -----------
    QTL_h5_path : str
        Path to the HDF5 file containing QTL data.
    genpos_dict : dict
        Dictionary with variant identifiers as keys and their genomic positions as values.
    phepos_dict : dict
        Dictionary with feature identifiers as keys and their start and end positions as values.
    cis_dist : int, optional
        Maximum distance for cis-acting variants (default is None).

    Returns:
    --------
    tuple
        A tuple containing the following elements:
        - test_dict : dict
            Dictionary with feature identifiers as keys and dictionaries of test results as values.
        - header : str
            The header line of the QTL file.

    Notes:
    ------
    - The function reads the QTL data from an HDF5 file.
    - It creates a dictionary for test results, including all variants for each feature, the variant with the lowest p-value, and the corresponding line from the QTL file.
    - Only variants within the specified cis distance are considered.
    """
    
    # make filehandle to the h5
    h5_fh = h5py.File(QTL_h5_path,'r')
    # create a dictionary that has each feature/phenotype/gene as a key, as as the value another dictionary with:
    # all the variants for this phenotype, 
    # the variant with the lowest p-value, 
    # the p-value of the variant with the lowest p-value, 
    # and the line of the QTL output of the variant wih the lowest p-value
    test_dict = {}
    # check each feature
    for feature in h5_fh.keys():
        # get the variants
        vars_feature = h5_fh[feature]['snp_id']
        # and the p-values
        ps_feature = h5_fh[feature]['p_value']
        # extract the chromosomal positon of the phenotype
        phepos = phepos_dict[feature]
        # check each variant
        for i in range(0, len(vars_feature)):
            # decode variant id
            try:
                var_id = vars_feature[i].decode("utf-8")
            except Exception:
                var_id = str(vars_feature[i])
            # if CHROM filter requested, try to use snp_chromosome dataset if present or genchrom_dict
            if CHROM is not None:
                chrom_ok = True
                if 'snp_chromosome' in h5_fh[feature]:
                    try:
                        sc = h5_fh[feature]['snp_chromosome'][i]
                        if isinstance(sc, (bytes, bytearray)):
                            sc = sc.decode('utf-8')
                        if str(sc).replace('chr', '') != str(CHROM).replace('chr', ''):
                            chrom_ok = False
                    except Exception:
                        pass
                elif genchrom_dict is not None and var_id in genchrom_dict:
                    try:
                        sc = str(genchrom_dict[var_id]).replace('chr', '')
                        if sc != str(CHROM).replace('chr', ''):
                            chrom_ok = False
                    except Exception:
                        pass
                if not chrom_ok:
                    continue
            # get the position of the variant
            genpos = genpos_dict.get(var_id)
            if genpos is None:
                # skip variants without position information
                continue
            # if we do cis filtering, check cis distance
            if cis_dist is not None:
                # check the absolute distance of the variant to the flanks of the phenotype, and take the closest, so the smallest value
                distance = min(abs(phepos - genpos))
                # check if this distance is within the cis window
                if distance <= cis_dist:
                    # get the p value
                    pval = float(ps_feature[i])
                    # check if this was the first time we handled this feature
                    if feature not in test_dict:
                        # if so, add the entry
                        test_dict[feature] = {'snps' : [genpos], 'best_snp' : genpos, 'pval' : pval, 'line' : '\t'.join([feature, str(phepos[0]), str(phepos[1])])}
                    else:
                        
                        # otherwise, check if this variant was more significant than the current most significant p-value for this feature
                        if pval < test_dict[feature]['pval']:
                            # if so, update for the best variant info to have this variants information
                            test_dict[feature]['best_snp'] = genpos
                            test_dict[feature]['pval'] = pval
                            #test_dict[feature]['line'] = '\t'.join([feature])
                        # and add the location of this variant to the list of variants tested for this feature
                        test_dict[feature]['snps'].append(genpos)
            else:
                # get the p value
                pval = float(ps_feature[i])
                # check if this was the first time we handled this feature
                if feature not in test_dict:
                    # if so, add the entry
                    test_dict[feature] = {'snps' : [genpos], 'best_snp' : genpos, 'pval' : pval, 'line' : '\t'.join([feature, str(phepos[0]), str(phepos[1])])}
                else:
                    # otherwise, check if this variant was more significant than the current most significant p-value for this feature
                    if pval < test_dict[feature]['pval']:
                        # if so, update for the best variant info to have this variants information
                        test_dict[feature]['best_snp'] = genpos
                        test_dict[feature]['pval'] = pval
                        #test_dict[feature]['line'] = '\t'.join([feature])
                    # and add the location of this variant to the list of variants tested for this feature
                    test_dict[feature]['snps'].append(genpos)
    # return the dictionary and the header of the file
    return test_dict, "\t".join(['feature', 'chromStart', 'chromEnd'])

def bf_eigen_windows(test_dict, gen_dict, phepos_dict, OUT_fh, input_header, var_thresh, window, tmp_dir=None):
    """
    Process a dictionary of SNP-gene tests to calculate the effective Bonferroni correction number.
    
    This function calculates the genotype correlation matrix for the SNPs tested for each gene using windows around the gene.
    It uses the Ledoit-Wolf estimator to calculate a regularized correlation matrix, finds the eigenvalues of this matrix,
    and determines how many eigenvalues are needed to reach the variance threshold. This final value is the effective Bonferroni correction number.
    The function outputs the corrected p-value for the best SNP per gene to a file.

    Parameters:
    test_dict (dict): Dictionary with gene IDs as keys and test results as values.
    gen_dict (dict): Dictionary with SNP positions as keys and genotypes as values.
    phepos_dict (dict): Dictionary with phenotype IDs as keys and their start positions as values.
    OUT_fh (str or file-like object): File handle or path to the output file.
    input_header (str): Header line for the output file.
    var_thresh (float): Variance threshold for determining the effective number of tests.
    window (int): Size of the window to process SNPs around each gene.

    Returns:
    None
    """
    
    # open the output file for writing (gzip if .gz extension)
    if isinstance(OUT_fh, str) and OUT_fh.endswith('.gz'):
        OUT = gzip.open(OUT_fh, 'wt')
    else:
        OUT = open(OUT_fh, 'w')
    # write the header to the file (add grand-total p-value and grand-total test count columns)
    OUT.write(input_header + '\tfeature_eigen_p\tfeature_n_tests\tglobal_eigen_p\tglobal_n_tests\n')
    # keep track of how many phenotypes we have processeds
    counter = 1.0
    # get the genes we are looking at from the dictionary
    genes = test_dict.keys()
    # get the number of genes we are looking at
    numgenes = len(genes)
    # buffer per-gene results to a temporary JSON-lines file to avoid holding many objects in memory
    # choose directory for temp file; if tmp_dir is None, let NamedTemporaryFile pick the system temp dir
    tmp_dir_arg = tmp_dir if tmp_dir else None
    tmpfh = tempfile.NamedTemporaryFile(mode='w+', delete=False, prefix='eigenmt_tmp_', dir=tmp_dir_arg)
    tmp_name = tmpfh.name
    # save the start positions of each phenotype
    TSSs = []
    # check each phenotype and add to the start position of each feature
    for gene in genes:
        TSSs.append(phepos_dict[gene][0])
    # now sort both the start positions of the features and the features based on the start position (ascending)
    TSSs, genes = [list(x) for x in zip(*sorted(zip(TSSs, genes), key=lambda p: p[0]))]
    # build reverse mapping from position -> variant ids (rsids) using genpos_dict if available
    pos_to_rsids = {}
    try:
        gpd = globals().get('genpos_dict', None)
        if gpd:
            for vid, p in gpd.items():
                try:
                    pi = int(p)
                except Exception:
                    try:
                        pi = int(float(p))
                    except Exception:
                        continue
                pos_to_rsids.setdefault(pi, []).append(str(vid))
    except Exception:
        pos_to_rsids = {}
    # check each phenotype
    for gene in genes:
        # calculate the percentage of features we have processed
        perc = (100 * counter / numgenes)
        # every 100 features, print at what percentage we are 
        if (counter % 100) == 0:
            print(str(counter) + ' out of ' + str(numgenes) + ' completed ' + '(' + str(round(perc, 3)) + '%)', flush=True)
        # increase the counter
        counter += 1
        # sort the variants associated with this feature, by their genomic position (the values in the list 'snps')
        snps = np.sort(test_dict[gene]['snps'])
        # quick diagnostics: how many of the listed snps have genotype data (prefer rsid matches)
        try:
            n_listed = len(snps)
            n_present_position = 0
            n_present_rsid = 0
            for s in snps:
                found = False
                try:
                    si = int(s)
                except Exception:
                    si = None
                # try rsids mapped to this position first
                if si is not None and si in pos_to_rsids:
                    for rs in pos_to_rsids[si]:
                        if rs in gen_dict:
                            found = True
                            n_present_rsid += 1
                            break
                # fallback to direct key lookup (pos or string)
                if not found:
                    if s in gen_dict:
                        found = True
                    elif si is not None and si in gen_dict:
                        found = True
                if found:
                    n_present_position += 1
        except Exception:
            n_listed = 0
            n_present_position = 0
            n_present_rsid = 0
        # get how many we eventually found
        n_present = n_present_position + n_present_rsid
        if n_present == 0:
            print('Warning: gene {} has {} listed snps but 0 present in genotypes; TESTS will be equal to total snp number'.format(gene, n_listed), file=sys.stderr)
        else:
            try:
                # if n_present_rsid > 0:
                #     print(f'Note: gene {gene} matched {n_present_rsid} variants by rsid and {n_present - n_present_rsid} by position', flush=True)
                # warn about variants that were only found by position
                if n_present_position > 0:
                    print(f'Warning: gene {gene} has {n_present_position} variant(s) matched only by position; ensure these are correct.', file=sys.stderr)
            except Exception:
                pass
        # compute how many listed variants are missing from genotypes
        try:
            missing_variants = max(0, n_listed - n_present)
        except Exception:
            missing_variants = 0
        if missing_variants > 0:
            print(f'Warning: gene {gene} has {missing_variants} variant(s) missing from genotype data; these will be counted as single tests.', file=sys.stderr)
        # start at zero
        start = 0
        # stop at the end of the end of the max window size for the variants
        stop = window
        # get the number of variants associated with this feature
        M = len(snps)
        # keep track of the effective number of tests
        m_eff = 0
        # keep track of our window
        window_counter = 0
        # process the variants in a window of window_size amount of variants per time
        while start < M:
            # of the window only has a single variant, the number of effective tests is increased by one, there is no correlation structure of variants for a single variant
            if stop - start == 1:
                m_eff += 1
                break ##can't compute eigenvalues for a scalar, so add 1 to m_eff and break from the while loop
            # get all the variant positions that are in the window
            snps_window = snps[start:stop]
            # gather genotype arrays for variants in this window
            rows = []
            lengths = []
            for snp in snps_window:
                row_added = False
                try:
                    s_pos = int(snp)
                except Exception:
                    s_pos = None
                # try rsids mapped to this position first
                if s_pos is not None and s_pos in pos_to_rsids:
                    for rs in pos_to_rsids[s_pos]:
                        if rs in gen_dict:
                            arr = gen_dict[rs]
                            arr = np.asarray(arr)
                            rows.append(arr)
                            lengths.append(arr.shape[0])
                            row_added = True
                            break
                if row_added:
                    continue
                # try exact string keys (chr:pos:ref:alt) by searching for a prefix if genchrom_dict available
                try:
                    if s_pos is not None and 'genchrom_dict' in globals() and globals().get('genchrom_dict') is not None:
                        genchrom = globals().get('genchrom_dict')
                        # iterate rsids to get chrom
                        if s_pos in pos_to_rsids:
                            for rs in pos_to_rsids[s_pos]:
                                chrom = genchrom.get(rs)
                                if chrom is None:
                                    continue
                                prefix = f"{str(chrom).replace('chr','')}:{s_pos}:"
                                for k in gen_dict.keys():
                                    if isinstance(k, str) and k.startswith(prefix) and len(k.split(':')) >= 4:
                                        arr = gen_dict[k]
                                        arr = np.asarray(arr)
                                        rows.append(arr)
                                        lengths.append(arr.shape[0])
                                        row_added = True
                                        break
                                if row_added:
                                    break
                except Exception:
                    pass
                if row_added:
                    continue
                # fallback: numeric or direct key
                if snp in gen_dict:
                    arr = gen_dict[snp]
                    arr = np.asarray(arr)
                    rows.append(arr)
                    lengths.append(arr.shape[0])
                elif s_pos is not None and s_pos in gen_dict:
                    arr = gen_dict[s_pos]
                    arr = np.asarray(arr)
                    rows.append(arr)
                    lengths.append(arr.shape[0])
            # diagnostic: if we have rows but lengths vary, report a brief warning
            if len(rows) > 0 and len(set(lengths)) > 1:
                print('Warning: gene {} window starting at {} has genotype rows with differing sample lengths: {}'.format(gene, start, sorted(set(lengths))), file=sys.stderr)
            # handle windows with no genotype data
            if len(rows) == 0:
                start += window
                stop += window
                if stop > M:
                    stop = M
                continue
            # if lengths differ, keep only those with the most common length
            if len(set(lengths)) > 1:
                # pick the most common length
                from collections import Counter
                c = Counter(lengths)
                common_len = c.most_common(1)[0][0]
                filtered_rows = [r for r, l in zip(rows, lengths) if l == common_len]
                rows = filtered_rows
                lengths = [common_len] * len(rows)
            # after filtering, if fewer than 1 row, skip
            if len(rows) == 0:
                start += window
                stop += window
                if stop > M:
                    stop = M
                continue
            # if only a single variant remains in rows, this contributes 1 to m_eff (handled below), but warn
            if len(rows) == 1:
                # we'll handle single-variant windows below but log for diagnostics
                pass
            # stack into a 2D array (rows: SNPs, cols: samples)
            try:
                genotypes = np.vstack(rows)
            except Exception:
                # fallback: coerce to array and reshape conservatively
                genotypes = np.asarray(rows)
                if genotypes.ndim == 1:
                    genotypes = genotypes.reshape((1, -1))
            # extracting the dimensions
            m, n = np.shape(genotypes)
            # if after shaping there's only a single variant, treat as single-variant window
            if m <= 1:
                m_eff += 1
                start += window
                stop += window
                if stop > M:
                    stop = M
                continue
            # Ledoit-Wolf shrinkage estimator, for estimating more stable covariance matrix,
            gen_corr, alpha = lw_shrink(genotypes) # regularized (shrinkage) covariance matrix and the shrinkage coefficient (alpha)
            # increase the number of windows we used
            window_counter += 1
            # compute the eigenvalues of the regularized (shrinkage) covariance matrix
            eigenvalues = splin.eigvalsh(gen_corr)
            # we cannot have negative correlations, so set those to be zero
            eigenvalues[eigenvalues < 0] = 0
            # find the effective number of tests, using the eigenvalues, the number of variants in the current window and the variance threshold, then add that to the number of effective tests
            m_eff += find_num_eigs(eigenvalues, m, var_thresh)
            # now start at the end of the current window, so the start of the next window_size amount of variants per time
            start += window
            # stop at the new start + the window size again, so the end of the next window_size amount of variants per time
            stop += window
            # if the stop would be bigger than the number of variants left, just take until the end of the list of variants
            if stop > M:
                stop = M
        # count each missing variant as a single independent test
        try:
            m_eff += missing_variants
        except Exception:
            pass
        # per-gene Bonferroni/eigen adjusted p-value
        gene_p = min(test_dict[gene]['pval'] * m_eff, 1)
        # write a compact JSON record for this gene to the temp file
        try:
            rec = {'line': test_dict[gene]['line'], 'orig_pval': float(test_dict[gene]['pval']), 'gene_p': float(gene_p), 'm_eff': float(m_eff)}
        except Exception:
            rec = {'line': str(test_dict[gene].get('line', '')), 'orig_pval': 1.0, 'gene_p': 1.0, 'm_eff': float(m_eff)}
        tmpfh.write(json.dumps(rec) + '\n')
        # free memory periodically
        gc.collect()
    # after processing all genes, compute grand total tests by summing per-feature m_eff from the temp file
    tmpfh.flush()
    tmpfh.seek(0)
    grand_total_tests = 0.0
    try:
        for l in tmpfh:
            try:
                obj = json.loads(l)
                grand_total_tests += float(obj.get('m_eff', 0.0))
            except Exception:
                continue
    finally:
        tmpfh.close()
    if grand_total_tests < 1:
        grand_total_tests = 1
    # write final output rows by reading the temporary file again
    with open(tmp_name, 'r') as R:
        for l in R:
            try:
                obj = json.loads(l)
            except Exception:
                continue
            line = obj.get('line', '')
            orig_pval = float(obj.get('orig_pval', 1.0))
            gene_p = float(obj.get('gene_p', 1.0))
            m_eff = int(obj.get('m_eff', 1))
            grand_p = min(orig_pval * grand_total_tests, 1)
            OUT.write(line + '\t' + str(gene_p) + '\t' + str(m_eff) + '\t' + str(grand_p) + '\t' + str(int(grand_total_tests)) + '\n')
    OUT.flush()
    OUT.close()
    # attempt to remove the temporary file
    try:
        os.unlink(tmp_name)
    except Exception:
        pass

def lw_shrink(genotypes):
    """
    Obtain a smoothed estimate of the genotype correlation matrix using the Ledoit-Wolf shrinkage estimator.
    
    This function uses the method proposed by Ledoit and Wolf to estimate the shrinkage parameter alpha.
    It returns a regularized correlation matrix and the estimated shrinkage parameter.

    Parameters:
    genotypes (np.ndarray): Genotype matrix where rows represent SNPs and columns represent samples.

    Returns:
    tuple: A tuple containing:
        - shrunk_cor (np.matrix): The smoothed correlation matrix.
        - alpha (float or str): The estimated shrinkage parameter. If the SNPs in the window are all in perfect LD, alpha is set to 'NA'.
    """
    
    # use Ledoit-Wolf shrinkage estimator to get a stable shrinkage covariance matrix
    lw = covariance.LedoitWolf()
    # get the dimensions of the genotype data
    m, n = np.shape(genotypes)
    try:
        # transpose the genotype data and perform the fit
        fitted = lw.fit(genotypes.T)
        # extract the alpha
        alpha = fitted.shrinkage_
        # and the shrinkage covariance matrix
        shrunk_cov = fitted.covariance_
        # get the variances of the variants (diagonal of shrinkage cov mat), calculate inverse Square Root, then put that in a matrix representing
        # sanitize diagonal variances to avoid divide-by-zero / infs
        diag_vals = np.diag(shrunk_cov).astype(float)
        # replace non-positive or zero variances with a small epsilon
        eps = 1e-8
        diag_vals_safe = np.where(np.isfinite(diag_vals) & (diag_vals > 0), diag_vals, eps)
        inv_sqrt = diag_vals_safe ** (-0.5)
        shrunk_precision = np.asmatrix(np.diag(inv_sqrt))
        # use this to make it so that the resulting correlation matrix has unit variances along the diagonal,
        # and scaled off-diagonal elements representing correlations
        shrunk_cor = shrunk_precision * shrunk_cov * shrunk_precision
        # guard against any NaN/Inf introduced by numerical issues
        if not np.isfinite(shrunk_cor).all():
            # fallback to an identity-like correlation matrix if computation failed
            m = shrunk_cor.shape[0]
            shrunk_cor = np.asmatrix(np.identity(m))
            alpha = 'NA'
    # if there is perfect LD
    except: #Exception for handling case where SNPs in the window are all in perfect LD
        # the covariances will just be 1, everywhere
        row = np.repeat(1, m)
        # the covariances will just be 1, everywhere
        shrunk_cor = []
        for i in range(0,m):
            shrunk_cor.append(row)
        shrunk_cor = np.asmatrix(shrunk_cor)
        # and we have no real alpha
        alpha = 'NA'
    return shrunk_cor, alpha

def find_num_eigs(eigenvalues, variance, var_thresh):
    """
    Find the number of eigenvalues required to reach a certain threshold of variance explained.
    
    This function sorts the eigenvalues in descending order and sums them until the cumulative sum reaches
    the specified threshold of the total variance.

    Parameters:
    eigenvalues (np.ndarray): Array of eigenvalues.
    variance (float): Total variance to be explained.
    var_thresh (float): Threshold of variance to be explained (as a fraction, e.g., 0.95 for 95%).

    Returns:
    int: The number of eigenvalues required to reach the specified variance threshold.
    """
    # make sure we have a numpy array of finite floats
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    # sort in descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    # sanitize: replace non-finite values with 0
    eigenvalues[~np.isfinite(eigenvalues)] = 0.0
    # total variance represented by eigenvalues
    total = eigenvalues.sum()
    target = float(variance) * float(var_thresh)
    # if total is non-positive (degenerate case), return 1 as a conservative default
    if total <= 0 or not np.isfinite(total):
        return 1
    # if the target is greater than or equal to total, all eigenvalues are required
    if target >= total:
        return int(eigenvalues.size)
    # use cumulative sum and searchsorted to find how many eigenvalues are required
    cumsum = np.cumsum(eigenvalues)
    # searchsorted gives the first index where cumsum >= target
    idx = np.searchsorted(cumsum, target, side='left')
    # idx is 0-based; number of eigenvalues required is idx+1
    return int(idx) + 1


##############MAIN##############

if __name__=='__main__':
    USAGE = """
    Takes in SNP-gene tests from MatrixEQTL output and performs gene level Bonferroni correction using eigenvalue decomposition of
    the genotype correlation matrix. Picks best SNP per gene.
    """

    # initalize parser
    parser = argparse.ArgumentParser(description = USAGE)
    # add the huge list of arguments
    parser.add_argument('--QTL', required = False, help = 'Matrix-EQTL output file with SNP-gene tests (can be gzipped)')
    parser.add_argument('--TENSOR', required = False, help = 'tensorQTL output file with SNP-gene tests (can be gzipped)')
    parser.add_argument('--LIMIX', required = False, help = 'LIMIX-QTL output file with SNP-gene tests (can be gzipped)')
    parser.add_argument('--LIMIX_H5', required = False, help = 'LIMIX-QTL H5 chunk output file with SNP-gene tests')
    parser.add_argument('--GEN', required = False, help = 'genotype matrix file in matrixEQTL format', default=None)
    parser.add_argument('--PLINK1', required = False, help = 'genotype file prefix in plink1 binary format (bed/bim/fam, only supply the base path)', default=None)
    parser.add_argument('--BGEN', required = False, help = 'genotype file prefix in bgen format (only supply the base path)', default=None)
    parser.add_argument('--var_thresh', type=float, default = 0.99, help = 'variance threshold')
    parser.add_argument('--OUT', required = True, help = 'output filename')
    parser.add_argument('--window', type=int, default = 200, help = 'SNP window size')
    parser.add_argument('--GENPOS', required = False, help = 'map of genotype to chr and position (as required by Matrix-eQTL)')
    parser.add_argument('--PHEPOS', required = False, help = 'map of measured phenotypes to chr and position (eg. gene expression to CHROM and TSS; as required by Matrix-eQTL)')
    parser.add_argument('--CHROM', required = False, help = 'Chromosome that is being processed (must match format of chr in POS)')
    parser.add_argument('--cis_dist', type=float, default = None, help = 'threshold for bp distance from the gene TSS to perform multiple testing correction, using no cis dist will consider all variants tested for a feature in the summary stats (default = None)')
    parser.add_argument('--external', action = 'store_true', help = 'indicates whether the provided genotype matrix is different from the one used to call cis-eQTLs initially (default = False)')
    parser.add_argument('--sample_list', default=None, help='File with sample IDs (one per line) to select from genotypes')
    parser.add_argument('--phenotype_groups', default=None, help='File with phenotype_id->group_id mapping')
    parser.add_argument('--tmp_dir', default=None, help='Directory to create temporary files in (optional)')
    # finally parse the arguments
    args = parser.parse_args()

    # Determine which QTL input was supplied and extract the set of variants used
    used_variants = []
    # ensure genchrom_dict exists for optional chromosome-aware filtering
    genchrom_dict = None
    if args.QTL:
        # Matrix-eQTL text format: variant is the first column
        print('Extracting variants from Matrix-eQTL file...', flush=True)
        used_variants = get_variants_from_qtl_file(args.QTL, variant_col=0, genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)
    elif args.TENSOR:
        # tensorQTL parquet format: variants in 'variant_id'
        print('Extracting variants from tensorQTL parquet file...', flush=True)
        used_variants = get_variants_from_qtl_file(args.TENSOR, variant_col='variant_id', genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)
    elif args.LIMIX:
        # LIMIX text format: uses 'snp_id' column
        print('Extracting variants from LIMIX-QTL file...', flush=True)
        used_variants = get_variants_from_qtl_file(args.LIMIX, variant_col='snp_id', genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)
    elif args.LIMIX_H5:
        # LIMIX h5 chunked output; use helper to collect snp_id datasets across features
        print('Extracting variants from LIMIX H5 file...', flush=True)
        used_variants = get_variants_from_limix_h5(args.LIMIX_H5, genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)

    # let the user know what the total number of used variants is
    print('  * variants extracted:', len(used_variants), flush=True)

    # initalize the phenotype position dictionary
    phepos_dict = {}
    # fill phenotype position dict using supplied PHEPOS file
    if args.PHEPOS:
        print('Processing phenotype position file.', flush=True)
        phepos_dict = make_phepos_dict(args.PHEPOS, args.CHROM)
    else:
        # try to extract phenotypes from LIMIX inputs when available
        if args.LIMIX:
            print('Extracting phenotype positions from LIMIX tests file.', flush=True)
            phepos_dict = get_phepos_from_limix_file(args.LIMIX, args.CHROM)
            if not phepos_dict:
                print('Warning: could not extract phenotype positions from LIMIX text file; phepos_dict will be empty.', file=sys.stderr)
        elif args.LIMIX_H5:
            print('Extracting phenotype positions from LIMIX H5 file.', flush=True)
            phepos_dict = get_phepos_from_limix_h5(args.LIMIX_H5, args.CHROM)
            if not phepos_dict:
                print('Warning: could not extract phenotype positions from LIMIX H5 file; phepos_dict will be empty.', file=sys.stderr)
        else:
            # we need the position information, or we cannot sort the variants into cis windows
            sys.exit('No phenotype positions file supplied and no LIMIX input to extract phenotypes from. Provide --PHEPOS or LIMIX/LIMIX_H5 input.')

    # get sample list
    if args.sample_list is not None:
        # extract sample IDs from file
        with open(args.sample_list) as f:
            # they should be with a new sample on each line
            sample_ids = f.read().strip().split('\n')
        # let the user know how many samples we are using
        print('  * using subset of '+str(len(sample_ids))+' samples.')
    else:
        sample_ids = None

    # Build genotype position and genotype dictionaries based on supplied inputs
    genpos_dict = {}
    gen_dict = {}

    # load genotype position data (needed for Matrix-eQTL format)
    if args.GENPOS:
        print('Processing genotype position file.', flush=True)
        genpos_dict = make_genpos_dict(args.GENPOS, args.CHROM)
        # also build a dictionary that stores which chromosome each variant is on (for chromosome-aware filtering)
        genchrom_dict = {}
        # read the genpos file again to populate genchrom_dict
        with open_file(args.GENPOS) as POS:
            # read each line
            POS.readline()
            for line in POS:
                parts = line.rstrip().split()
                if len(parts) >= 2:
                    genchrom_dict[parts[0]] = parts[1]

    # Load genotype matrix if provided (Matrix-eQTL format)
    if args.GEN and (not args.GEN.endswith('.bgen')):
        print('Processing genotype matrix (Matrix-eQTL format).', flush=True)
        # we need the positions for filtering and ordering
        if not genpos_dict:
            print('Warning: GENPOS not supplied; genpos_dict will be empty.', file=sys.stderr)
        # make the dictionary of genotypes keyed by variant ID
        gen_dict = make_gen_dict_matrixqtl(args.GEN, genpos_dict, sample_ids)
    # read bgen format
    elif args.BGEN:
        print('Reading BGEN genotype data...', flush=True)
        try:
            bim, fam, bed, bgen = get_genotype_data_bgen(args.BGEN)
            gen_dict = bgen_to_genotypes(bim, fam, bgen, args.CHROM, minimumProbabilityStep=0.1, genpos_dict=genpos_dict)
        except Exception as e:
            print('Warning: could not read/process BGEN file; genotypes will not be loaded: ' + str(e), file=sys.stderr)
            gen_dict = {}
    # read plink1 format
    elif args.PLINK1:
        print('Reading PLINK1 genotype data...', flush=True)
        try:
            bim, fam = get_genotype_data_plink1(args.PLINK1)
            # if no GENPOS supplied, populate genpos_dict from BIM so we can map variant ids -> positions
            if not genpos_dict:
                try:
                    genpos_dict = {str(r['snp']): int(r['pos']) for _, r in bim.iterrows()}
                except Exception:
                    # ensure at least string positions
                    genpos_dict = {str(r['snp']): r['pos'] for _, r in bim.iterrows()}
                # also create genchrom_dict for chromosome-aware filtering
                try:
                    genchrom_dict = {str(r['snp']): str(r['chrom']).replace('chr', '') for _, r in bim.iterrows()}
                except Exception:
                    genchrom_dict = None
            gen_dict = plink_to_genotypes(bim, fam, args.PLINK1 + '.bed', args.CHROM, sample_ids=sample_ids, genpos_dict=genpos_dict)
        except Exception as e:
            print('Warning: could not read/process PLINK1 files; genotypes will not be loaded: ' + str(e), file=sys.stderr)
            gen_dict = {}

    # Initialize the test_dict and input_header parameters that are filled based on the QTL input type
    test_dict = {}
    input_header = ''

    # phenotype groups for tensorQTL grouping
    if args.phenotype_groups is not None:
        try:
            group_s = pd.read_csv(args.phenotype_groups, sep='\t', index_col=0, header=None, squeeze=True)
            group_size_s = group_s.value_counts()
        except Exception:
            group_size_s = None
    else:
        group_size_s = None

    # the method for extracting the tests that were performed, and the header of the file are different depending on which type of QTL input is supplied
    if args.TENSOR:
        print('Processing tensorQTL tests file.', flush=True)
        test_dict, input_header = make_test_dict_tensorqtl(args.TENSOR, genpos_dict, args.cis_dist, group_size_s=group_size_s, genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)
    elif args.LIMIX:
        print('Processing LIMIX-QTL tests summary file.', flush=True)
        genpos_dict, phepos_dict, test_dict, input_header = make_test_dict_limix(args.LIMIX, args.cis_dist, genchrom_dict if 'genchrom_dict' in locals() else None, args.CHROM)
    elif args.LIMIX_H5:
        print('Processing LIMIX-QTL tests h5 file.', flush=True)
        test_dict, input_header = make_test_dict_limix_h5(args.LIMIX_H5, genpos_dict, phepos_dict, args.cis_dist, genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)
    elif args.QTL:
        print('Processing Matrix-eQTL tests file.', flush=True)
        if args.external:
            test_dict, input_header = make_test_dict_external(args.QTL, gen_dict, genpos_dict, phepos_dict, args.cis_dist, genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)
        else:
            test_dict, input_header = make_test_dict_matrixqtl(args.QTL, gen_dict, genpos_dict, phepos_dict, args.cis_dist, genchrom_dict=genchrom_dict if 'genchrom_dict' in locals() else None, CHROM=args.CHROM)

    ##Perform BF correction using eigenvalue decomposition of the correlation matrix
    print('Performing eigenMT correction.', flush=True)
    bf_eigen_windows(test_dict, gen_dict, phepos_dict, args.OUT, input_header, args.var_thresh, args.window, tmp_dir=args.tmp_dir)
    # make a checksum
    create_hash_file(args.OUT)