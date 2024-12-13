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

##############FUNCTIONS##############

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

def make_genpos_dict(POS_fh, CHROM):
    """
    Read SNPs and their positions from a file and create a dictionary.
    
    Parameters:
    POS_fh (str or file-like object): File handle or path to the file containing SNP positions.
    CHROM (str): Chromosome identifier to filter SNPs by chromosome.
    
    Returns:
    dict: Dictionary with SNP IDs as keys and their positions as values, filtered by the specified chromosome.
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
            # we do one chromosome at a time, so only if we are looking at the relevant chromosome, will we add to the dictionary
            if line[1] == CHROM:
                # add to the dictionary the variant as key, and the chromosomal position as the value
                pos_dict[line[0]] = float(line[2])
    return pos_dict

def make_phepos_dict(POS_fh, CHROM):
    """
    Read phenotypes (probes, genes, peaks) with their start and end positions from a file and create a dictionary.
    
    Parameters:
    POS_fh (str or file-like object): File handle or path to the file containing phenotype positions.
    CHROM (str): Chromosome identifier to filter phenotypes by chromosome.
    
    Returns:
    dict: Dictionary with phenotype IDs as keys and their start and end positions as values, filtered by the specified chromosome.
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
            # we do one chromosome at a time, so only if we are looking at the relevant chromosome, will we add to the dictionary
            if line[1] == CHROM:
                # grab the last two values, which should be the start and stop
                pos_array = np.array(line[2:4])
                # add to the dictionary the phenotype as key, and the list of start and stop as values
                pos_dict[line[0]] = np.float64(pos_array)
    return pos_dict

def get_genotype_data_bgen(bgen_loc):
    # read the bgen file using bgen_reader
    bgen = read_bgen(bgen_loc, verbose=False)
    # the bed will be empty
    bed=None
    # fake the fam to be like plink format
    fam = bgen['samples']
    fam = fam.to_frame("iid")
    fam.set_index('iid',inplace=True)
    fam.index = fam.index.astype(str)
    # fake the bim to be like plink format
    bim = bgen['variants'].compute()
    bim = bim.assign(i = range(bim.shape[0]))
    bim['id'] = bim['rsid']
    bim = bim.rename(index = str, columns = {"id": "snp"})
    bim['a1'] = bim['allele_ids'].str.split(",", expand=True)[0]
    bim.index = bim["snp"].astype(str).values
    bim.index.name = "candidate"
        
    ##Fix chromosome ids
    #bim['chrom'].replace('^chr','',regex = True,inplace = True)
    bim['chrom'] = bim['chrom'].replace('^chr', '', regex=True)
    #bim['chrom'].replace(['X', 'Y', 'XY', 'MT'], ['23', '24', '25', '26'],inplace=True)
    bim['chrom'] = bim['chrom'].replace(['X', 'Y', 'XY', 'MT'], ['23', '24', '25', '26'])
    ##Remove non-biallelic & non-ploidy 2 (to be sure). (These can't happen in binary plink files).
    print("Warning, the current software only supports biallelic SNPs and ploidy 2")
    bim = bim.loc[np.logical_and(bim['nalleles'] < 3,bim['nalleles'] > 0),:]

    # return the variables
    return bim,fam,bed,bgen

def bgen_to_positions_and_genotypes(bim, fam, bgen, CHROM, minimumProbabilityStep=0.1, genpos_dict=None):
    # create dictionary of chromosomal positions as keys and the genotypes for that variant as a numpy array
    gen_dict = {}
    # get the SNP indices from the bim
    snp_idxs = bim['i'].values
    # get the SNP identifiers
    snp_names = bim.rsid.tolist()
    # get the chromosomal locations
    chrom_locs = bim.pos.tolist()
    # get the chromosomes present
    snp_chromosomes = bim.chrom.tolist()
    # subset to the variant indices that are of the chromosome we are looking at
    snp_idxs = [snp_idxs[i] for i, x in enumerate(snp_chromosomes) if x == CHROM]
    snp_names = [snp_names[i] for i, x in enumerate(snp_chromosomes) if x == CHROM]
    # and the variants that are our search space
    if genpos_dict is not None:
        #snp_idxs = snp_idxs[x for x in snp_names if x in genpos_dict.keys()]
        snp_idxs = [snp_idxs[i] for i, x in enumerate(snp_names) if x in genpos_dict.keys()]
    # check each snp index
    for snpId in snp_idxs :
        # get the genotype
        geno = bgen["genotype"][snpId].compute()
        if (all(geno["ploidy"]==2)) :
            # initialize the dosage
            snp_df_dosage_t = None
            # depending on the phasing we might do slightly different things
            if(geno["phased"]):
                snp_df_dosage_t = geno["probs"][:,[0,2]].sum(1).astype(float)
                naId = (np.amax(geno["probs"][:,:2],1)+np.amax(geno["probs"][:,2:4],1))<(1+minimumProbabilityStep)
                snp_df_dosage_t[naId] = -1
            else :
                snp_df_dosage_t = ((geno["probs"][:,0]* 2)+geno["probs"][:,1]).astype(float)
                naId = np.amax(geno["probs"][:,:3],1)<((1/3)+minimumProbabilityStep)
                snp_df_dosage_t[naId] = -1
            # convert to float values
            snp_df_dosage_t = np.float64(snp_df_dosage_t)
            # set the variants we had marked as -1, so the unknown ones, to be the mean of the variants that we do know the values of
            snp_df_dosage_t[snp_df_dosage_t == -1] = np.mean(snp_df_dosage_t[snp_df_dosage_t != -1])
            # finally put in the dictionary for each variant position as key, the genotypes for that variant
            gen_dict[chrom_locs[snpId]] = snp_df_dosage_t
    return gen_dict
    
def make_gen_dict(GEN_fh, pos_dict, sample_ids=None):
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

def make_test_dict(QTL_fh, gen_dict, genpos_dict, phepos_dict, cis_dist=None, pvalue_column=None):
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

def make_test_dict_tensorqtl(QTL_fh, genpos_dict, cis_dist=None, group_size_s=None):
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

def make_test_dict_external(QTL_fh, gen_dict, genpos_dict, phepos_dict, cis_dist=None, pvalue_column=None, variant_index_col=0, feature_index_col=1):
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

def make_test_dict_limix(QTL_fh, cis_dist=None):
    # for the limix output, the feature column is actually the first column
    feature_index_col = 0
    # and the variant is the second one
    variant_index_col = 1
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

    # get indices of each column
    feature_id_index = header.index('feature_id')
    snp_id_index = header.index('snp_id')
    feature_chromosome_index = header.index('feature_chromosome')
    feature_start_index = header.index('feature_start')
    feature_end_index = header.index('feature_end')
    snp_chromosome_index = header.index('snp_chromosome')
    snp_position_index = header.index('snp_position')
    
    # check each line in the QTL output
    for line in QTL:
        # line looks like this: 
        # feature_id,snp_id,p_value,beta,beta_se,empirical_feature_p_value,feature_chromosome,feature_start,feature_end,ENSG,biotype,n_samples,n_e_samples,snp_chromosome,snp_position,assessed_allele,call_rate,maf,hwe_p
        # remove trailing newline, and split by whitespace
        line = line.rstrip().split()
        # grab the values
        variant = line[snp_id_index]
        feature = line[feature_id_index]
        p_value = line[pvalIndex]
        feature_pos = [line[feature_start_index], line[feature_end_index]]
        var_pos = line[snp_position_index]
        # put the variant position in the dictionary
        genpos_dict[variant] = var_pos
        # features positions too
        phepos_dict[feature] = feature_pos
        # check the absolute distance of the variant to the flanks of the phenotype, and take the closest, so the smallest value
        distance = min(abs(feature_pos - var_pos))
        # check if we are filtering by distance, and if this distance is within the cis window
        if cis_dist is None or distance <= cis_dist:
            # convert to a float
            pval = float(p_value)
            # check if this is the first time we encounter this gne
            if gene not in test_dict:
                # if so, add it to the dictionary
                test_dict[gene] = {'snps' : [snp], 'best_snp' : var_pos, 'pval' : pval, 'line' : '\t'.join(line)}
            else:
                # if not, then check if the variant is more significant that the current best hit
                if pval < test_dict[gene]['pval']:
                    # if so, update the parameters describing the best hit
                    test_dict[gene]['best_snp'] = var_pos
                    test_dict[gene]['pval'] = pval
                    test_dict[gene]['line'] = '\t'.join(line)
                # and add the location of this variant to the list of variants tested for this feature
                test_dict[gene]['snps'].append(var_pos)
    QTL.close()
    return genpos_dict, phepos_dict, test_dict, "\t".join(header)
        
        
    

def make_test_dict_limix_h5(QTL_h5_path, genpos_dict, phepos_dict, cis_dist=None):
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
            # get the position of the variant
            genpos = genpos_dict[vars_feature[i].decode("utf-8")]
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
                        if pval < test_dict[gene]['pval']:
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

def bf_eigen_windows(test_dict, gen_dict, phepos_dict, OUT_fh, input_header, var_thresh, window):
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
    
    # open the output file for writing
    OUT = open(OUT_fh, 'w')
    # write the header to the file
    OUT.write(input_header + '\tBF\tTESTS\n')
    # keep track of how many phenotypes we have processeds
    counter = 1.0
    # get the genes we are looking at from the dictionary
    genes = test_dict.keys()
    # get the number of genes we are looking at
    numgenes = len(genes)
    # save the start positions of each phenotype
    TSSs = []
    # check each phenotype and add to the start position of each feature
    for gene in genes:
        TSSs.append(phepos_dict[gene][0])
    # now sort both the start positions of the features and the features based on the start position (ascending)
    TSSs, genes = [list(x) for x in zip(*sorted(zip(TSSs, genes), key=lambda p: p[0]))]
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
            # we'll keep track of the genotypes of all variants in this window
            genotypes = []
            # check each variant position in this window
            for snp in snps_window:
                # if we have genotype information for the variant at this window, add it to the list (of lists)
                if snp in gen_dict:
                    genotypes.append(gen_dict[snp])
            # convert the doublet list into a double array
            genotypes = np.asarray(genotypes)
            # extracting the dimensions
            m, n = np.shape(genotypes)
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
        OUT.write(test_dict[gene]['line'] + '\t' + str(min(test_dict[gene]['pval'] * m_eff, 1)) + '\t' + str(m_eff) + '\n')
        OUT.flush()
        gc.collect()
    OUT.close()

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
        shrunk_precision = np.asmatrix(np.diag(np.diag(shrunk_cov)**(-.5)))
        # use this to make it so that the resulting correlation matrix has unit variances along the diagonal, and scaled off-diagonal elements representing correlations
        shrunk_cor = shrunk_precision * shrunk_cov * shrunk_precision
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
    # sort the eigenvalues in ascending order, then reverse to descending order
    eigenvalues = np.sort(eigenvalues)[::-1]
    # keep track of the sum of eigenvalues
    running_sum = 0
    # keep track of the number of eigenvalues summed
    counter = 0
    # as long as the sum of eigenvalues is smaller than the variance times the variance threshold
    while running_sum < variance * var_thresh:
        # keep summing the eigenvalues
        running_sum += eigenvalues[counter]
        # and increasing the counter
        counter += 1
    # return the number of eigenvalues that was required to get to the threshold of variance explained
    return counter


##############MAIN##############

if __name__=='__main__':
    USAGE = """
    Takes in SNP-gene tests from MatrixEQTL output and performs gene level Bonferroni correction using eigenvalue decomposition of
    the genotype correlation matrix. Picks best SNP per gene.
    """

    parser = argparse.ArgumentParser(description = USAGE)
    parser.add_argument('--QTL', required = True, help = 'Matrix-EQTL output file for one chromosome')
    parser.add_argument('--GEN', required = True, help = 'genotype matrix file')
    parser.add_argument('--var_thresh', type=float, default = 0.99, help = 'variance threshold')
    parser.add_argument('--OUT', required = True, help = 'output filename')
    parser.add_argument('--window', type=int, default = 200, help = 'SNP window size')
    parser.add_argument('--GENPOS', required = True, help = 'map of genotype to chr and position (as required by Matrix-eQTL)')
    parser.add_argument('--PHEPOS', required = True, help = 'map of measured phenotypes to chr and position (eg. gene expression to CHROM and TSS; as required by Matrix-eQTL)')
    parser.add_argument('--CHROM', required = True, help = 'Chromosome that is being processed (must match format of chr in POS)')
    parser.add_argument('--cis_dist', type=float, default = None, help = 'threshold for bp distance from the gene TSS to perform multiple testing correction, using no cis dist will consider all variants tested for a feature in the summary stats (default = None)')
    parser.add_argument('--external', action = 'store_true', help = 'indicates whether the provided genotype matrix is different from the one used to call cis-eQTLs initially (default = False)')
    parser.add_argument('--sample_list', default=None, help='File with sample IDs (one per line) to select from genotypes')
    parser.add_argument('--phenotype_groups', default=None, help='File with phenotype_id->group_id mapping')
    args = parser.parse_args()

    
    ##Make phenotype position dict
    print('Processing phenotype position file.', flush=True)
    phepos_dict = make_phepos_dict(args.PHEPOS, args.CHROM)

    ### get sample list
    if args.sample_list is not None:
        with open(args.sample_list) as f:
            sample_ids = f.read().strip().split('\n')
        print('  * using subset of '+str(len(sample_ids))+' samples.')
    else:
        sample_ids = None

    ##Make SNP position dict
    print('Processing genotype position file.', flush=True)
    genpos_dict = make_genpos_dict(args.GENPOS, args.CHROM)
    
    # for matrixEQTL input/output we have a separate position file
    if args.GEN.endswith('.bgen') is False:
        ##Make genotype dict
        print('Processing genotype matrix.', flush=True)
        gen_dict = make_gen_dict(args.GEN, genpos_dict, sample_ids)
    # for bgen we have both in the same file
    else:
        # read bgen file
        print('Reading genotype data (bgen format)', flush=True)
        bim,fam,bed,bgen = get_genotype_data_bgen(bgen_test_loc)
        # and get the position and genotype data
        print('Processing genotype data (bgen format)')
        gen_dict = bgen_to_positions_and_genotypes(bim, fam, bgen, args.CHROM, minimumProbabilityStep=0.1, genpos_dict)
    

    ##Make SNP-gene test dict
    if not args.external:
        # parquet format (such as tensorqtl)
        if args.QTL.endswith('.parquet'):
            print('Processing tensorQTL tests file.', flush=True)
            if args.phenotype_groups is not None:
                group_s = pd.read_csv(args.phenotype_groups, sep='\t', index_col=0, header=None, squeeze=True)
                group_size_s = group_s.value_counts()
            else:
                group_size_s = None
            test_dict, input_header = make_test_dict_tensorqtl(args.QTL, genpos_dict, args.cis_dist, group_size_s=group_size_s)
        # full summary stats of LIMIX-QTL
        elif args.QTL.endswith('qtl_results_all.txt.gz'):
            print('Processing LIMIX-QTL tests summary file.', flush=True)
            genpos_dict, phepos_dict, test_dict, input_header = make_test_dict_limix(args.QTL, genpos_dict, args.cis_dist)
        # chunked summary stats of LIMIX-QTL
        elif args.QTL.endswith('h5'):
            print('Processing LIMIX-QTL tests h5 file.', flush=True)
            test_dict, input_header = make_test_dict_limix_h5(args.QTL, genpos_dict, phepos_dict, args.cis_dist)
        # matrixEQTL format
        else:
            print('Processing Matrix-eQTL tests file.', flush=True)
            test_dict, input_header = make_test_dict(args.QTL, gen_dict, genpos_dict, phepos_dict, args.cis_dist)
    # matrixEQTL format with different genotype data
    else:
        print('Processing Matrix-eQTL tests file. External genotype matrix and position file assumed.', flush=True)
        test_dict, input_header = make_test_dict_external(args.QTL, gen_dict, genpos_dict, phepos_dict, args.cis_dist)


    ##Perform BF correction using eigenvalue decomposition of the correlation matrix
    print('Performing eigenMT correction.', flush=True)
    bf_eigen_windows(test_dict, gen_dict, phepos_dict, args.OUT, input_header, args.var_thresh, args.window)
