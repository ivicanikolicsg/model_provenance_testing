import numpy as np
import glob, os, copy, math, csv, time, pickle, hashlib, random, sys, pickle, argparse, re, json, subprocess
from scipy import stats
from scipy.stats import fisher_exact
from datetime import datetime
import multiprocessing
from multiprocessing import Manager


cached_prompts_folder = 'cached_prompts'
if not os.path.exists(cached_prompts_folder):
    os.makedirs(cached_prompts_folder)

cached_tokens_folder = 'cached_tokens'
if not os.path.exists(cached_tokens_folder):
    os.makedirs(cached_tokens_folder)

tester_runs_folder = 'runs'
if not os.path.exists(tester_runs_folder):
    os.makedirs(tester_runs_folder)


file_random_sentences = './data/random_sentences.txt'



def load_models_from_file(filepath):
    if not os.path.exists(filepath):
        raise ValueError(f"File {filepath} does not exist") 
    models = []
    with open(filepath, 'r') as f:
        for line in f:
            model_name = line.strip()
            if len(model_name) > 0:
                models.append(model_name)
    return models





def compare_bernoulli_means(t1, n1, t2, n2, small_count_threshold=5):
    """
    Compare two bernoulli distributions using either Fisher's exact test (for small counts)
    or z-test (for larger counts).
    
    Parameters:
    t1, n1: successes and total trials for first group
    t2, n2: successes and total trials for second group
    small_count_threshold: threshold below which Fisher's exact test is used
    
    Returns:
    float: p-value for the comparison
    """
    # Input validation
    if n1 <= 0 or n2 <= 0:
        return 1.0
    
    # Create contingency table
    contingency_table = np.array([[t1, t2],
                                [n1-t1, n2-t2]])


    
    # Use Fisher's exact test if any count is below threshold
    if min(t1, t2, n1-t1, n2-t2) < small_count_threshold:
        _, p_value = stats.fisher_exact(contingency_table)
        return p_value
    
    # Otherwise use z-test
    # Calculate sample proportions
    p1 = t1/n1
    p2 = t2/n2
    
    # Calculate pooled proportion
    p_pooled = (t1 + t2) / (n1 + n2)
    
    # Handle edge cases
    if p_pooled == 0 or p_pooled == 1:
        return 1.0
    
    # Standard error with small value protection
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    # Avoid division by zero
    if se < 1e-10:
        return 1.0
    
    # Z-statistic
    z_stat = (p1 - p2) / se
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    
    return p_value




