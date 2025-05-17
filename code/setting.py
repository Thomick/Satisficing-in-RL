# Simple module for storing global settings
# Used for sharing configuration between modules

def init():
    """Initialize global variables used across the experiment"""
    # Confidence level for UCB-alpha algorithm
    global delta_conf_level
    
    # Alpha parameter for UCB-alpha
    global alpha
    
    # Variance of prior (for Bayesian algorithms)
    global val_prior
    
    # Threshold for satisfaction (used in some algorithms)
    global threshold_sat
