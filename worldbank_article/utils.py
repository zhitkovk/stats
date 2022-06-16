import numpy as np
import pandas as pd

from scipy.stats import t, norm
from tqdm import tqdm
from statsmodels.stats.power import TTestIndPower

def t_test(test, control):
    """
    Simple version. Assume equal sample sizes and equal variances.
    I have created my own version of t_test because I wanted it to return more stuff than the basic one
    
    Returns:
    - estimated delta
    - its standard deviation
    - value of t statistics
    - two sided pvalue
    """
    
    assert np.size(test) == np.size(control)
    
    sample_size = np.size(test)
    
    delta = np.mean(test) - np.mean(control)
    std = np.sqrt(np.sum([np.var(test, ddof=1), np.var(control, ddof=1)]) / sample_size)
    
    tvalue = delta / std
    pvalue = 2 * t.sf(tvalue, df = 2 * sample_size - 2)
    
    return delta, std, tvalue, pvalue


def mde(alpha, stat_test_power, std):
    """
    We want to find a threshold value c, such that if the sample mean difference is larger than c, 
    we'll reject the null hypothesis. Basically c is difference between theta_exp and theta_cnt, that
    we can detect with certain alpha, beta and sample size.
    
    See formula for sample size estimation here:
    https://online.stat.psu.edu/stat415/lesson/25/25.3
    """
    
    qnts = norm.isf(alpha / 2.0) + norm.isf(1.0 - stat_test_power)
    effect_size = qnts * std
    
    return effect_size


def relative_mde(control, abs_mde):
    """
    Same as simple mde, but normalized by control group in order to get percentage values
    """
    
    return 100.00 * abs_mde / np.mean(control)


def simulate_tests(nsims, ex_ante_power, **tp):
    """
    Simulate a bunch of t tests and save the test results for each simulation to pd.DataFrame.
    
    You need to provide number of simulations, ex_ante_power you wish to achieve
    and dict of test parameters. Example dict:
    
    test_params_02 = {
    'sample_size': 400,
    'effect': 0.2,
    'alpha': 0.05,}
    """
    
    effects = []
    std_devs = []
    tvalues = []
    pvalues = []
    obs_pwr = []
    obs_mde = []
    obs_rel_mde = []
    
    for i in tqdm(range(nsims)):
        x = norm.rvs(size=tp['sample_size'], loc=0.0)  
        y = norm.rvs(size=tp['sample_size'], loc=tp['effect']) 
        
        tt = t_test(y, x)
        
        # ex post values of power, mde and relative mde
        omde = mde(tp['alpha'], ex_ante_power, tt[1])
        orelmde = relative_mde(np.mean(x), omde)
        
        # calculate power using the observed effect size
        opwr = TTestIndPower().solve_power(effect_size=tt[0],
                                           nobs1=tp['sample_size'],
                                           ratio=1.0,
                                           alpha=tp['alpha'])
        
        # collect values to a list
        effects.append(tt[0])
        std_devs.append(tt[1])
        tvalues.append(tt[2])
        pvalues.append(tt[3])
        
        obs_mde.append(omde)
        obs_rel_mde.append(orelmde)
        obs_pwr.append(opwr)
    
    return pd.DataFrame({'effects': effects,
                         'std_devs': std_devs,
                         'tvalues': tvalues,
                         'pvalues': pvalues,
                         'obs_power': obs_pwr,
                         'obs_rel_mde': obs_rel_mde,
                         'obs_mde': obs_mde})


def get_power_stats(df):
    """"
    Calculate posthoc power statistics for subsamples of data
    """
    
    pwr_insign_res = np.mean(df['obs_power'][df['pvalues'] >= 0.05])
    pwr_sign_res = np.mean(df['obs_power'][df['pvalues'] < 0.05])
    pwr_all_res = np.mean(df['obs_power'])
    
    return dict(zip(['mean_power_pv>=0.05', 'mean_power_pv<0.05', 'mean_power_all'],
                    [pwr_insign_res, pwr_sign_res, pwr_all_res]))


def modify_mpl(ax, title='', x_title='', y_title=''):
    """
    Make matplotlib charts cleaner
    """
    
        
    ax.set_xlabel(x_title, 
                  horizontalalignment='right', 
                  x=1,
                  labelpad=12,
                  fontsize=14,
                  color='#464346')
    
    ax.set_ylabel(y_title, 
                  rotation='horizontal',
                  horizontalalignment='right', 
                  x=0,
                  y=0.9,
                  labelpad=12,
                  fontsize=14,
                  color='#464346')
    
    ax.set_title(title, 
                 loc='left',
                 y=1.05,
                 color='#464346',
                 fontsize=22,
                 fontweight=1000)


    # grey ticks
    ax.tick_params('x', labelsize=12, labelcolor='#4F4E50')
    ax.tick_params('y', labelsize=12, labelcolor='#4F4E50')


    # rm useless borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
