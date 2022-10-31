# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def rm_ext_and_nan(CTG_features, extra_feature):

    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_ctg = {ctg_key: [cell_val for cell_val in CTG_features.replace(regex='[^0-9]', value=np.nan)[ctg_key]
                       if pd.notna(cell_val)] for ctg_key in CTG_features.keys() if ctg_key is not extra_feature}

    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas dataframe of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe c_cdf containing the "clean" features
    """

    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_cdf = CTG_features.copy()
    c_cdf.drop(columns=[extra_feature], inplace=True)
    c_cdf = c_cdf.replace(regex='[^0-9]', value=np.nan)

    for feature in c_cdf.columns:
        one_feature = c_cdf.loc[:, feature].tolist()

        for idx, x in enumerate(one_feature):
            while np.isnan(x):
                rand_idx = np.random.choice(len(one_feature))
                x = one_feature[rand_idx]
                c_cdf.loc[idx, feature] = x

    c_cdf = c_cdf.convert_dtypes()
    # -------------------------------------------------------------------------
    return c_cdf


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_samp
    :return: Summary statistics as a dictionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary = {feature: {'min': c_feat.loc[:, feature].min(),
                           'Q1': c_feat.loc[:, feature].quantile(0.25),
                           'median': c_feat.loc[:, feature].median(),
                           'Q3': c_feat.loc[:, feature].quantile(0.75),
                           'max': c_feat.loc[:, feature].max()} for feature in c_feat.columns}
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_samp
    :param d_summary: Output of sum_stat
    :return: Dataframe containing c_feat with outliers removed
    """
    c_no_outlier = c_feat.copy()
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    for feature in c_no_outlier.columns:
        Q1 = d_summary[feature]['Q1']
        Q3 = d_summary[feature]['Q3']
        IQR = Q3 - Q1
        minimum_tresh = Q1 - 1.5 * IQR
        maximum_tresh = Q3 + 1.5 * IQR

        one_feature = c_no_outlier.loc[:, feature].tolist()
        for idx, cell in enumerate(one_feature):
            if cell < minimum_tresh or cell > maximum_tresh:
                c_no_outlier.loc[idx, feature] = np.nan

    # -------------------------------------------------------------------------
    return c_no_outlier


def phys_prior(c_samp, feature, thresh):
    """

    :param c_samp: Output of nan2num_samp
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = []
    for cell in c_samp.loc[:, feature]:
        if cell < thresh:
            filt_feature.append(cell)
    # -------------------------------------------------------------------------
    return np.array(filt_feature)


class NSD:

    def __init__(self):
        self.max = np.nan
        self.min = np.nan
        self.mean = np.nan
        self.std = np.nan
        self.fit_called = False
    
    def fit(self, CTG_features):
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        self.max = CTG_features.apply(np.max)
        self.min = CTG_features.apply(np.min)
        self.mean = CTG_features.apply(np.mean)
        self.std = CTG_features.apply(np.std)
        # -------------------------------------------------------------------------
        self.fit_called = True

    def transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        """
        Note: x_lbl should only be either: 'Original values [N.U]', 'Standardized values [N.U.]', 'Normalized values [N.U.]' or 'Mean normalized values [N.U.]'
        :param mode: A string determining the mode according to the notebook
        :param selected_feat: A two elements tuple of strings of the features for comparison
        :param flag: A boolean determining whether or not plot a histogram
        :return: Dataframe of the normalized/standardized features called nsd_res
        """
        ctg_features = CTG_features.copy()
        if self.fit_called:
            if mode == 'none':
                nsd_res = ctg_features
                x_lbl = 'Original values [N.U]'
            # ------------------ IMPLEMENT YOUR CODE HERE (for the remaining 3 methods using elif):--------------------
            elif mode == 'standard':
                nsd_res = (ctg_features-self.mean)/self.std
                x_lbl = 'Standard values [N.U]'
            elif mode == 'MinMax':
                nsd_res = (ctg_features-self.min)/(self.max-self.min)
                x_lbl = 'MinMax values [N.U]'
            elif mode == 'mean':
                nsd_res = (ctg_features-self.mean)/(self.max-self.min)
                x_lbl = 'Mean values [N.U]'
            # -------------------------------------------------------------------------
            if flag:
                self.plot_hist(nsd_res, mode, selected_feat, x_lbl)
            return nsd_res
        else:
            raise Exception('Object must be fitted first!')

    def fit_transform(self, CTG_features, mode='none', selected_feat=('LB', 'ASTV'), flag=False):
        self.fit(CTG_features)
        return self.transform(CTG_features, mode=mode, selected_feat=selected_feat, flag=flag)

    def plot_hist(self, nsd_res, mode, selected_feat, x_lbl):
        x, y = selected_feat
        if mode == 'none':
            bins = 50
        else:
            bins = 80
        # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(2, 1, 1)
        ax1.hist(nsd_res.loc[:, x], bins=bins)
        ax1.set(title=f'feature: {x}', xlabel=x_lbl)

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(2, 1, 2)
        ax2.hist(nsd_res.loc[:, y], bins=bins)
        ax2.set(title=f'feature: {y}', xlabel=x_lbl)
        # -------------------------------------------------------------------------
