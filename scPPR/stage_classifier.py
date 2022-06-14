import sys

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

np.random.seed()

class StageClassifier(object):
    stages = ('G0/G1', 'S', 'G2/M')
    
    def __init__(self):
        self.stage_probability = pd.DataFrame(columns=('gene1', 'gene2', 'stage', 'G0/G1', 'S', 'G2/M'))
    
    def fit(self, stage_labeled_expression, cycle_genes):
        """Estimate the probability of each gene pairs, i.e. gene a and gene b 
        where expression levels satisfy Ea > Eb
        
        
        Parameters
        ----------
        stage_labeled_expression : DataFrame of shape (n_cycle_genes, n_samples)
            with genes as index and stage ('G0/G1', 'S', 'G2/M') as column
        cycle_genes : array-like of shape (n_cycle_genes, )
        """
        stage_labeled_expression = stage_labeled_expression.copy()
        stage_labeled_expression.index = stage_labeled_expression.index.str.upper()
        stage_labeled_expression = stage_labeled_expression.sort_index()
        
        cycle_genes_set = set(map(str.upper, cycle_genes))
        train_gene_set = set(stage_labeled_expression.index)
        cycle_genes = sorted(list(cycle_genes_set & train_gene_set))
        stage_labeled_expression = stage_labeled_expression.loc[cycle_genes]
        
        n_stage_samples = stage_labeled_expression.columns.value_counts()
        
        for gene1, gene2 in permutations(cycle_genes, 2):
            pair_compare = (stage_labeled_expression.loc[gene1] >= stage_labeled_expression.loc[gene2]).apply(lambda x: 1 if x else 0)
            stage = self._test_pair(pair_compare, n_stage_samples)
            if stage:
                line = {
                    'stage': stage,
                    'gene1': gene1,
                    'gene2': gene2
                }
                self._calculate_probability(pair_compare, n_stage_samples, line)
                self.stage_probability = self.stage_probability.append(line, ignore_index=True)
        self.stage_probability = self.stage_probability.set_index(['gene1', 'gene2'])
        for stage, pair_num in self.stage_probability.stage.value_counts().items():
            print("Number of {} pairs: {}".format(stage, pair_num))
    
    def predict(self, gene_expression, estimated_phases):
        """Predict cycle stage for samples
        
        Parameters
        ----------
        gene_expression : DataFrame of shape (n_genes, n_samples)
            with gene names as index
        estimated_phases : array-like of shape (n_samples, )
        
        Returns
        -------
        estimated_stages : array-like of shape (n_samples, )
        """
        gene_expression = gene_expression.copy()
        gene_expression.index = gene_expression.index.str.upper()
        gene_expression = gene_expression.sort_index()
        n_samples = gene_expression.shape[1]
        
        gm = GaussianMixture(n_components=3, random_state=0)
        max_likelihood = -sys.maxsize - 1
        optimum_offset = None
        prev_offset = None
        predicted_labels = None
        for offset in sorted(estimated_phases):
            if offset != prev_offset:
                prev_offset = offset
                phases = (estimated_phases - offset) % (2 * np.pi)
                phases = phases.reshape((-1, 1))
                gm = gm.fit(phases)
                likelihood = np.sum(gm.score_samples(phases))
                if likelihood > max_likelihood:
                    max_likelihood = likelihood
                    optimum_offset = offset
                    predicted_labels = gm.predict(phases)
                    self.predicted_labels = predicted_labels
        print('Optimum offset:', optimum_offset)
        
        stage_pair_prob = np.zeros((3, n_samples))
        for (gene1, gene2), pair_prob in self.stage_probability.iterrows():
            compare = gene_expression.loc[gene1] >= gene_expression.loc[gene2]
            for i, stage in enumerate(self.stages):
                prob1 = np.log(pair_prob[stage])
                prob2 = np.log(1 - pair_prob[stage])
                stage_pair_prob[i] += compare.map(lambda x: prob1 if x else prob2)
        self.stage_pair_prob = stage_pair_prob

        self.cluster_stage_prob = np.zeros((3, 3))
        for i in range(3):
            self.cluster_stage_prob[i] = np.sum(stage_pair_prob[:, np.where(predicted_labels == i)[0]], axis=1)

        max_likelihood = -sys.maxsize - 1
        optimum_order = None
        for p in permutations(range(3), 3):
            likelihood = 0
            for i, idx in enumerate(p):
                likelihood += self.cluster_stage_prob[i, idx]
            print(likelihood, p)
            if likelihood > max_likelihood:
                max_likelihood = likelihood
                optimum_order = p
        print('maximun likelihood:', max_likelihood)
        ordered_cluster_stage = [self.stages[_] for _ in optimum_order]
        return np.array([ordered_cluster_stage[_] for _ in predicted_labels])
        
    
    def _test_pair(self, pair_compare, n_stage_samples):
        total = np.sum(pair_compare)
#         for stage in n_stage_samples.index:
        for stage, num in n_stage_samples.items():
            stage_compare = pair_compare[stage]
            stage_total = np.sum(stage_compare)
            if (stage_total > total - stage_total) and (2 * stage_total > num):
                return stage
        return None
    
    def _calculate_probability(self, pair_compare, n_stage_samples, line):
        for stage, num in n_stage_samples.items():
            stage_compare = pair_compare[stage]
            cnt = len(stage_compare.loc[stage_compare == 1])
            line[stage] = (cnt + 1) / (num + 2)