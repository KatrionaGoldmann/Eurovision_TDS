# This program takes as inputs:
#   A set of mcmc samples from a model fit
#   A scipy scaler object to scale the new covariates
#   A set of new covariates
#   A lookup table mapping voter-performer pairs to an index

# Multiple full competitions are simulated, using different sets of model parameters
# taken from the file of mcmc samples.


import sys
from pathlib import Path
import arviz as az
import numpy as np
from numpy.random import default_rng
import pandas as pd
import pickle
import json
import stan
import matplotlib.pyplot as plt
import datetime

class Competition():
    def __init__(self,n_runs,comp_structure,fitted_model_output,X_performers,X_scaler,vp_idx_dict):
        self.n_runs = n_runs
        self.comp_structure = comp_structure
        self.fitted_model_output = fitted_model_output
        self.X_performers = X_performers
        self.X_scaler = X_scaler
        self.vp_idx_dict = vp_idx_dict

        self.rng = default_rng()

        self.__load_fitted_model()
        self.__sample_model_params()
        self.__load_data()

    def run(self):
        print('Predicting votes for semi-finals...')
        self.__predict_sf_votes()
        print('Ranking semi-finalists and determining finalists...')
        self.__get_finalists()
        print('Predicting votes for finals...')
        self.__predict_f_votes()
        print('Ranking finalists...')
        results = []
        for run in range(self.n_runs):
            results.append(self.__rank_on_votes(self.f_votes[run]))
        self.results = results
        print('Saving results...')
        self.__save_results()
        print('Simulation complete.')
    
    def __save_results(self):
        results_dict = {"f_ranked_total_points" : self.results,
                        "sf1_ranked_total_points" : self.sf1_ranked,
                        "sf2_ranked_total_points" : self.sf2_ranked}
        now = datetime.datetime.now()
        model_name = self.fitted_model_output.split('/').pop()
        with open(f'eurovision/simulation_results/{now} -- {model_name}_{self.n_runs}_sim_runs.pkl','wb') as f:
            pickle.dump(results_dict, f)
    
    def print_stats(self):
        # explore winners
        winners = []
        for run in range(self.n_runs):
            # take winning country name (results = [[(2charcode, sum_votes), ...more_countries...], ...more_sim_runs...])
            winners.append(self.results[run][0][0])
        print(f'Competition winners ranked by win rate across {self.n_runs} simulations:')
        self.__print_occurances(winners)
        
        # explore 'top-5' acheivement
        top5 = []
        for run in range(self.n_runs):
            top5.extend(list(map(lambda x:x[0], self.results[run][:5])))
        print(f'Top-5 appearances ranked by frequency across {self.n_runs} simulations:')
        self.__print_occurances(top5)

    def make_plots_for_run(self,run):
        # plot vote distribution in sf1, sf2 and f
        fig,ax = plt.subplots(ncols=3, nrows=1)
        ax[0].set_title(f'sf1 votes run#{run}')
        ax[0].hist(list(map(lambda x:x[2],self.sf_votes[run]['sf1'])), bins=100)
        ax[1].set_title(f'sf2 votes run#{run}')
        ax[1].hist(list(map(lambda x:x[2],self.sf_votes[run]['sf2'])), bins=100)
        ax[2].set_title(f'final votes run#{run}')
        ax[2].hist(list(map(lambda x:x[2],self.f_votes[run])), bins=100)
        fig.set_size_inches(12,8)
        fig.savefig(f'eurovision/simulation_results/votes_dist_run_{run}.png')

        # show sum votes in sf1 and sf2
        fig,ax = plt.subplots(ncols=2, nrows=1)
        ax[0].set_title(f'sf1 sum votes run#{run}')
        sf1_data = list(zip(*self.sf1_ranked[run]))
        ax[0].bar(sf1_data[0],sf1_data[1])
        ax[1].set_title(f'sf2 sum votes run#{run}')
        sf2_data = list(zip(*self.sf2_ranked[run]))
        ax[1].bar(sf2_data[0],sf2_data[1])
        fig.set_size_inches(12,8)
        fig.savefig(f'eurovision/simulation_results/sf_sum_votes_dist_run_{run}.png')

        # plot vote sumations in final
        fig,ax = plt.subplots()
        ax.set_title(f'final sum votes run#{run}')
        f_data = list(zip(*self.results[run]))
        ax.bar(f_data[0],f_data[1])
        fig.set_size_inches(12,8)
        fig.savefig(f'eurovision/simulation_results/f_votes_dist_run_{run}.png')
        
        
    def __load_fitted_model(self):
        self.az_fit = az.from_json(self.fitted_model_output)
    
    def __sample_model_params(self):
        # sample n_runs sets of model params for independant competition simulation runs
        n_chains,n_draws = self.az_fit.sample_stats.chain.shape[0],self.az_fit.sample_stats.draw.shape[0]

        if self.n_runs == n_chains*n_draws:
            print('Using all mcmc samples - no random sampling...')
            # then one sim for each mcmc sample without random sampling
            rnd_idxs = []
            for c in range(n_chains):
                for d in range(n_draws):
                    rnd_idxs.append((c,d))
            self.rnd_idxs = rnd_idxs
        else:
            self.rnd_idxs = list(zip(self.rng.integers(low=0, high=(n_chains), size=self.n_runs),
                             self.rng.integers(low=0, high=(n_draws), size=self.n_runs)))
        
    def __load_data(self):
        df_X = pd.read_csv(self.X_performers)
        # Given gender is a categoric variable with 3 classes, encode as binary w.r.t default gender='group'
        df_X['male'] = [1 if gender=='male' else 0 for gender in df_X['gender']]
        df_X['female'] = [1 if gender=='female' else 0 for gender in df_X['gender']]

        # Evaluate binary variables for boolean covariates to be used
        df_X['Contains_English_bin'] = df_X['Contains_English'].apply(lambda x: 1 if x else 0)
        df_X['Contains_Own_Language_bin'] = df_X['Contains_Own_Language'].apply(lambda x: 1 if x else 0)
        self.df_X = df_X
    
        with open(self.X_scaler, 'rb') as f:
            self.X_scaler = pickle.load(f)

        with open(self.vp_idx_dict) as f:
            vptoi = f.read()
        self.vptoi = json.loads(vptoi)
   
    def __predict_sf_votes(self):
        sf_votes = []
        for run in range(n_runs):
            sf_votes.append({ 'sf1': None, 'sf2': None })
            for sf_id in ['sf1','sf2']:
                # sf voters are countries performing in that sf + several seeded voters
                performers = self.df_X.loc[ self.df_X['comp_round'] == sf_id ]['to_code2'].unique()
                voters = np.append(performers, self.comp_structure[f'{sf_id}_v_seeded'])
                sf_votes[run][sf_id] = self.__predict_votes(voters,performers,run)
        self.sf_votes = sf_votes

    def __predict_f_votes(self):
        f_votes = []
        voters = self.comp_structure['f_v']
        for run in range(n_runs):
            performers = self.finalists[run]
            f_votes.append(self.__predict_votes(voters,performers,run))
        self.f_votes = f_votes

    def __predict_votes(self,voters,performers,run_idx):
        df_p = self.df_X.loc[ self.df_X['to_code2'].isin(performers) ]
        #  drop empty from_code2 column before cross merging in voters
        df_p = df_p.drop(['from_code2','from_code3','from_country'], axis=1)
        # expand df_sf to introduce a row for each vote (note: self-voting is not allowed)
        df_p = pd.merge(df_p, pd.DataFrame({"from_code2" : voters}), how='cross')
        # remove self-voting rows
        df_p = df_p.drop(df_p.loc[ df_p['to_code2'] == df_p['from_code2'] ].index)
        # scale covariates with scaler fit during model training
        xbeta = df_p.loc[:,['Contains_English_bin','Contains_Own_Language_bin','male','female','comps_without_win']].values
        xbeta_norm = self.X_scaler.transform(xbeta)
        # evaluate the vp indicies that correspond to the voter-performer pairs in the test data
        df_p['vp'] = df_p.apply(lambda x: self.vptoi[f'{x["from_code2"]}-{x["to_code2"]}'], axis=1)
        # define stan code for inference only
        model = """
        // overload add function for adding an int to an array of ints
        functions {
          array[] int add(array[] int x, int y) {
              int x_size = size(x);
              array[x_size] int z;
              for (i in 1:x_size){
                z[i] = x[i] + y;
              }
              return z;
          }
        }
        data {
          int<lower=2> S;
          int<lower=0> N;   // total number of performances
          int<lower=1> B;   // number of performance dependent covariates
          int<lower=1> VP;  // number of voter/performer combinations
          vector[B] beta;
          vector[VP] alpha;
          ordered[S-1] lambda;
          real gamma;
          matrix[N,B] xbeta;       // performance dependent covariates
          array[N] int<lower=0,upper=VP-1> vp;  // voter/performer pair index
        }
        generated quantities {
          // predictions (scores we expect to observe for new data)
          vector[N] y_pred;
          for (n in 1:N) {
            y_pred[n] = ordered_logistic_rng( gamma + alpha[ add(vp[n],1) ] + (xbeta[n] * beta), lambda);
          }
        }
        """

        data = {
            "S" : 11,
            "N" : xbeta_norm.shape[0],
            "B" : xbeta_norm.shape[1],
            "VP" : len(self.vptoi),
            "gamma" : self.az_fit.posterior.gamma[self.rnd_idxs[run_idx]].item(),
            "beta" : self.az_fit.posterior.beta[self.rnd_idxs[run_idx]].values,
            "alpha" : self.az_fit.posterior.alpha[self.rnd_idxs[run_idx]].values,
            "lambda" : self.az_fit.posterior['lambda'][self.rnd_idxs[run_idx]].values,
            "xbeta" : xbeta_norm,
            "vp" : df_p['vp'].values
        }
        posterior = stan.build(model, data=data)
        fit = posterior.fixed_param(num_chains=1, num_samples=1)
        az_fit = az.from_pystan(
            posterior=fit, 
            predictions="y_pred",
            posterior_model=posterior)

        # decode y_pred (1->11) into votes (0->12)
        def format_votes(x):
          if x == 10.:
            return 12.
          elif x == 9.:
            return 10.
          return x
        # single chain and draw must be indexed into
        df_p['y_pred'] = az_fit.predictions.y_pred[0][0]
        df_p['points'] = (df_p['y_pred'] - 1.).apply(format_votes)
        return list(zip(df_p['from_code2'].values,df_p['to_code2'].values,df_p['points'].values))

    def __rank_on_votes(self, votes):
        votes_sum = {}
        for v,p,score in votes:
            votes_sum[p] = votes_sum.get(p,0) + score
        ranked = sorted(votes_sum.items(), key=lambda x:x[1], reverse=True)
        return list(ranked)
    
    def __get_finalists(self):
        finalists = []
        sf1_ranked = []
        sf2_ranked = []
        for run in range(self.n_runs):
            sf1_ranked.append(self.__rank_on_votes(self.sf_votes[run]['sf1']))
            sf2_ranked.append(self.__rank_on_votes(self.sf_votes[run]['sf2']))
            qualified = list(map(lambda x:x[0], sf1_ranked[run][:10] + sf2_ranked[run][:10]))
            finalists.append(qualified + self.comp_structure["f_p_seeded"])
        self.sf1_ranked = sf1_ranked
        self.sf2_ranked = sf2_ranked
        self.finalists = finalists

    def __print_occurances(self,occurances):
        occurs = {}
        for country in occurances:
            occurs[country] = ((occurs.get(country, (0,None))[0] + 1), None)
        # establish occurance rates
        for country,(count,event_rate) in occurs.items():
            occurs[country] = (count,(count / self.n_runs))
        
        self.sorted_occurs = sorted(occurs.items(), key=lambda x:x[1][0], reverse=True )
        for i,(w,(count,event_rate)) in enumerate(self.sorted_occurs):
            print(f'#{i+1} {w} {100*event_rate}%')
            
def print_syntax():
    print('\nSyntax: 2023_comp_simulation.py <n-runs> <fitted-model-output> <X-performers> <X-scaler> <vp-idx-dict>')
    print()
    print('\tRun 2023 competition simulation.\n')
    print('\t<n-runs>               : An iteger setting the number of simulations to run.')
    print('\t<fitted-model-output>  : .json export of an arviz.InferenceData object containing output from an mcmc model fit.')
    print('\t<X-performers>         : .csv file containing covariate data for the performers in the competition.')
    print('\t<X-scaler>             : .pkl file containing an sklearn Scaler object that was fit on the model covariated during training.')
    print('\t<vp-idx-dict>          : .json file containing a dict mapping from 2-char code voter-performer pairs to indexes (as used during model training).')
    print()
    print('Example usage:')
    print('\t2023_comp_simulation.py model_3_voter_bias_1000_samples_4_chains.json df_2023_covariates.csv scaler_model_3_voter_bias_1000_samples_4_chains.json vptoi.json')

if __name__ == "__main__":
    comp_structure = {
        "f_p_seeded": ['FR','DE','IT','ES','UA','GB'],
        "sf1_v_seeded": ['FR','DE','IT'],
        "sf2_v_seeded": ['ES','UA','GB'],
        "f_v": ['AL','AU','AM','AT','AZ','BE','HR','CY','CZ','DK','EE','FI','FR','GE','DE','GR','IS','IE','IL','IT','LV','LT','MT','MD','NL','NO','PL','PT','RO','SM','RS','SI','ES','SE','CH','UA','GB'],
    }
    if len(sys.argv) != 6:
        print_syntax()
        exit()
    n_runs = int(sys.argv[1])
    fitted_model_output = sys.argv[2]
    X_performers = sys.argv[3]
    X_scaler = sys.argv[4]
    vp_idx_dict = sys.argv[5]
    
    comp = Competition(n_runs=n_runs,
                       comp_structure=comp_structure,
                       fitted_model_output=fitted_model_output,
                       X_performers=X_performers,
                       X_scaler=X_scaler,
                       vp_idx_dict=vp_idx_dict)

    comp.run()
    comp.print_stats()
    # comp.make_plots_for_run(run=0)

