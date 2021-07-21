from main_types.BasicMain import BasicMain
from matplotlib import cm
import joypy
import pandas as pd
from scipy.stats import norm
import tqdm
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sb
import torch

#font = {
#        'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 12
#       }

#matplotlib.rc('font', **font)
#sb.set_context('paper', rc={"font.size":14,"axes.labelsize":14, "axes.ticksize":8, 'axes.titlesize':14, 'weight': 'bold'})

sb.set_context('paper', rc={"font.size":12,"axes.labelsize":14, "axes.ticksize":4, 'axes.titlesize':12, 'fig.titlesize':9})
#sb.set_context('paper', rc={"font.size":9,"axes.labelsize":6, "axes.ticksize":3, 'axes.titlesize':6, 'fig.titlesize':6})
#plt.rcParams['font.weight'] = 'bold'
class DeltaPerStepResultsPlottingMain:
    
    def __init__(self, params):
        self.params = params
        self.setup_main = BasicMain(params)
        self.means = None

    def main(self):
        torch.random.manual_seed(self.params['random_seed'])
        np.random.seed(self.params['random_seed'])
        torch.set_default_dtype(torch.float64)

        print('currently assuming cov times is same as number of measurements, may need to update for non-covid data if processed differently')
        ### NEW ###
        self.setup_data_and_model()
        # condense releveant delta/shift info into df 
        # for easy plotting
        # currently just plotting over train data
        self.make_tr_df_to_plot()
        #self.make_example_plots()
#        self.make_split_by_age_plot()
        self.make_subdivided_box_plots()
        # uncomment to make some training plots
#        self.make_tr_plots()

        ### OLD ###
#        self.setup_data_and_model()
#        self.get_deltas()
#        self.plot_deltas()
    
    def setup_data_and_model(self):
        data_input = self.setup_main.load_data()
        self.setup_main.preprocess_data(data_input)
        model = self.setup_main.load_model()

        self.data_input = data_input
        self.dynamic_covs_order = data_input.dynamic_covs_order
        self.tr_data = data_input.get_tr_data_as_single_batch()
        self.te_data = data_input.get_te_data_as_single_batch()
        self.unnormalized_tr_data = data_input.get_unnormalized_tr_data_as_single_batch()
        self.unnormalized_te_data = data_input.get_unnormalized_te_data_as_single_batch()
        print('modifying savedir for quick fix!!!!---------------------------------------------------------------------------------------------------------------------------------- undo before doing plots with non-covid linear model!!')
        self.setup_main.params['savedir'] = '../output/covid/rayleigh/linear_delta_per_step/'
        self.savedir = os.path.join(self.setup_main.params['savedir'], 'plots')
        model_path = os.path.join(self.setup_main.params['savedir'], 'model.pkl')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        model.eval()
        self.model = model
        print(torch.exp(-self.model.global_param_logspace))
        self.global_theta = torch.exp(-self.model.global_param_logspace).cpu().detach().numpy()**(1/2)
        #print([param for param in model.parameters()])
        #print(torch.sum(model.linear.weight))


    def make_tr_df_to_plot(self): 
        self.get_deltas()
        tr_deltas = self.tr_deltas.detach().numpy()
        #static_idxs = np.array([0, -4, 1, 2]) # THESE IDXS MUST CHANGE FOR NEW SEVERITY RESULTS WITH ONE HOT ENCODINGS!!
        static_idxs = np.array([0, -25, 1, 2]) 
        static_cov_names = ['Age', 'BMI', 'Sex', 'Race']
        static_covs = self.unnormalized_tr_data.static_covs[:, static_idxs].detach().numpy()
        # for covid hosp synch data
        #cont_cov_names = ['icu', 'o2', 'sys_bp', 'dia_bp', 'temp']
        cont_cov_names = ['sys_bp', 'dia_bp', 'temp', 'LYMPHOCYTES %']
#        cont_cov_idxs = {cov:self.dynamic_covs_order.index(cov) for cov in cont_cov_names}
        # for hosp synch data
#        cont_cov_idxs = {'icu':248, 'o2':247, 'sys_bp':112, 'dia_bp':113, 'temp':114}
        cont_cov_idxs = {'LYMPHOCYTES %':72, 'sys_bp':210, 'dia_bp':211, 'temp':212}
        cont_covs = self.unnormalized_tr_data.get_unpacked_padded_cov_trajs()
        #print(cont_covs[0][0], self.tr_data.get_unpacked_padded_cov_trajs()[0][0])
        col_names = [
            'ind_idx', 'delta', 'mean_tte_rem', 
            'mean_tte', 'hazard', 'shifted_time', 'day'
        ]
        col_names = col_names + static_cov_names + cont_cov_names
        df_all = pd.DataFrame(columns=col_names)

        cov_times = self.tr_data.cov_times.detach().numpy()       
        for ind_idx in range(tr_deltas.shape[0]):
            traj_len = int(self.tr_data.traj_lens[ind_idx].detach().numpy())
            days = cov_times[ind_idx][0:traj_len]
            deltas_i = tr_deltas[ind_idx][0:traj_len]
            mean_tte_rem_i = self.get_mean_tte_remaining(deltas_i)
            hazard_i = self.get_hazards_i(deltas_i, days)
            mean_tte_i = self.get_mean_ttes(deltas_i)
            shifted_times_i = np.array([delta + day for day, delta in enumerate(deltas_i)])
            # for hosp synch data
#            days = np.arange(deltas_i.shape[0]) + 1
#            days = self.tr_data.cov_times.detach().numpy()

            age_i = np.tile(static_covs[ind_idx][0], traj_len)
            bmi_i = np.tile(static_covs[ind_idx][1], traj_len)
            sex_i = np.tile(static_covs[ind_idx][2], traj_len)
            race_i = np.tile(static_covs[ind_idx][3], traj_len)

            cont_covs_i = [
                np.array(
                    [
                        cont_covs[ind_idx][t][cont_cov_idxs[cov_name]].detach().numpy()
                        for t in range(traj_len)
                    ]
                )
                for cov_name in cont_cov_names
            ]


            ind_idx_rep = np.tile(ind_idx, traj_len)
            data_i = np.stack(
                [
                    ind_idx_rep, deltas_i, mean_tte_rem_i, mean_tte_i, hazard_i,
                    shifted_times_i, days, age_i, bmi_i, sex_i, race_i,
                ] + cont_covs_i, axis=0
            )
            data_i = np.transpose(data_i)
            df_i = pd.DataFrame(data_i, columns=col_names)
            df_all = pd.concat([df_all, df_i])
        df_all['sys_bp'] = df_all['sys_bp'].apply(lambda x: x if not x == -1 else np.nan)
        df_all['dia_bp'] = df_all['dia_bp'].apply(lambda x: x if not x == -1 else np.nan)
        # uncomment for hosp synch data
#        df_all['icu_name'] = df_all['icu'].apply(lambda x: 'In ICU' if x == 1 else 'Not in ICU')
        self.df_to_plot = df_all
        #self.df_to_plot.to_csv('plotting_data.csv')
#        with open('plot_data.pkl', 'wb') as f:
#            pickle.dump(self.df_to_plot, f)
        # uncomment for hosp synch data
#        o2_enu_to_name = self.data_input.o2_enu_to_name
#        self.df_to_plot['o2_name'] = self.df_to_plot['o2'].apply(
#            lambda x: o2_enu_to_name[x] if not x == 3 else o2_enu_to_name[0]
#        )


    def make_example_plots_old(self):
        # for hosp synch data
#        order_for_o2 = ['None (Room air)', 'ETT', 'Trached-to-vent', 'Trach mask', 'Nasal cannula']
#        self.make_seaborn_cov_plot_i(8, figsize=1, oxygen_label_order=order_for_o2)
#        self.make_seaborn_cov_plot_i(22, figsize=1, oxygen_label_order=order_for_o2)
        axis_ylimits = [[0, 0.03], [100, 155], [55, 90], [96, 102]]
        size = 1.75
        fig, axes = self.make_seaborn_cov_plot_i(5, figsize=size, axis_ylimits=axis_ylimits)
        days = [i + 1 for i in range(10)]
        mean_hazards = []
        for day in days:
            mean_hazards.append(np.mean(self.df_to_plot[self.df_to_plot['day'] == day].values))
            
        axes[0].plot(days, mean_hazards, marker='X')
        axes[0].lines[1].set_linestyle('--')
        fig.savefig('covs_with_hazard_5.png')
        plt.clf()
        fig, axes = self.make_seaborn_cov_plot_i(80, figsize=size, axis_ylimits=axis_ylimits)  
        fig.savefig('covs_with_hazard_80.png')
            
    def make_example_plots(self):
#        fig, axes = self.make_plt_cov_plot_i(79, figsize=size, axis_ylimits=axis_ylimits, n_days=8)
#        fig, axes = self.make_plt_cov_plot_i(80, figsize=size, axis_ylimits=axis_ylimits, n_days=11)  
        axis_ylimits = [[0, 0.035], [30, 110], [96, 102]]
        fig, axes = plt.subplots(3, 2, sharex=True, figsize=(5, 6))
        self.plt_cov_plot_i_single_figure_col(79, axes[:, 0], axis_ylimits=axis_ylimits, n_days=8, col_title='Patient A', suppress_axis_labels=False) 
        self.plt_cov_plot_i_single_figure_col(80, axes[:, 1], axis_ylimits=axis_ylimits, n_days=11, col_title='Patient B', suppress_axis_labels=True) 
        axes[2, 1].legend()
        fig.tight_layout()

        bbox = axes[0][0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height

        plt.savefig('testing.png')
        plt.clf()

        figsize = (2.5, 3) 
        title = 'Densities per Day Patient A'
        self.make_ridgeline_plot_i(79, n_days=8, figsize=figsize, title=title)
        title = 'Densities per Day Patient B'
        self.make_ridgeline_plot_i(80, n_days=11, figsize=figsize, title=title)

    def make_ridgeline_plot_i(self, ind_idx, n_days=8, samples_per_day=100000, figsize=(5, 3), title='name me!'):
        days = [i for i in range(n_days + 1)]
        ind_df_to_plot = self.df_to_plot[self.df_to_plot['ind_idx'] == ind_idx]
        sampled_densities = []
        days_for_df = []
        for day in days:
            delta = ind_df_to_plot[ind_df_to_plot['day'] == day]['delta'].values[0]
            sampled_densities.extend(list(self.sample_trunc_distribution(delta, day, samples_per_day)))
            days_for_df.extend([day + 1 for i in range(samples_per_day)])
        df_to_plot = pd.DataFrame({'Day': days_for_df, 'Sampled Trunc-Densities': sampled_densities})
        labels = ['Day ' + str(day) for day in days]
        fig, axes = joypy.joyplot(
            df_to_plot, by='Day', column='Sampled Trunc-Densities',
            labels=labels, range_style='own', 
            grid='y', linewidth=1, legend=False, figsize=(figsize[0], figsize[1]),
            title='',
            colormap=cm.autumn_r,
            x_range=[-1, 13],
            bw_method=0.05,
            ylabelsize=6,
            xlabelsize=6
#            kind='normalized_counts', bins=1000,
        )
        fig.suptitle(title, weight='bold', fontsize=10)
        axes[-1].set_xlabel('Days Since Hospitalization', weight='bold', fontsize=8)
        plt.savefig('ridgeline_plot%d.png' %ind_idx)

        
    def sample_trunc_distribution(self, delta, truncation_time, n_samples):
        CDF = np.random.uniform(size=n_samples)
        discriminant = \
            4 * delta**2 +\
            4 * (2 * delta * truncation_time + truncation_time**2 + \
            2*self.global_theta * np.log(1/(1 - CDF)))
        sample = (-2 * delta + discriminant**(1/2))/2

        return sample

        
#    def make_split_by_age_plot(self, day_cutoff=5):
#        means = self.df_to_plot[['ind_idx', 'mean_tte_rem', 'Age']].groupby('ind_idx').mean().reset_index()
#        means['Age'] = means['Age'].apply(
#            lambda x: '>=75' if x>=75 else '<75'
#        )
#        boxplot = sb.histplot(
#            x='mean_tte_rem', hue='Age',
#            data=means
#        )
#
#        boxplot.set_xlabel('Predicted %s' %'Mean Time to Event Remaining')        
##        boxplot.get_figure().legend(loc='lower left')
#        handles, labels = boxplot.get_legend_handles_labels()
##        boxplot.legend(handles, labels, loc='upper left', title='Age', fontsize=12, title_fontsize=12)
##        boxplot.get_legend().get_title().set_fontsize('10')
#        
#        boxplot.get_figure().tight_layout()
#        boxplot.get_figure().savefig(
#            self.get_savepath('%s_boxplot_split_by_age>75_vert.png' %'mean_tte_rem')
#        )
#        boxplot.get_figure().clf()

    def make_subdivided_box_plots(self, day_cutoff=5):
        def make_hoz_boxplot(
            col_name, hue=None, day_cutoff=day_cutoff,
            skip_days_for_plot=1
        ):
            trunc_df_to_plot = self.df_to_plot[self.df_to_plot['day'] < day_cutoff]
            boxplot = sb.boxplot(
                x=col_name, y='day', orient='h',
                data=trunc_df_to_plot, hue=hue
            )
            unique_days = np.arange(day_cutoff)
    #        delta_boxplot.get_yaxis().set(ticks=unique_days[0::skip_days_for_plot].astype(int))
            y_labels = [i if i%skip_days_for_plot == 0 else '' for i in range(day_cutoff)]
            boxplot.set_yticklabels(y_labels)
            boxplot.set_ylabel('Days Since Admission')
            return boxplot
        # Boxplot split by age
        self.df_to_plot['age_thresh_75'] = self.df_to_plot['Age'].apply(
            lambda x: '>=75' if x>=75 else '<75'
        )
        boxplot = make_hoz_boxplot(
            'hazard', hue='age_thresh_75', day_cutoff=10,
             skip_days_for_plot=1
        )
        boxplot.set_xlabel('Predicted %s' %'Hazard')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower right', title='Age', fontsize=12, title_fontsize=12)
#        boxplot.get_legend().get_title().set_fontsize('10')
        boxplot.get_figure().tight_layout()
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_age>75.png' %'hazard')
        )
        boxplot.get_figure().clf()

        ### Boxplot Split by gender
        self.df_to_plot['sex_enu'] = self.df_to_plot['Sex'].apply(
            lambda x: 'Male' if x==1 else 'Female'
        )
        boxplot = make_hoz_boxplot(
            'hazard', hue='sex_enu', day_cutoff=10,
             skip_days_for_plot=1
        )
        boxplot.set_xlabel('Predicted %s' %'Hazard')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower right', title='Sex')
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_sex.png' %'hazard')
        )
        boxplot.get_figure().clf()
        
        ### Boxplot split by obesity
        print(self.df_to_plot['BMI'].values)
        self.df_to_plot['obs_thresh_40'] = self.df_to_plot['BMI'].apply(
            lambda x: '>=40' if x>=40 else '<40'
        )
        boxplot = make_hoz_boxplot(
            'hazard', hue='obs_thresh_40', day_cutoff=20,
             skip_days_for_plot=2
        )
        boxplot.set_xlabel('Predicted %s' %'Hazard')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower left', title='BMI')
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_bmi>30.png' %'hazard')
        )
        boxplot.get_figure().clf()


    def make_split_by_age_plot(self, day_cutoff=5):
        def make_hoz_boxplot(
            col_name, hue=None, day_cutoff=day_cutoff,
            skip_days_for_plot=1
        ):
            trunc_df_to_plot = self.df_to_plot[self.df_to_plot['day'] < day_cutoff]
            boxplot = sb.boxplot(
                x=col_name, y='day', orient='h',
                data=trunc_df_to_plot, hue=hue
            )
            unique_days = np.arange(day_cutoff)
    #        delta_boxplot.get_yaxis().set(ticks=unique_days[0::skip_days_for_plot].astype(int))
            y_labels = [i if i%skip_days_for_plot == 0 else '' for i in range(day_cutoff)]
            boxplot.set_yticklabels(y_labels)
            boxplot.set_ylabel('Days Since Admission')
            return boxplot

        # Boxplot split by age
        self.df_to_plot['age_thresh_75'] = self.df_to_plot['Age'].apply(
            lambda x: '>=75' if x>=75 else '<75'
        )
        boxplot = make_hoz_boxplot(
            'mean_tte_rem', hue='age_thresh_75', day_cutoff=10,
             skip_days_for_plot=1
        )
        boxplot.set_xlabel('Predicted %s' %'Mean Time to Event Remaining')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower left', title='Age', fontsize=12, title_fontsize=12)
#        boxplot.get_legend().get_title().set_fontsize('10')
        boxplot.get_figure().tight_layout()
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_age>75.png' %'mean_tte_rem')
        )
        boxplot.get_figure().clf()



    # cutoff 40 and skip 5 for the hosp synch data
    def make_tr_plots(self, day_cutoff=24, skip_days_for_plot=3):
        def make_hoz_boxplot(
            col_name, hue=None, day_cutoff=day_cutoff,
            skip_days_for_plot=skip_days_for_plot
        ):
            trunc_df_to_plot = self.df_to_plot[self.df_to_plot['day'] < day_cutoff]
            boxplot = sb.boxplot(
                x=col_name, y='day', orient='h',
                data=trunc_df_to_plot, hue=hue
            )
            unique_days = np.arange(day_cutoff)
    #        delta_boxplot.get_yaxis().set(ticks=unique_days[0::skip_days_for_plot].astype(int))
            y_labels = [i if i%skip_days_for_plot == 0 else '' for i in range(day_cutoff)]
            boxplot.set_yticklabels(y_labels)
            boxplot.set_ylabel('Days Since Admission')
            return boxplot
        boxplot = make_hoz_boxplot('delta')
        boxplot.set_xlabel('Predicted %s' %'Deltas')        
        boxplot.get_figure().savefig(self.get_savepath('%s_boxplot.png' %'delta'))
        boxplot.get_figure().clf()

        boxplot = make_hoz_boxplot('mean_tte_rem')
        boxplot.set_xlabel('Predicted %s' %'Mean Time to Event Remaining')        
        boxplot.get_figure().savefig(self.get_savepath('%s_boxplot.png' %'mean_tte_rem'))
        boxplot.get_figure().clf()

        ### TODO:update this to use the saved mapping
        ### Boxplot Split by gender
        self.df_to_plot['sex_enu'] = self.df_to_plot['Sex'].apply(
            lambda x: 'Male' if x==1 else 'Female'
        )
        boxplot = make_hoz_boxplot(
            'mean_tte_rem', hue='sex_enu', day_cutoff=10,
             skip_days_for_plot=2
        )
        boxplot.set_xlabel('Predicted %s' %'Mean Time to Event Remaining')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower left', title='Sex')
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_sex.png' %'mean_tte_rem')
        )
        boxplot.get_figure().clf()

        ### Boxplot split by age mean tte rem
        self.df_to_plot['age_thresh_75'] = self.df_to_plot['Age'].apply(
            lambda x: '>=75' if x>=75 else '<75'
        )
        boxplot = make_hoz_boxplot(
            'mean_tte_rem', hue='age_thresh_75', day_cutoff=10,
             skip_days_for_plot=1
        )
        boxplot.set_xlabel('Predicted %s' %'Mean Time to Event Remaining')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower left', title='Age', fontsize=12, title_fontsize=12)
#        boxplot.get_legend().get_title().set_fontsize('10')
        boxplot.get_figure().tight_layout()
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_age>75.png' %'mean_tte_rem')
        )
        boxplot.get_figure().clf()

        ### Boxplot split by obesity
        self.df_to_plot['obs_thresh_30'] = self.df_to_plot['BMI'].apply(
            lambda x: '>=30' if x>=30 else '<30'
        )
        boxplot = make_hoz_boxplot(
            'mean_tte_rem', hue='obs_thresh_30', day_cutoff=20,
             skip_days_for_plot=2
        )
        boxplot.set_xlabel('Predicted %s' %'Mean Time to Event Remaining')        
#        boxplot.get_figure().legend(loc='lower left')
        handles, labels = boxplot.get_legend_handles_labels()
        boxplot.legend(handles, labels, loc='lower left', title='BMI')
        boxplot.get_figure().savefig(
            self.get_savepath('%s_boxplot_split_by_bmi>30.png' %'mean_tte_rem')
        )
        boxplot.get_figure().clf()

        ### Make covariate plots with seaborn instead of plt
        to_plot = 200
        for ind_idx in range(to_plot):
    #        self.make_seaborn_cov_plot_i(ind_idx)
            self.make_plt_cov_plot_i(ind_idx)

    def plt_cov_plot_i_single_figure_col(self, ind_idx, axes_col, oxygen_label_order=None, axis_ylimits=None, n_days=9, col_title='meow', suppress_axis_labels=True):
        days = [i for i in range(n_days + 1)]
        mean_vals = pd.read_csv('mean_vals.csv')
        mean_hazards = mean_vals['hazard'].values[0:n_days + 1]
        mean_sys_bp = mean_vals['sys_bp'].values[0:n_days + 1]
        mean_dia_bp = mean_vals['dia_bp'].values[0:n_days + 1]
        mean_temps = mean_vals['temp'].values[0:n_days + 1]

        ind_df_to_plot = self.df_to_plot[self.df_to_plot['ind_idx'] == ind_idx]
#        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5*figsize, 3*figsize))
        if axis_ylimits:
            axes_col[0].set_ylim(axis_ylimits[0][0], axis_ylimits[0][1])
            axes_col[0].set(ylim=tuple(axis_ylimits[0]))
        axes_col[0].plot(ind_df_to_plot['day'], ind_df_to_plot['hazard'], 'ro-')
#        axes_col[0].lines[0].set_linestyle('--')
        if not suppress_axis_labels:
            axes_col[0].set_ylabel('Hazard', weight='bold')
        axes_col[0].set_title(col_title, weight='bold')

        axes_col[0].plot(days, mean_hazards, 'r--')


        bp_df = ind_df_to_plot[['day', 'sys_bp', 'dia_bp']]
        axes_col[1].plot(ind_df_to_plot['day'], ind_df_to_plot['sys_bp'] - ind_df_to_plot['dia_bp'], 'bo-')
        if not suppress_axis_labels:
            axes_col[1].set_ylabel('Delta BP', weight='bold')
        if axis_ylimits:
            axes_col[1].set_ylim(axis_ylimits[1][0], axis_ylimits[1][1])

        axes_col[1].plot(days, mean_sys_bp - mean_dia_bp, 'b--')

        axes_col[2].plot(ind_df_to_plot['day'], ind_df_to_plot['temp'], 'ko-', label='Individual')
#        axes_col[2].lines[0].set_linestyle('--')
#        axes_col[2].set_title('Temperature')
        if not suppress_axis_labels:
            axes_col[2].set_ylabel('Temperature', weight='bold')
        if axis_ylimits:
            axes_col[2].set_ylim(axis_ylimits[2][0], axis_ylimits[2][1])

        axes_col[2].plot(days, mean_temps, 'k--', label='Average')
#        axes_col[2].lines[1].set_linestyle('--')

        axes_col[2].set_xlabel('Days Since Hospitalization', weight='bold')
#        axes_col[2].legend()


    def make_plt_cov_plot_i(self, ind_idx, figsize=1, oxygen_label_order=None, axis_ylimits=None, n_days=9):
        days = [i for i in range(n_days + 1)]
        mean_hazards = []
        mean_sys_bp = []
        mean_dia_bp = []
        mean_temp = []
        def get_mean_at_day(value, day):
            vals = self.df_to_plot[self.df_to_plot['day'] == day][value].values
            vals = vals[vals > 1e-10]
            return np.nanmean(vals)
        for day in days:
            mean_hazards.append(get_mean_at_day('hazard', day))
            mean_sys_bp.append(get_mean_at_day('sys_bp', day))
            mean_dia_bp.append(get_mean_at_day('dia_bp', day))
            mean_temp.append(get_mean_at_day('temp', day))
        self.means = {
            'hazard':mean_hazards, 'sys_bp':mean_sys_bp,
            'dia_bp':mean_dia_bp, 'temp':mean_temp
        }
        pd.DataFrame(self.means).to_csv('mean_vals.csv')    
        ind_df_to_plot = self.df_to_plot[self.df_to_plot['ind_idx'] == ind_idx]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(5*figsize, 3*figsize))
        if axis_ylimits:
            axes[0].set_ylim(axis_ylimits[0][0], axis_ylimits[0][1])
            axes[0].set(ylim=tuple(axis_ylimits[0]))
        axes[0].plot(ind_df_to_plot['day'], ind_df_to_plot['hazard'], marker='o')
        axes[0].lines[0].set_linestyle('--')
        axes[0].set_title('Hazard')
        axes[0].set_ylabel('')

        axes[0].plot(days, self.means['hazard'], 'ro--')
#        axes[0].lines[1].set_linestyle('--')


        bp_df = ind_df_to_plot[['day', 'sys_bp', 'dia_bp']]
        axes[1].plot(ind_df_to_plot['day'], ind_df_to_plot['sys_bp'], marker='o')
        axes[1].set_title('Systolic Blood Pressure')
        axes[1].set_ylabel('')
        axes[1].lines[0].set_linestyle('--')
        if axis_ylimits:
            axes[1].set_ylim(axis_ylimits[1][0], axis_ylimits[1][1])

        axes[1].plot(days, self.means['sys_bp'], 'ro--')
#        axes[1].lines[1].set_linestyle('--')
        axes[2].plot(ind_df_to_plot['day'], ind_df_to_plot['dia_bp'], marker='o')

        axes[2].set_title('Diastolic Blood Pressure')
        axes[2].set_ylabel('')
        axes[2].set_xlabel('')
        if axis_ylimits:
            axes[2].set_ylim(axis_ylimits[2][0], axis_ylimits[2][1])

        axes[2].plot(days, self.means['dia_bp'], 'ro')
        axes[2].lines[0].set_linestyle('--')
        axes[2].lines[1].set_linestyle('--')


        axes[3].plot(ind_df_to_plot['day'], ind_df_to_plot['temp'], marker='o', label='Individual')
        axes[3].lines[0].set_linestyle('--')
        axes[3].set_title('Temperature')
        axes[3].set_ylabel('')
        if axis_ylimits:
            axes[3].set_ylim(axis_ylimits[3][0], axis_ylimits[3][1])

        axes[3].plot(days, self.means['temp'], 'ro--', label='Average')
#        axes[3].lines[1].set_linestyle('--')

        axes[3].set_xlabel('Days Since Covid-19 Diagnosis')
        axes[3].legend()

        fig.tight_layout()
        plt.savefig(self.get_savepath('covs_with_hazard_%d' %(ind_idx)))
        return fig, axes

    # TODO update this plot to have the oxygen labels be constant across both of the versions
    # and also make the fonts larger so the images can be shrunk more
    def make_seaborn_cov_plot_i(self, ind_idx, figsize=1, oxygen_label_order=None, axis_ylimits=None):
        ind_df_to_plot = self.df_to_plot[self.df_to_plot['ind_idx'] == ind_idx]
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(4*figsize, 4*figsize))
        if axis_ylimits:
            axes[0].set_ylim(axis_ylimits[0][0], axis_ylimits[0][1])
            axes[0].set(ylim=tuple(axis_ylimits[0]))
        # toggle comments to switch between mean_tte_rem and hazards
        #sb.lineplot(ax=axes[0], x='day', y='mean_tte_rem', data=ind_df_to_plot, marker='o')
        plot = sb.lineplot(ax=axes[0], x='day', y='hazard', data=ind_df_to_plot, marker='o')
        axes[0].lines[0].set_linestyle('--')
        #axes[0].set_title('Mean Time to Event Remaining')
        axes[0].set_title('Hazard')
        axes[0].set_ylabel('')
        mean_hazards = []
        days = [i +1 for i in range(9)]
        for day in days:
            mean_hazards.append(np.mean(self.df_to_plot[self.df_to_plot['day'] == day].values))
            
        plot.plot(days, mean_hazards, marker='X')
        axes[0].lines[1].set_linestyle('--')
#        if axis_ylimits:
#            axes[0].set_ylim(axis_ylimits[0][0], axis_ylimits[0][1])
        # uncomment for hosp synch data       
#        plot = sb.scatterplot(ax=axes[1], x='day', y='icu_name', data=ind_df_to_plot, hue='icu_name')
#        plot = sb.scatterplot(ax=axes[1], x='day', y='LYMPHOCYTES %', data=ind_df_to_plot)

#        data_lymph = ind_df_to_plot[ind_df_to_plot['LYMPHOCYTES %'] > 0]
#        sb.scatterplot(ax=axes[1], x='day', y='LYMPHOCYTES %', data=data_lymph, marker='o')
#        axes[1].set_title('Lymphocytes %')
#        axes[1].set_ylabel('')
        #plot.legend().set_visible(False)

        bp_df = ind_df_to_plot[['day', 'sys_bp', 'dia_bp']]
        sb.lineplot(ax=axes[1], x='day', y='sys_bp', data=bp_df, marker='o')
        axes[1].set_title('Systolic Blood Pressure')
        axes[1].set_ylabel('')
        axes[1].lines[0].set_linestyle('--')
        if axis_ylimits:
            axes[1].set_ylim(axis_ylimits[1][0], axis_ylimits[1][1])

        sb.lineplot(ax=axes[2], x='day', y='dia_bp', data=bp_df, marker='o')
#        sb.lineplot(ax=axes[2], x='day', y='dia_bp', data=bp_df, marker='o', label='diastolic_blood_pressure')
#        sb.lineplot(ax=axes[2], x='day',  data=ind_df_to_plot, marker='o')
        #uncomment for hosp synch data
#        sb.stripplot(x='day', y='o2_name', 
#            order=oxygen_label_order, data=ind_df_to_plot, ax=axes[2],
#            jitter=False
#        )

#        plot = sb.scatterplot(ax=axes[2], x='day', y='o2_name', data=ind_df_to_plot, hue='o2_name', hue_order=oxygen_label_order)
        axes[2].set_title('Diastolic Blood Pressure')
        axes[2].set_ylabel('')
        axes[2].set_xlabel('')
        axes[2].lines[0].set_linestyle('--')
        if axis_ylimits:
            axes[2].set_ylim(axis_ylimits[2][0], axis_ylimits[2][1])
#        plot.legend().set_visible(False)

        sb.lineplot(ax=axes[3], x='day', y='temp', data=ind_df_to_plot, marker='o')
        axes[3].lines[0].set_linestyle('--')
        axes[3].set_title('Temperature')
        axes[3].set_ylabel('')
        if axis_ylimits:
            axes[3].set_ylim(axis_ylimits[3][0], axis_ylimits[3][1])

#        axes[3].set_xlabel('Days Since Hospitalization')
        axes[3].set_xlabel('Days Since Covid-19 Diagnosis')

#        sb.lineplot(ax=axes[4], x='day', y='sys_bp', data=ind_df_to_plot, marker='x')
#        sb.lineplot(ax=axes[4], x='day', y='dia_bp', data=ind_df_to_plot, marker='x')
#        axes[4].lines[0].set_linestyle('--')
#        axes[4].lines[1].set_linestyle('--')
#        axes[4].set_ylabel('Blood Pressure')
        fig.tight_layout()
        plt.savefig(self.get_savepath('covs_with_mean_tte_rem_%d' %(ind_idx)))
        return fig, axes
    def get_savepath(self, filename):
        savepath = os.path.join(self.savedir, filename)
        return savepath

    # note used in both old and new so dont delete
    def get_deltas(self):
        tr_deltas, _, _ = self.model(self.tr_data)
        self.tr_deltas = torch.squeeze(tr_deltas)
        te_deltas, _, _ = self.model(self.te_data)
        self.te_deltas = torch.squeeze(te_deltas)

    def get_mean_tte_remaining(self, deltas):
        mean_ttes = self.get_mean_ttes(deltas)
        cov_times = np.arange(len(deltas))
        return (mean_ttes - cov_times)
    '''
        deltas: traj_length list of tensor deltas for single individual or numpy array
        returns: mean tte from the appropiately truncated and renormalized distribution
    '''
    def get_mean_ttes(self, deltas):
        # add split statement here per model type 
        dist_type = self.setup_main.params['train_params']['loss_params']['distribution_type']
        if dist_type == 'rayleigh':
            mean_ttes = self.get_mean_ttes_rayleigh(deltas)
        else:
            raise NotImplementedError('Dist type %s not implemented for plots yet' %dist_type)
        return mean_ttes
        
    def get_mean_ttes_rayleigh(self, deltas):
        scale = torch.exp(-self.model.global_param_logspace).detach().numpy()
        cov_times = np.arange(len(deltas))
        if not (type(deltas[0]) is np.float64):
            deltas = np.array([d.detach().numpy() for d in deltas])
        survival = np.exp(-(cov_times + deltas)**2/(2*scale))
#        print(cov_times + deltas)
#        print(survival)
#        print((cov_times + deltas)/scale**(1/2))
#        print(
#            (1 - norm.cdf((cov_times + deltas)/scale**(1/2)))
#        )
        mean_ttes = \
            cov_times + \
            1./(survival) * (2 * np.pi * scale)**(1/2) * \
            (1 - norm.cdf((cov_times + deltas)/scale**(1/2)))
        return mean_ttes 

    def get_hazards_i(self, deltas_i, days):
        scale = torch.exp(-self.model.global_param_logspace).detach().numpy()
#        cov_times = np.arange(len(deltas_i))
#        print('assuming plotting training data in get_hazards_i')
#        cov_times = self.tr_data.cov_times.detach().numpy()
        hazards_i = (days + deltas_i ) * (1./scale)
        return hazards_i

#    def get_hazards(self, deltas_all):
#        # hazard computed at the current time
#        scale = torch.exp(-self.model.global_param_logspace).detach().numpy()

#    def get_hazard_single_time(self, deltas_t, time, scale):
#        hazards_all = []
#        for t, deltas_t in enumerate(deltas):
#            hazards = 1./(scale) * (t + deltas_t)
#            hazards_all.append(hazards)
#        return hazards_all


    '''
        deltas: a list of numpy arrays with all the deltas per timestep (note that each array is
            not neccesarily the same shape
        returns: list of mean_tte_remaining per day/timestep
    '''
        
    def get_mean_ttes_remaining_aggregate_deltas(self, deltas, remaining=True):
        mean_ttes_all = []
        for t, deltas_t in enumerate(deltas):
            if remaining:
                mean_ttes_t, _ = self.get_mean_ttes_remaining_agg_deltas_single_time(deltas_t, t)
            else:
                _, mean_ttes_t = self.get_mean_ttes_remaining_agg_deltas_single_time(deltas_t, t)
                
#            mean_ttes_all.append([tte - t for tte in mean_ttes_t])
            mean_ttes_all.append(mean_ttes_t)
        
        return mean_ttes_all

    def get_mean_ttes_remaining_agg_deltas_single_time(self, deltas, time):
        # add split statement here per model type 
        dist_type = self.setup_main.params['train_params']['loss_params']['distribution_type']
        if dist_type == 'rayleigh':
            mean_ttes = self.get_mean_ttes_agg_rayleigh(deltas, time)
        else:
            raise NotImplementedError('Dist type %s not implemented for plots yet' %dist_type)
        return mean_ttes - time, mean_ttes
        
    def get_mean_ttes_agg_rayleigh(self, deltas, time):
        scale = torch.exp(-self.model.global_param_logspace).detach().numpy()
        survival = np.exp(-(time + deltas)**2/(2*scale))
#        print(cov_times + deltas)
#        print(survival)
#        print((cov_times + deltas)/scale**(1/2))
#        print(
#            (1 - norm.cdf((cov_times + deltas)/scale**(1/2)))
#        )
        mean_ttes = \
            time + \
            1./(survival) * (2 * np.pi * scale)**(1/2) * \
            (1 - norm.cdf((time + deltas)/scale**(1/2)))
        return mean_ttes 

    def plot_deltas(self, num_meas_times=50, figsize=12):
        timesteps = [i for i in range(num_meas_times)]
        labels = ['%d' %i for i in timesteps]
        deltas_and_most_recent_idxs = [
            self.get_most_recent_deltas(self.te_deltas, self.te_data, time)
            for time in timesteps
        ]
        most_recent_deltas = [tpl[0] for tpl in deltas_and_most_recent_idxs]
        most_recent_idxs = [tpl[1] for tpl in deltas_and_most_recent_idxs]

        bool_idxs_to_keep_per_time = [
            np.where(
                most_recent_idxs[time] == time,
                np.ones(most_recent_idxs[time].shape[0]).astype(bool),
                np.zeros(most_recent_idxs[time].shape[0]).astype(bool)
            )
            for time in timesteps
        ]
        deltas_te = [
            most_recent_deltas[time][bool_idxs_to_keep_per_time[time]]
            for time in timesteps
        ]
        plt.boxplot(deltas_te, labels=labels)
        savepath = os.path.join(self.savedir, 'deltas_boxplots_te.png')
        plt.savefig(savepath)
        plt.clf()
    
        plt.boxplot(deltas_te[0])
        savepath = os.path.join(self.savedir, 'deltas_boxplot_te_t=0.png')
        plt.savefig(savepath)
        plt.clf()
    
        # TR plot
        deltas_and_most_recent_idxs = [
            self.get_most_recent_deltas(self.tr_deltas, self.tr_data, time)
            for time in timesteps
        ]
        most_recent_deltas = [tpl[0] for tpl in deltas_and_most_recent_idxs]
        most_recent_idxs = [tpl[1] for tpl in deltas_and_most_recent_idxs]

        bool_idxs_to_keep_per_time = [
            np.where(
                most_recent_idxs[time] == time,
                np.ones(most_recent_idxs[time].shape[0]).astype(bool),
                np.zeros(most_recent_idxs[time].shape[0]).astype(bool)
            )
            for time in timesteps
        ]
        deltas_tr = [
            most_recent_deltas[time][bool_idxs_to_keep_per_time[time]]
            for time in timesteps
        ]
#        ind_idxs_per_timestep = [
#            np.arange(self.tr_data.shape[0])[bool_idxs_to_keep_per_time[time]]
#            for time in timesteps
#        ]
        fig, axes = plt.subplots(2, 1, figsize=(figsize,figsize * 2), sharex=True)
        axes[0].boxplot(deltas_tr, labels=labels, vert=False)
        x, y = self.get_distribution_to_plot()
        axes[1].plot(x, y)
        savepath = os.path.join(self.savedir, 'deltas_boxplots_tr.png')
        plt.savefig(savepath)
        plt.clf()
    
        fig, axes = plt.subplots(2, 1, figsize=(figsize,figsize * 2), sharex=True)
        mean_ttes_remaining = self.get_mean_ttes_remaining_aggregate_deltas(deltas_tr)
        axes[0].boxplot(mean_ttes_remaining, labels=labels, vert=False)
        x, y = self.get_distribution_to_plot()
        axes[1].plot(x, y)
        savepath = os.path.join(self.savedir, 'mean_ttes_remaining_boxplots_tr.png')
        plt.savefig(savepath)
        plt.clf()

        fig, axes = plt.subplots(2, 1, figsize=(figsize * 2,figsize), sharex=True)
        mean_ttes_remaining = self.get_mean_ttes_remaining_aggregate_deltas(deltas_tr)
        axes[0].boxplot(mean_ttes_remaining, labels=labels, vert=True)
        axes[0].set_ylabel('Mean Predicted Time Remaining Until Death')
        x, y = self.get_distribution_to_plot(plot_over=len(deltas_tr))
        axes[1].plot(x, y, label='Global Distribution')
        axes[1].set_xlabel('Days Since Hospital Admission')
        axes[1].legend()
        savepath = os.path.join(self.savedir, 'mean_ttes_remaining_boxplots_tr_vert.png')
        plt.savefig(savepath)
        plt.clf()

        ### Global dist with avg shifted times per day as lines on the plot
        x, y = self.get_distribution_to_plot()
        plt.plot(x, y, label='Global Distribution')
        days_to_plot = 50
        shifted_times_per_day = []
        for t, deltas_t in enumerate(deltas_tr[0:days_to_plot]):
            shifted_time_avg_t =  np.median(deltas_t) + t
            shifted_times_per_day.append(shifted_time_avg_t)
            plt.plot([shifted_time_avg_t, shifted_time_avg_t], [0, .04], '-', label='Day%d' %(t + 1))
        plt.legend()
        savepath = os.path.join(self.savedir, 'median_shifted_times_with_global_distribution.png')
        plt.savefig(savepath)
        plt.clf()

        ### Global dist with avg shifted times per day binned by shifted time and avg computed
        total_days = len(shifted_times_per_day)
        x, y = self.get_distribution_to_plot()
        plt.plot(x, y, label='Global Distribution')
        bin_size = 5
        num_iter = total_days//5
        days = np.arange(total_days) + 1
        for i in range(num_iter):
            start = i * bin_size
            end = (i + 1) * bin_size
            shifts_to_avg = shifted_times_per_day[start:end]
            avg_shift = np.mean(shifts_to_avg)
            days_avg  = np.mean(days[start:end])

            plt.plot([avg_shift, avg_shift], [0, .04], '-', label='Avg Day%.1f' %(days_avg))
         
        plt.legend()
        savepath = os.path.join(self.savedir, 'binned_avg_median_shifted_times_with_global_distribution.png')
        plt.savefig(savepath)
        plt.clf()
            
        ### Horizontal mean ttes boxpolot time remaining?
        fig, axes = plt.subplots(2, 1, figsize=(figsize,figsize * 2), sharex=True)
        mean_ttes_remaining = self.get_mean_ttes_remaining_aggregate_deltas(deltas_tr, remaining=False)
        axes[0].boxplot(mean_ttes_remaining, labels=labels, vert=False)
        x, y = self.get_distribution_to_plot()
        axes[1].plot(x, y)
        savepath = os.path.join(self.savedir, 'mean_ttes_boxplots_tr.png')
        plt.savefig(savepath)
        plt.clf()

        plt.boxplot(deltas_tr[0])
        savepath = os.path.join(self.savedir, 'deltas_boxplot_tr_t=0.png')
        plt.savefig(savepath)
        plt.clf()

        ### Single individual plot for person 35 at times 0, 18, and 30
        ind_idx = 35
        deltas_per_time, idx_per_time = \
            self.get_most_recent_deltas_and_idxs_multiple_start_times_single_ind(
                ind_idx, self.tr_deltas, start_times
            )
        to_plots = []
        to_plots.append((1, deltas_per_time[0] + 1))
        to_plots.append((18, deltas_per_time[17] + 18))
        to_plots.append((30, deltas_per_time[29] + 30))
        fig, axes = plt.subplots(1, 3)
        for to_plot in to_plots:
            self.plot_trunc_dist(axes[p], to_plots[0], to_plots[1])

        savepath = os.path.join(self.savedir, 
            'shifted_times_on_global_dist_individual%d.png' %ind_idx
        )
        plt.savefig(savepath)
        plt.clf()
    
        def plot_trunc_dist(self, axis, trunc_time, shifted_time):
            pass 

#        deltas_tr_one_individual = [d[10] for d in deltas_tr]
#        plt.plot(np.arange(len(deltas_tr_one_individual)), deltas_tr_one_individual)
#        savepath = os.path.join(self.savedir, 'ex_individual0_deltas_boxplots_tr.png')
#        plt.savefig(savepath)
#        plt.clf()
        # Plot with covariates of an example individual
        # currently plotting the same number of individuals as timesteps set
        for ind_idx in tqdm.tqdm(range(len(deltas_tr))):
            self.plot_individual_deltas_with_covs(ind_idx, timesteps)

    def get_most_recent_deltas(self, deltas, batch, start_time):
        max_times_less_than_start, idxs_max_times_less_than_start = \
            batch.get_most_recent_times_and_idxs_before_start(start_time)
 
        idxs_deltas = \
            [
                torch.arange(0, idxs_max_times_less_than_start.shape[0]),
                idxs_max_times_less_than_start
            ]
        deltas_at_most_recent_time = deltas[idxs_deltas].squeeze(-1)
        return deltas_at_most_recent_time.detach().numpy(), idxs_max_times_less_than_start.detach().numpy()
    
         
        
    def get_distribution_to_plot(self, plot_over=None):
        if plot_over:
            x = np.linspace(0, plot_over, 1000)
        else:
            x = np.linspace(0, 100, 1000)
        dist_type = self.params['train_params']['loss_params']['distribution_type']
        global_param = torch.exp(-self.model.global_param_logspace).detach().numpy()
        if dist_type == 'rayleigh':
            ret = x/global_param * (np.exp(-x**2/(2*global_param)))
        else:
            raise NotImplementedError('distribution %s type not implemented in plotting' %(dist_type))
        return x, ret

    def plot_mean_tte_rem_with_covs_ind(self, ind_idx):
        pass
#        grid = sb.FacetGrid(   

    def plot_individual_deltas_with_covs(self, ind_idx, start_times, size=4):
        fig, axes = plt.subplots(5, 1, sharex=True, figsize=(size*4, size*5))
        #ind_idx = 15
        #ind_idx = 100
        #print([(len(d), d[0]) for d in deltas_tr])
        batch_covs = self.tr_data.get_unpacked_padded_cov_trajs()
        traj_len = int(self.tr_data.traj_lens[ind_idx].detach().numpy())
        deltas_per_time, idx_per_time = \
            self.get_most_recent_deltas_and_idxs_multiple_start_times_single_ind(
                ind_idx, self.tr_deltas, start_times
            )
        deltas_per_time = deltas_per_time[0:traj_len] #throwaway padding
        mean_tte_remaining_per_time = self.get_mean_tte_remaining(deltas_per_time)
        eff_traj_len = len(deltas_per_time)
        
         
#        axes[0].plot(np.arange(eff_traj_len), deltas_per_time)
        axes[0].plot(np.arange(eff_traj_len), mean_tte_remaining_per_time, 'x--')
        axes[0].set_ylabel('Mean Time to Death')
        icu_idx = 248
        o2_idx = 247
        sys_bp_idx = 112
        dia_bp_idx = 113
        temp_idx = 114
#        print( 
#            [   str(meas[icu_idx].detach().numpy())
#                for meas in batch_covs[ind_idx][0:eff_traj_len]
#            ]
#        )
        to_plot = \
            [   str(meas[icu_idx].detach().numpy())
                for meas in batch_covs[ind_idx][0:eff_traj_len]
            ]
        to_plot = to_plot if len(set(to_plot)) > 1 else [int('%d' %float(entry)) for entry in to_plot]
        axes[1].scatter(
            np.arange(eff_traj_len), to_plot
        )
        axes[1].set_ylabel('ICU Status')

        o2_enu_to_name = self.data_input.o2_enu_to_name
        to_plot_str = \
            [   o2_enu_to_name[int(meas[o2_idx].detach().numpy())][0:5]
                if not meas[o2_idx].detach().numpy() == 3 else o2_enu_to_name[0]
                for meas in batch_covs[ind_idx][0:eff_traj_len]
            ]
        to_plot_int = \
            [   int(meas[o2_idx].detach().numpy())
                if not meas[o2_idx].detach().numpy() == 3 else 0
                for meas in batch_covs[ind_idx][0:eff_traj_len]
            ]
        to_plot = to_plot_str if len(set(to_plot_str)) > 1 else to_plot_int
        axes[2].scatter(
            np.arange(eff_traj_len), to_plot
        )
        axes[2].set_ylabel('Oxygen type')
        axes[3].plot(
            np.arange(eff_traj_len),
            [   meas[temp_idx].detach().numpy() 
                for meas in batch_covs[ind_idx][0:eff_traj_len]
            ], 'x--'
        )
        axes[3].set_ylabel('Temperature')
        axes[4].plot(
            np.arange(eff_traj_len),
            [   meas[sys_bp_idx].detach().numpy() 
                for meas in batch_covs[ind_idx][0:eff_traj_len]
            ], 'x--', label='Sys BP'
        )
        axes[4].plot(
            np.arange(eff_traj_len),
            [   meas[dia_bp_idx].detach().numpy() 
                for meas in batch_covs[ind_idx][0:eff_traj_len]
            ], 'x--', label='Dia BP'
        )
        axes[4].legend() 
        axes[4].set_xlabel('Time in Days Since Hospital Admission')

        age = self.tr_data.static_covs[ind_idx][0]
        bmi = self.tr_data.static_covs[ind_idx][-4]
        sex = self.tr_data.static_covs[ind_idx][1]
        race = self.tr_data.static_covs[ind_idx][2]
        death = \
            'Died day %d' %self.tr_data.event_times[ind_idx]\
            if not self.tr_data.censoring_indicators[ind_idx]\
            else 'Left hospital on day %d' %self.tr_data.event_times[ind_idx]
        plt.text(
            0.0, 1.0,
            'Age: %d\nBMI: %d\nSex: %d\nRace: %d\nStatus:%s' %(age, bmi, sex, race, death),
            ha='left', va='top', transform=fig.transFigure
        )

        fig.tight_layout(pad=1.25)
        savepath = os.path.join(self.savedir, 'ex_individual%d_deltas_boxplots_tr.png' %ind_idx)
        plt.savefig(savepath)
        plt.clf()

    def get_most_recent_deltas_and_idxs_multiple_start_times_single_ind(
        self, ind_idx, all_deltas, start_times
    ):
        idxs_most_recent_times = []
        deltas = []
        for s, start_time in enumerate(start_times):
            if start_time == 0:
                idx_most_recent_time = 0
            else:
                bool_idxs_less_than_start = self.tr_data.cov_times[ind_idx] <= start_time
                truncated_at_start = torch.where(
                    bool_idxs_less_than_start,
                    self.tr_data.cov_times[ind_idx],
                    torch.zeros(self.tr_data.cov_times[ind_idx].shape)
                )
#                print(bool_idxs_less_than_start, truncated_at_start)
#                print(truncated_at_start.shape, torch.max(truncated_at_start))
                idx_most_recent_time = torch.max(truncated_at_start).int()
                # handle edge cases where torch.max picks the last zero
                # instead of the first when t_ij = 0
                if torch.sum(truncated_at_start) == 0:
                    idx_most_recent_time = 0
                
#                idx_most_recent_time = torch.where(
#                    torch.sum(truncated_at_start) == 0,
#                    0, #torch.zeros(idxs_most_recent_times.shape, dtype=torch.int64),
#                    idx_most_recent_time
#                )

            delta = all_deltas[ind_idx][
                    idx_most_recent_time
            ]
            deltas.append(delta)
            idxs_most_recent_times.append(idx_most_recent_time)        
#            most_recent_times = self.cov_times[ind_idx][
#                torch.arange(idxs_most_recent_times.shape[0]),
#                idxs_most_recent_times
#            ]
    
        return deltas, idxs_most_recent_times
