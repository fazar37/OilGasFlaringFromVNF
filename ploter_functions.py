import matplotlib.pyplot as plt
import matplotlib.style as mst
import matplotlib.patches as mpatch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np
import pandas as pd
from itertools import product
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import warnings
mst.use('seaborn-v0_8-paper')

class PloterClass:
    """
    The PloterClass class provides the functions to plot the data and outputs.
    Usually, pbt and vnf objects are key inputs for fuctions, representing ProbClass and VNF/VNF_filter classes.
    """
    def __init__(self,):
        warnings.simplefilter("ignore", UserWarning)
        self.start_years_list:list = ['1 year', '2 years', '3 years', '4 years']
        self.start_year:int = 2
        self.start_status:list = ['Start-up', 'Regular Operation']
        self.meanprops:dict = dict(linestyle='--', linewidth=1.5, color='red')
        self.medianprops:dict = dict(linestyle='-', linewidth=1.5, color='orange')
        self.boxprops:dict = dict(linestyle='-', linewidth=1.2, edgecolor='blue')
        self.boxprops2:dict = dict(linestyle='-', linewidth=1.2, color='blue')
        self.capprops:dict = dict(linestyle='-', linewidth=1, color='purple')
        self.whiskerprops:dict = self.capprops
        self.legend_els:list = [Line2D([0], [0], linestyle='--', linewidth=2, color='red', label='mean'),
                           Line2D([0], [0], linestyle='-',  linewidth=2, color='orange',    label='median'),
                           Line2D([0], [0], linestyle='-',  linewidth=2, color='blue',  label='(25%-75%) caps'),
                           Line2D([0], [0], linestyle='-',  linewidth=2, color='purple',   label='(5%-95%) whiskers'),]
        self.start_label_legend :list = [mpatch.Patch(color='turquoise', label=self.start_status[0]), mpatch.Patch(color='greenyellow', label=self.start_status[1])]
        self.legend_text:str = 'Significance codes: 0.005,***; 0.05,**; 0.1,*; no significant,^'

    def plot_selected_facilities_for_comparison(self, vnf, n_col:int=3, border:float=0.02, figbase=1):
        """
        The plot_selected_facilities_for_comparison plots bar plot for total annual number of flaring days and line plots of volume of gas
        flared per capcity for selected facilities.
        It first, categorizes the facilities, make them available to have similar y-axis.
        Then, it creates GridSpecFromSubplotSpec to track the facilities with similar VNF or GAS. Finally, it plots the data on each
        subplots
        Input:
            vnf: VNF() class
            n_col:int = to be used number of columns in GridSpec [default = 4]
            border:float = to  be used with y labels [default = 0.07]
            figbase:float = to be used with setting the figure size = (figbase*4, figbase*5) [default=3.4]
        Output:
            Figure: png format
        """
        self.check_class(vnf, 'vnf', 'VNF')
        def ghr_fun(i) -> int:
            pp = (len(i) // n_col) * 2 if len(i) % n_col == 0 else (len(i) // n_col + 1)*2
            if pp == 2: return 1
            return pp

        def get_y_lim(df) -> tuple:
            if type(df) is list: return (pd.concat(df).min().min(), pd.concat(df).max().max())
            return (df.min().min(), df.max().max())

        def plot_bar(x, y, ax, ylims, color:str='tab:blue', width=0.6, im_=None):
            ax.bar(x, y, color=color, tick_label=x, width=width)
            if im_: ax.set_ylim(0, ylims[1]+30)
            else: ax.set_ylim(0, ylims[1]+10) 
            plt.xlim(x.min(), x.max())
            ax.tick_params(axis='y', colors=color)

        def plot_line(x, y, ax, ylims, color:str='tab:red', sty='--', in_=True):
            ax.plot(x, y, sty, color=color)
            if in_: ax.set_ylim(0, ylims[1]+2)
            else: ax.set_ylim(0, ylims[1]+0.2)
            ax.tick_params(axis='y', colors=color)
        
        def set_x_ticks(iL:int, len_list:int, ax, check=True):
            if check:
                if iL >= len_list-n_col:
                    ax.tick_params(axis='x', rotation=90)
                    ax.set_xlabel('Year'); return
            ax.tick_params(labelbottom=False)

        def title_FAC(id_:str) -> str:
                if id_ == 'MLNG, DM LNG, SM LNG, TM LNG': return 'Group Malaysia'
                if id_ == 'AP LNG, Gladstone, QC LNG': return 'Group Australia'
                if id_ == 'Vysotsk LNG, PO LNG': return 'Group Russia'
                return id_

        index_sorted_by_year = pd.Index(['Marsa LNG', 'Pluto LNG', 'PNG LNG', 'NLNG', 'Wheatstone LNG', 'Gorgon LNG', 'Cameroon FLNG', 'DS LNG'], name='FAC_NAME')
        gas_per_cap = (vnf.gas / vnf.capacity * 100).loc[index_sorted_by_year]
        wb_per_cap = (vnf.wb / vnf.capacity * 100).loc[index_sorted_by_year]
        daily_flares = vnf.vnf.loc[index_sorted_by_year]
        sts = vnf.status.loc[index_sorted_by_year]

        index_above_81  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, np.logical_and(gas_per_cap.max(axis=1) <= 12, gas_per_cap.max(axis=1) > 5))].index
        index_above_82  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, np.logical_and(gas_per_cap.max(axis=1) <= 5, gas_per_cap.max(axis=1) > 1))].index
        index_above_83  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, gas_per_cap.max(axis=1) < 1)].index
        index_above_40  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 40, daily_flares.max(axis=1) < 80)].index
        
        index_lists = [index_above_81, index_above_82, index_above_83, index_above_40]
        height_ratios: list = [ghr_fun(index_list) for index_list in index_lists]
        
        fig = plt.figure(figsize=(7, 3.7)); gs0 = GridSpec(len(index_lists), 1, hspace=1.1, figure=fig, height_ratios=height_ratios)
        gs_all = [GridSpecFromSubplotSpec(max(1, ghr_fun(index_list) // 2), n_col, wspace=0.51, hspace=2, subplot_spec=gs0[i]) for i, index_list in enumerate(index_lists)]

        for gsI, index_list in zip(gs_all, index_lists):
            ylims_vnf = get_y_lim(daily_flares.loc[index_list])
            ylims_gas = get_y_lim([wb_per_cap.loc[index_list], gas_per_cap.loc[index_list]])#; ylims_gas = (ylims_gas[0], min(13, ylims_gas[1]))
            cj = True if (index_list.equals(index_lists[-1]) or index_list.equals(index_lists[-2])) else False
            im_ = True if (index_list.equals(index_lists[0]) or index_list.equals(index_lists[1]) or index_list.equals(index_lists[2])) else False
            in_ = True if (index_list.equals(index_lists[0]) or index_list.equals(index_lists[1])) else False
            for ic_, id_ in enumerate(index_list):
                ax = fig.add_subplot(gsI[ic_ // n_col, ic_ % n_col]); ax2 = ax.twinx()
                plt.title(f'{title_FAC(id_)}\n[{sts.at[id_, "COUNTRY"]}] {sts.at[id_, "StartYear"]}')
                plot_bar(np.asarray(vnf.years, dtype=int), daily_flares.loc[id_], ax, ylims_vnf, im_=im_)
                plot_line(np.asarray(vnf.years, dtype=int), gas_per_cap.loc[id_], ax2, ylims_gas, sty='--', in_=in_)
                plot_line(np.asarray(vnf.years, dtype=int), wb_per_cap.loc[id_], ax2, ylims_gas, sty='-', in_=in_)
                plt.xlim(np.asarray(vnf.years, dtype=int).min()-1, np.asarray(vnf.years, dtype=int).max()+1) 
                if index_list.equals(index_lists[-1]): set_x_ticks(ic_, len(index_list), ax, check=cj)
                elif ic_ == 2 and index_list.equals(index_lists[-2]): set_x_ticks(ic_, len(index_list), ax, check=cj)
                else: set_x_ticks(ic_, len(index_list), ax, check=False)
        
        fig.text(border, 0.5, 'Total Number of Flaring Days per Year\n[#]', transform=fig.transFigure, va='center', ha='center', color='tab:blue', rotation=90) #, fontsize=12)
        fig.text(1-border, 0.5, 'Annual Volume of Gas Flared per Capacity\n(bcm / bcm) [%]', transform=fig.transFigure, va='center', ha='center', color='tab:red', rotation=-90)#, fontsize=12)

        handles =  [Line2D([0], [0], linestyle='--', linewidth=2, color='tab:red', label='VNF Volume of Gas Flared Per Capacity'), \
                    Line2D([0], [0], linestyle='-', linewidth=2, color='tab:red', label='WB Volume of Gas Flared Per Capacity'), \
                    mpatch.Patch(color='tab:blue', label='Total Number of Flaring Days per Year')]
        fig.legend(handles=handles, fancybox=True, ncol=1, bbox_to_anchor=(0.65, 0.87), loc='center', fontsize=9.8)
        plt.savefig('./Figures/General/AnnualFlareCount_vs_VolumeOfGasFlaredPerCapacity_Selected.png', bbox_inches='tight')
        plt.close()
        return

    def plot_all_facilities_for_comparison(self, vnf, n_col:int=4, border:float=0.02, figbase:float=2, sc_:int = 3):
        """
        The plot_all_facilities_for_comparison plots bar plot for total annual number of flaring days and line plots of volume of gas
        flared per capcity for each facility.
        It first, categorize the facilities, make them available to have similar y-axis.
        Then, it creates GridSpecFromSubplotSpec to track the facilities with similar VNF or GAS. Finally, it plots the data on each
        subplots
        Input:
            vnf: VNF() class
            n_col:int = to be used number of columns in GridSpec [default = 4]
            border:float = to  be used with y labels [default = 0.07]
            figbase:float = to be used with setting the figure size = (figbase*4, figbase*5) [default=3.4]
        Output:
            Figure: png format
        """
        self.check_class(vnf, 'vnf', 'VNF')
        
        def ghr_fun(i) -> int:
            pp = (len(i) // n_col) * sc_ if len(i) % n_col == 0 else (len(i) // n_col + 1) * sc_
            if pp == sc_: return 1
            return pp

        def get_y_lim(df) -> tuple:
            if type(df) is list: return (pd.concat(df).min().min(), pd.concat(df).max().max())
            return (df.min().min(), df.max().max())

        def plot_bar(x, y, ax, ylims, color:str='tab:blue', width=0.6, im_=None):
            ax.bar(x, y, color=color, tick_label=x, width=width)
            ax.set_ylim(0, ylims[1]+10)
            plt.xlim(x.min(), x.max())
            ax.tick_params(axis='y', colors=color)

        def plot_line(x, y, ax, ylims, color:str='tab:red', sty='--'):
            ax.plot(x, y, sty, color=color)
            ax.set_ylim(ylims[0], ylims[1])
            ax.tick_params(axis='y', colors=color)
        
        def set_x_ticks(iL:int, len_list:int, ax, check=True):
            if check:
                if  iL >= len_list-n_col:
                    ax.tick_params(axis='x', rotation=90)
                    ax.set_xlabel('Year'); return
            ax.tick_params(labelbottom=False)

        def title_FAC(id_:str) -> str:
                if id_ == 'MLNG, DM LNG, SM LNG, TM LNG': return 'Group Malaysia'
                if id_ == 'AP LNG, Gladstone, QC LNG': return 'Group Australia'
                if id_ == 'Vysotsk LNG, PO LNG': return 'Group Russia'
                return id_

        index_sorted_by_year = vnf.status.sort_values(by='StartYear').index.drop(['Calcasieu Pass', 'CS FLNG'])
        gas_per_cap = (vnf.gas / vnf.capacity * 100).loc[index_sorted_by_year]
        wb_per_cap = (vnf.wb / vnf.capacity * 100).loc[index_sorted_by_year]
        daily_flares = vnf.vnf.loc[index_sorted_by_year]
        sts = vnf.status.loc[index_sorted_by_year]
        
        index_above_80  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, gas_per_cap.max(axis=1) > 12)].index
        index_above_81  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, np.logical_and(gas_per_cap.max(axis=1) <= 12, gas_per_cap.max(axis=1) > 5))].index
        index_above_82  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, np.logical_and(gas_per_cap.max(axis=1) <= 5, gas_per_cap.max(axis=1) > 1))].index
        index_above_83  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 80, gas_per_cap.max(axis=1) < 1)].index
        index_above_40  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 40, daily_flares.max(axis=1) < 80)].index
        index_above_15  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 15, daily_flares.max(axis=1) < 40)].index
        index_above_05  = sts.loc[np.logical_and(daily_flares.max(axis=1) >= 5, daily_flares.max(axis=1) < 15)].index
        
        index_lists = [index_above_83, index_above_40, index_above_15, index_above_05] #[index_above_80, index_above_81, index_above_82] #, index_above_83, index_above_40, index_above_15, index_above_05] #, index_below_05]
        height_ratios: list = [ghr_fun(index_list) for index_list in index_lists]
        fig = plt.figure(figsize=(figbase*n_col, figbase*4)); gs0 = GridSpec(len(index_lists), 1, hspace=0.4, figure=fig, height_ratios=height_ratios)
        gs_all = [GridSpecFromSubplotSpec(max(1, ghr_fun(index_list) // sc_), n_col, wspace=0.8, hspace=2, subplot_spec=gs0[i]) for i, index_list in enumerate(index_lists)]

        for gsI, index_list in zip(gs_all, index_lists):
            ylims_vnf = get_y_lim(daily_flares.loc[index_list])
            ylims_gas = get_y_lim([wb_per_cap.loc[index_list], gas_per_cap.loc[index_list]])
            cj = True if (index_list.equals(index_lists[-1]) or index_list.equals(index_lists[-2])) else False
            im_ = True if (index_list.equals(index_lists[0]) or index_list.equals(index_lists[1]) or index_list.equals(index_lists[2])) else False
            for ic_, id_ in enumerate(index_list):
                ax = fig.add_subplot(gsI[ic_ // n_col, ic_ % n_col]); ax2 = ax.twinx()
                plt.title(f'{title_FAC(id_)}\n[{sts.at[id_, "COUNTRY"]}] {sts.at[id_, "StartYear"]}')
                plot_bar(np.asarray(vnf.years, dtype=int), daily_flares.loc[id_], ax, ylims_vnf, im_=im_)
                plot_line(np.asarray(vnf.years, dtype=int), gas_per_cap.loc[id_], ax2, ylims_gas, sty='--')
                plot_line(np.asarray(vnf.years, dtype=int), wb_per_cap.loc[id_], ax2, ylims_gas, sty='-')
                plt.xlim(np.asarray(vnf.years, dtype=int).min()-1, np.asarray(vnf.years, dtype=int).max()+1)
                if index_list.equals(index_lists[-1]): set_x_ticks(ic_, len(index_list), ax, check=True)
                # elif ic_ in [1, 2] and index_list.equals(index_lists[-2]): set_x_ticks(ic_, len(index_list), ax, check=cj)
                else: set_x_ticks(ic_, len(index_list), ax, check=False)
        
        fig.text(border, 0.5, 'Total Number of Flaring Days per Year [#]', transform=fig.transFigure, va='center', color='tab:blue', rotation=90) #, fontsize=12)
        fig.text(1-border, 0.5, 'Annual Volume of Gas Flared per Capacity (bcm / bcm) [%]', transform=fig.transFigure, va='center', color='tab:red', rotation=-90)#, fontsize=12)

        handles =  [Line2D([0], [0], linestyle='--', linewidth=2, color='tab:red', label='VNF Volume of Gas Flared Per Capacity'), \
                    Line2D([0], [0], linestyle='-', linewidth=2, color='tab:red', label='WB Volume of Gas Flared Per Capacity'), \
                    mpatch.Patch(color='tab:blue', label='Total Number of Flaring Days per Year')]
        # fig.legend(handles=handles, fancybox=True, ncol=1, bbox_to_anchor=(0.65, 0.88), loc='center')
        fig.legend(handles=handles, fancybox=True, ncol=2, bbox_to_anchor=(0.5, 0), loc='center')
        plt.savefig('./Figures/General/AnnualFlareCount_vs_VolumeOfGasFlaredPerCapacity.png', bbox_inches='tight')
        plt.close()
        return

    def plot_compare_two_scenarios(self, vnfs:list, pbt=None, test:str='ttest', xticks=None, twoLegend:bool=True):
        """
        plot_compare_two_scenarios function creates a figure of boxplot for paper and analysis of volume of gas flared, annual number of flaring
        days and number of consecutive days with flaring to compare different vnf_filter scenarios.
        Input:
            vnfs: list = [vnf1, vnf2, ...]
            vnf1: vnf_filter_obj class = VNF filtered for scenario 1
            vnf2: vnf_filter_obj class = VNF filtered for scenario 2
            ...
            pbt= ProbClass class object
            test: str = one of the 'ttest', 'ks_2samp', 'mannwhitneyu'
        """
        for i_, vnf in enumerate(vnfs):
            self.check_class(vnf, f'vnf {i_}', 'VNF')
        self.check_class(pbt, 'pbt', 'ProbClass')
        if not type(vnfs) is list and not type(xtick) is list and len(vnfs) != len(xtick): raise ValueError("plot_compare_two_scenarios function requires the list VNF objects as the Input. length vnfs and xtcks must be the same.")

        grouped = pd.concat([pbt.merge_categories([pbt.rm_nan_fl(vnf.gas_filter_start.values), \
                                                    pbt.rm_nan_fl(vnf.vnf_filter_start.values), \
                                                    pbt.rm_nan_fl(vnf.flareEpisodes_filter_start['length'])], vnf_filter_name=xtick) \
                                                        for vnf, xtick in zip(vnfs, xticks)], ignore_index=True).groupby(['Category'])
        
        fig = plt.figure(figsize=(12, 3)); gs = GridSpec(1, 3, wspace=0.4)
        for i, (categ, label, title) in enumerate(zip(pbt.static_table_columns, pbt.static_table_units, pbt.static_table_title)):
            df = grouped.get_group((categ,))
            ax = fig.add_subplot(gs[i]); plt.title(title)
            self.boxplot_by_sns(df, ax=ax, x_name='vnf_filter_name', y_name='values', \
                                ylabel=label, fill=False, rowXlabel=True, colYlabel=True, xlabel='')
            plt.xticks(rotation=30, ha='right'); plt.grid(axis='y', which='major')
            self.plot_classify_pvalue(df, pbt=pbt, categs=xticks, ax=ax)
        
        sname = '_'.join(f'{vnf.sname}' for vnf in vnfs)
        if twoLegend:
            self.legend_box(fig=fig, bbox_to_anchor=(0.3, -0.3), start_legend=False)
            self.legend_significant(fig=fig, xloc=0.7, yloc=-0.3)
        else:
            self.legend_box(fig=fig, bbox_to_anchor=(0.5, -0.3), start_legend=False)
    
        plt.savefig(f'./Figures/Output/Compare{sname}_{test}.png', bbox_inches='tight'); plt.close()
        return

    def plot_classify_pvalue(self, df, pbt:None, categs=None, ax=None):
        """
        If there are two boxplots in a subplot, it plots the significance 
        """
        if len(categs) > 2: return
        a, b = df.loc[df['vnf_filter_name'] == categs[0], 'values'].values, df.loc[df['vnf_filter_name'] == categs[1], 'values'].values
        plt.text(0.5, 0.8, f'{pbt.classify_pvalue(pbt.significant_test(a, b))}', transform=ax.transAxes)
        return

    def check_class(self, class_obj, name:str, nameClass:str):
        if not hasattr(class_obj, "__init__"): raise ValueError(f'{name} is not the {nameClass} object')
        return
        
    def plot_compare_all(self, vnf=None, pbt=None, ylims_:list=[12, 300, 7]):
        """
        Create a GridSpec of 3 (rows) by 2 (columns) to plot box plot for all faiclities in one figure.
        """
        self.check_class(vnf, 'vnf', 'VNF')
        self.check_class(pbt, 'pbt', 'ProbClass')
        if hasattr(vnf, 'gas_filter') and hasattr(vnf, 'sname'):
            df_lists:list = [vnf.gas_filter.stack(future_stack=True).reset_index(name='values'), vnf.vnf_filter.stack(future_stack=True).reset_index(name='values'), vnf.flareEpisodes_filter.reset_index()]
            sname:str = f'DistributionOf{vnf.sname}'; order = vnf.status_filter.index
        else:
            df_lists:list = [(vnf.gas / vnf.capacity * 100).stack(future_stack=True).reset_index(name='values'), vnf.vnf.stack(future_stack=True).reset_index(name='values'), vnf.flareEpisodes.reset_index()]
            sname:str = f'CompareAll'; order = vnf.status.index
        if 'Kiyanly LNG' in order:
            df_lists = [df.loc[df['FAC_NAME'] != 'Kiyanly LNG'] for df in df_lists]
            order = order.drop(['Kiyanly LNG'])

        def setting_ylim2(i):
            if i == 0: plt.ylim(0, ylims_[0])
            elif i == 1: plt.ylim(0, ylims_[1])
            else: plt.ylim(0, ylims_[2])
    
        fig = plt.figure(figsize=(12, 5)); gs = GridSpec(3, 2, width_ratios=[15, 1], wspace=0.2, hspace=0.3)
        for i, (df, val_col) in enumerate(zip(df_lists, ['values', 'values', 'length'])):
            ax = fig.add_subplot(gs[i, 0]); plt.title(pbt.static_table_title[i])
            rowXlabel = True if i == 2 else False; ylim_ = df[val_col].max() #ylim_ = np.percentile(df[val_col].values, 97)
            self.boxplot_by_sns(df, ax=ax, x_name='FAC_NAME', y_name=val_col, ylabel=pbt.static_table_units[i], fill=False, \
                                rowXlabel=rowXlabel, colYlabel=True, xlabel='', order=order)
            plt.xticks(rotation=90); plt.grid(axis='y', lw=0.6, linestyle='--'); plt.ylim(0, ylim_)#setting_ylim2(i)

            ax = fig.add_subplot(gs[i, 1]); xlabel = 'All Together' if i == 2 else ''
            self.boxplot_by_sns(df, ax=ax, x_name=None, y_name=val_col, ylabel=pbt.static_table_units[i], fill=False, rowXlabel=rowXlabel, colYlabel=False, xlabel=xlabel)
            plt.grid(axis='y', lw=0.6, linestyle='--'); plt.ylim(0, ylim_) #setting_ylim2(i)

        self.legend_box(fig=fig, bbox_to_anchor=(0.5, -0.3), start_legend=False)
        fig.savefig(f'./Figures/Output/{sname}.png', bbox_inches='tight'); plt.close()

    def ygrid(self,):
        plt.grid(axis='y', lw=0.6, linestyle='--')

    def xgrid(self,):
        plt.grid(axis='x', lw=0.4, linestyle='--')

    def plot_start_up_priod_box_plot(self, vnfs:list, pbt=None, test:str='ttest', static_locs:list=[2, 100, 3]):
        """
        plot_start_up_priod_box_plot function creates a figure of boxplot for paper and analysis of volume of gas flared, annual number of flaring
        days and number of consecutive days with flaring during startup periods.
        Input:
            vnfs: list = [vnf1, vnf2, vnf3, vnf4]
            vnf1: vnf_filter_obj class = VNF filtered for 1 year  start-up
            vnf2: vnf_filter_obj class = VNF filtered for 2 years start-up
            vnf3: vnf_filter_obj class = VNF filtered for 3 years start-up
            vnf4: vnf_filter_obj class = VNF filtered for 4 years start-up
            pbt: ProcClass class object
            test: str = one of the 'ttest', 'ks_2samp', 'mannwhitneyu'
        """
        for i_, vnf in enumerate(vnfs):
            self.check_class(vnf, f'vnf {i_}', 'VNF')
        self.check_class(pbt, 'pbt', 'ProbClass')
        if not type(vnfs) is list and len(vnfs) != 4: raise ValueError("prep_df_start function requires the list of four years startup VNF objects as the Input.")
        
        for i in vnfs:
            vnf.vnf_filter_start = vnf.vnf_filter_start.where(vnf.vnf_filter_start > 0, np.nan)
            vnf.gas_filter_start = vnf.gas_filter_start.where(vnf.gas_filter_start > 0, np.nan)

        fig = plt.figure(figsize=(10, 5)); gs = GridSpec(3, 1, hspace=0.4)
        plt.suptitle('Analysis by Number of Year(s) Considered as Start-up', fontweight='bold', fontsize=14)
        ax = fig.add_subplot(gs[0]); plt.title('Annual Volume of Gas Flared per Capacity')
        df_gas, self.static_ttest = pbt.prep_df_start(dfs=[vnf.gas_filter_start for vnf in vnfs], from_pivot=False, test=test)
        self.boxplot_by_sns(df_gas, ax=ax, ylabel='(bcm/bcm) [%]', hue_name='OperatingTime')#; plt.ylim(0, 3)
        self.text_significant(y_loc=static_locs[0]); self.ygrid()

        ax = fig.add_subplot(gs[1]); plt.title('Number of Flaring Days per Year')
        df_vnf, self.static_ttest = pbt.prep_df_start(dfs=[vnf.vnf_filter_start for vnf in vnfs], from_pivot=False, test=test)
        self.boxplot_by_sns(df_vnf, ax=ax, ylabel='[# days]', hue_name='OperatingTime')
        self.text_significant(y_loc=static_locs[1]); self.ygrid()

        ax = fig.add_subplot(gs[2]); plt.title('Number of Consecutive Days with Flaring')
        df_len, self.static_ttest = pbt.prep_df_start(dfs=[vnf.flareEpisodes_filter_start for vnf in vnfs], from_pivot=True, test=test)
        self.boxplot_by_sns(df_len, ax=ax, ylabel='[# of consecutive days]', rowXlabel=True, hue_name='OperatingTime')
        self.text_significant(y_loc=static_locs[2]); plt.ylim(0, 15); self.ygrid()

        self.legend_box(fig=fig, bbox_to_anchor=(0.5, 0), start_legend=True)
        plt.savefig(f'./Figures/Output/startup_analysis_{vnfs[0].sname.split("_")[0]}_{test}.png', bbox_inches='tight'); plt.close()        
        return

    def text_significant(self, y_loc=1):
        """
        plot signifcance code on startup figures at the given year(s)
        """
        for x, key in self.static_ttest.items():
            plt.text(x, y_loc, key, va='center', ha='center')
    
    def legend_box(self, fig=None, bbox_to_anchor:tuple=(0.5, -0.05), start_legend:bool=False):
        """
        legend_box function create the legend object in a figure for the give bbox_to_anchor input
        Input:
            bbox_to_anchor:tuple = (0.5, -0.1) [default]
        """
        if fig is None: raise ValueError("legend_box function requires fig object as the input")
        fig.legend(handles=self.legend_els, ncols=4, loc='center', fancybox=True, bbox_to_anchor=bbox_to_anchor)
        if start_legend:
            fig.legend(handles=self.start_label_legend, ncols=2, loc='center', fancybox=True, bbox_to_anchor=(0.3, bbox_to_anchor[1]-0.05))
            self.legend_significant(fig=fig, xloc=0.7, yloc=bbox_to_anchor[1]-0.05)

    def legend_significant(self, fig=None, xloc:float=0, yloc:float=0):
        """
        Plot a legend text of self.legend_text with a fancy box on the given location of xloc and yloc
        xloc and yloc (floats) are relative to the fig axis.
        """
        fig.text(xloc, yloc, self.legend_text, transform=fig.transFigure, ha='center', va='center', \
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", pad=0.3, alpha=0.3), fontsize=8)

    def boxplot_by_sns(self, dfI:pd.DataFrame, ax=None, x_name:str='Year', y_name:str='values', hue_name=None, color_list:list=['turquoise', 'greenyellow'], log_scale:bool=False, \
                             hue_order=None, legend:bool=False, showmeans:bool=True, whis:tuple=(5,95), showfliers:bool=False, meanline:bool=True, \
                             colYlabel:bool=True, ylabel:str='', rowXlabel:bool=False, xlabel:str='', gap=0.1, fill:bool=True, order=None):
        """
        plot box plot by seaborn
        Check seaborn documentation for the potential usage.
        """
        boxprops = self.boxprops if fill is True else self.boxprops2
        
        if hue_name is None:
            sns.boxplot(data=dfI, x=x_name, y=y_name, fill=fill, log_scale=log_scale, legend=legend, order=order, \
                        meanprops=self.meanprops, medianprops=self.medianprops, boxprops=boxprops, capprops=self.capprops, whiskerprops=self.whiskerprops, \
                        showmeans=showmeans, whis=whis, showfliers=showfliers, meanline=meanline, gap=gap)

        else:
            if hue_order is None: hue_order = self.start_status
            sns.boxplot(data=dfI, x=x_name, y=y_name, hue=hue_name, hue_order=hue_order, palette=color_list, fill=fill, log_scale=log_scale, legend=legend, \
                            meanprops=self.meanprops, medianprops=self.medianprops, boxprops=self.boxprops, capprops=self.capprops, whiskerprops=self.whiskerprops, \
                            showmeans=showmeans, whis=whis, showfliers=showfliers, meanline=meanline, gap=gap, order=order)
        
        if rowXlabel: plt.xlabel(xlabel)
        else: plt.tick_params(labelbottom=False); plt.xlabel(xlabel)

        if colYlabel: plt.ylabel(ylabel)
        else: plt.tick_params(labelleft=False); plt.ylabel(ylabel)
    
    def spatial_sensitity_radius(self, vnf_big, vnf_small, sname=None, facilities=['Peru LNG', 'Sabine Pass LNG', 'ADNOC LNG'], figbase=3):
        """
        plot the location of box plots
        Check seaborn documentation for the potential usage.
        """
        def get_box_size(lon, lat, extra:float=0.1) -> list:
            lat_min, lat_max, lon_min, lon_max = lat.min(), lat.max(), lon.min(), lon.max()
            lat_c, lon_c, lat_d, lon_d = (lat_min+lat_max)/2, (lon_min+lon_max)/2, (lat_max-lat_min)/2, (lon_max-lon_min)/2
            bounds = np.round(max(lat_d, lon_d) + extra, 1)
            return [lon_c-bounds, lon_c+bounds, lat_c-bounds, lat_c+bounds]

        if sname is None: raise ValueError("sname is empty")
        self.check_class(vnf_big, 'vnf', 'VNF class'); self.check_class(vnf_small, 'vnf', 'VNF class')
        fig = plt.figure(figsize=[figbase*len(facilities), figbase]); gs = GridSpec(1, len(facilities), wspace=0.6)
        for id_, fac in enumerate(facilities):
            dfB = vnf_big.clustered_flares_df.loc[vnf_big.clustered_flares_df['FAC_NAME'] == fac]
            dfS = vnf_small.clustered_flares_df.loc[vnf_small.clustered_flares_df['FAC_NAME'] == fac]
            ax = fig.add_subplot(gs[id_], projection=ccrs.PlateCarree())
            sns.scatterplot(data=dfB, x='Lon_GMTCO', y='Lat_GMTCO', c='tab:red', s=8, transform=ccrs.PlateCarree())
            sns.scatterplot(data=dfS, x='Lon_GMTCO', y='Lat_GMTCO', c='tab:blue', s=13, transform=ccrs.PlateCarree())
            plt.title(fac); ax.add_feature(cf.COASTLINE, linewidth=0.4); ax.add_feature(cf.BORDERS, linewidth=0.2)
            ax.set_extent(get_box_size(dfB['Lon_GMTCO'], dfB['Lat_GMTCO']), crs=ccrs.PlateCarree())
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    linewidth=1, color='black', alpha=0.35, linestyle='--', dms=True, x_inline=False, y_inline=False)
            gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
            gl.left_labels, gl.top_labels, gl.bottom_labels, gl.right_labels = True, False, True, False
            ax.add_feature(cf.OCEAN, facecolor=("lightblue"))

        leg_elm = [Line2D([0], [0], markersize=10, marker='o', markerfacecolor='tab:red', linestyle='', label='Buffer Size = 1500 meters'), \
                   Line2D([0], [0], markersize=10, marker='o', markerfacecolor='tab:blue', linestyle='', label='Buffer Size = 750 meters')]
        fig.legend(handles=leg_elm, bbox_to_anchor=(0.5, 0.1), ncol=2, loc='center', fancybox=True)
        plt.savefig(sname, bbox_inches='tight'); plt.close()
    
    def get_cmap_colors(self, cmap_name:str='rainbow', num_categories=10):
        """
        Extract `num_categories` evenly spaced colors from the specified colormap.
        
        Parameters:
            cmap_name (str): Name of the matplotlib colormap (e.g., 'viridis', 'tab10', 'plasma').
            num_categories (int): Number of distinct colors needed.
            
        Returns:
            List of RGBA tuples.
        """
        cmap = plt.get_cmap(cmap_name, num_categories)
        return [cmap(i) for i in range(num_categories)]

    def barplot_with_bottom(self, dfI, width=0.5, x_index=None, facility_index=None, colors=None):
        bottom = np.zeros(len(x_index))
        for i, name in enumerate(facility_index):
            values_to_add = dfI.loc[name].values
            values_to_add[pd.isna(values_to_add)] = 0
            plt.bar(x_index, values_to_add, width, label=name, bottom=bottom, color=colors[i])
            bottom += values_to_add

    def plot_compare_vnf_wb(self, vnf=None, width=0.5):

        """
        Plot the stacked bar plot of volume of gas flared per year for both VNF and World Bank datasets for sake of comparison.
        It creates two figures in FIgure directory:
            1. volume of gas flared with each color representing each facility
            2. volume of gas flared with each color representing each country
        """
        #####
        # Plot the figure for per facility
        #####
        def plot_grid():
            plt.ylim(0, 3.5); plt.grid(axis='y', lw=0.6, linestyle='--')

        self.check_class(vnf, 'vnf', 'VNF')
        colors = self.get_cmap_colors(cmap_name='nipy_spectral_r', num_categories=len(vnf.status))

        fig = plt.figure(figsize=(10, 5)); gs = GridSpec(2, 1, hspace=0.1)

        ax = fig.add_subplot(gs[0]); plt.ylabel('VNF')
        self.barplot_with_bottom(vnf.gas, width=width, x_index=vnf.years, facility_index=vnf.capacity.index, colors=colors)
        plt.tick_params(labelbottom=False); plot_grid()

        ax = fig.add_subplot(gs[1]); plt.ylabel('WB')
        self.barplot_with_bottom(vnf.wb, width=width, x_index=vnf.years, facility_index=vnf.capacity.index, colors=colors)
        plot_grid(); plt.xlabel('Year')

        fig.legend(handles=[mpatch.Patch(label=lab, color=c) for lab, c in zip(vnf.capacity.index, colors)], loc='center', bbox_to_anchor=(0.5, -0.3), fancybox=True, ncols=5)
        plt.text(-0.1, 1.1, 'Volume of Gas Flared $(billion m^{-3})$', rotation=90, va='center', transform=ax.transAxes)

        plt.savefig('./Figures/General/Plot_per_facility.png', bbox_inches='tight'); plt.close()

        #####
        # Plot the figure for per country
        #####
        df1 = vnf.gas.copy(); df1['Country'] = vnf.status['COUNTRY']; dfgas = df1.groupby('Country').sum()
        df2 = vnf.wb.copy(); df2['Country'] = vnf.status['COUNTRY']; dfwb = df2.groupby('Country').sum()
        
        colors = self.get_cmap_colors(cmap_name='nipy_spectral_r', num_categories=len(dfgas))

        fig = plt.figure(figsize=(10, 5)); gs = GridSpec(2, 1, hspace=0.1)

        ax = fig.add_subplot(gs[0]); plt.ylabel('VNF')
        self.barplot_with_bottom(dfgas, width=width, x_index=vnf.years, facility_index=dfgas.index, colors=colors)
        plt.tick_params(labelbottom=False); plot_grid()

        ax = fig.add_subplot(gs[1]); plt.ylabel('WB')
        self.barplot_with_bottom(dfwb, width=width, x_index=vnf.years, facility_index=dfwb.index, colors=colors)
        plot_grid(); plt.xlabel('Year')

        fig.legend(handles=[mpatch.Patch(label=lab, color=c) for lab, c in zip(dfgas.index, colors)], loc='center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncols=5)
        plt.text(-0.1, 1.1, 'Volume of Gas Flared $(billion m^{-3})$', rotation=90, va='center', transform=ax.transAxes)

        plt.savefig('./Figures/General/Plot_per_countries.png', bbox_inches='tight'); plt.close()

    def check_different_distribution_start(self, vnf=None, pbt=None, n_s:int=2, startUp:bool=True, prob_funs:list=['GUM', 'GEV', 'LOG', 'EXP', 'GPD'], scen_name:str='', \
                                                    static_tests:list=['AIC', 'AICc', 'KS', 'BIC'], plot_figure:bool=True, perform_MLT:bool=True, title_d=['A)', 'B)', 'C)']):
        """
        This function performs series of statistical methods to evaluate the performance of different probability distributions fitted to the
        Volume of gas flared, Annual number of flaring days, and Length of flaring events.
        Statistical evaludation methods include
            Maximum Likelihood Ratio test (https://en.wikipedia.org/wiki/Likelihood-ratio_test)
            AIC test (https://en.wikipedia.org/wiki/Akaike_information_criterion)
            BIC test (https://en.wikipedia.org/wiki/Bayesian_information_criterion)
            KS test (https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)
            Q-Q plot (https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)
            Distribution plot (https://en.wikipedia.org/wiki/Probability_density_function)
        Available prob_fun:
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        Input:
            pnt: ProbClass = The user must pass the ProbClass class from probability_theory
            StartUp:bool = whether we are interesting in start-up period (True) or all year (False)
            n_s:int = if StartUp is True, n_s represent the number of startup years in consideration.
        Output include Figures and Tables in following directory
            Figures/Statics
            Results/Statics
        """
        self.check_class(vnf, 'vnf', 'VNF')
        self.check_class(pbt, 'pbt', 'ProbClass')
        if len(prob_funs) > 5: raise ValueError('check_different_distribution_start function can only check 5 distributions at a time')
        nullfuns, alterfuns = pbt.distinguish_prob_funs(prob_funs)
        columns:list = [f'MLT_{nu}_{al}' for nu, al in product(nullfuns, alterfuns)] + [f"{prefix}_{func}" for prefix, func in product(static_tests, prob_funs)]

        outDF = pd.DataFrame(index=['VNF', 'GAS', 'LENGTH'], columns=columns)
        gs = GridSpec(2, 3, wspace=0.4, hspace=0.4)
        for ti_,  (df, name) in enumerate(zip([vnf.gas_filter_start, vnf.vnf_filter_start, vnf.flareEpisodes_filter_start], ['GAS', 'VNF', 'LENGTH'])):
            
            # Select, clean, sort data
            if name in ['VNF', 'GAS']:
                data = df.values[:, :n_s].flatten() if startUp else df.values.flatten()
                data = np.sort(pbt.rm_nan_fl(data)) #; data = data[data>0]
            else:
                data = df.loc[df.index.get_level_values(1) <= n_s, 'length'].values.flatten() if startUp else df['length'].values.flatten()
                data = np.sort(pbt.rm_nan_fl(data))

            # Perform Maximum Likelihood Ratio test
            if perform_MLT:
                for nu, al in product(nullfuns, alterfuns):
                    outDF.at[name, f'MLT_{nu}_{al}'] = pbt.MLT_test(data, nullFun=nu, alterFun=al)[1]
            
            # Perform AIC and BIC tests
            aicc, aic, bic = pbt.AIC_BIC_test(data, prob_funs=prob_funs)
            aicc, aic = np.exp((aicc.min()-aicc)/2), np.exp((aic.min()-aic)/2)
            for i, j in enumerate(prob_funs):
                if 'AIC' in static_tests:  outDF.at[name, f'AIC_{j}'] = aic[i]
                if 'AICc' in static_tests: outDF.at[name, f'AICc_{j}'] = aicc[i]
                if 'BIC' in static_tests:  outDF.at[name, f'BIC_{j}'] = bic[i]
                if 'KS' in static_tests:   outDF.at[name, f'KS_{j}'] = pbt.kstest_distribution(data, j)

            if plot_figure:
                # Histogram subplot
                fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(gs[0, 0]); plt.title('Ordinary histogram')
                if 'GUM' in prob_funs: plt.plot(data, pbt.gevfun('dgev', data, pbt.gumbel_by_lmoments(data)), '-', color='tab:red', label='Gumbel')
                if 'GEV' in prob_funs: plt.plot(data, pbt.gevfun('dgev', data, pbt.hybrid_gev_params(data)), '-', color='tab:blue', label='GEV')
                if 'LOG' in prob_funs: plt.plot(data, pbt.lognormalfun('dlogn', data, pbt.lognormal_by_lmoments(data)), '-', color='tab:green', label='Log-Normal')
                if 'EXP' in prob_funs: plt.plot(data, pbt.exponentialfun('dexp', data, pbt.exponential_by_lmoments(data)), '-', color='tab:pink', label='Exponential')
                if 'GPD' in prob_funs: plt.plot(data, pbt.gpdfun('dgp', data, pbt.gpd_by_lmoments(data)), '-', color='tab:orange', label='GPD')
                if 'NOR' in prob_funs: plt.plot(data, pbt.normalfun('dnorm', data, pbt.normal_by_lmoments(data)), '-', color='tab:orange', label='Normal')
                plt.hist(data, density=True, bins='sqrt', histtype='stepfilled', alpha=0.2, label='Empirical'); plt.ylabel('Density'); plt.xlabel('Values')
                plt.legend(loc='best'); plt.suptitle(f'{title_d[ti_]} {pbt.static_table_columns[ti_]}\n{scen_name}') #plt.suptitle(f'{vnf.sname}: {name}')
                
                # Q-Q subplot
                if 'GUM' in prob_funs: ax = fig.add_subplot(gs[0, 1]); self.qq_ploter(data, pbt=pbt, color='tab:red', prob_fun='GUM', label='Gumbel'); self.qq_plot_labels()
                if 'GEV' in prob_funs: ax = fig.add_subplot(gs[0, 2]); self.qq_ploter(data, pbt=pbt, color='tab:blue', prob_fun='GEV', label='GEV'); self.qq_plot_labels()
                if 'LOG' in prob_funs: ax = fig.add_subplot(gs[1, 2]); self.qq_ploter(data, pbt=pbt, color='tab:green', prob_fun='LOG', label='Log-Normal'); self.qq_plot_labels()
                if 'EXP' in prob_funs: ax = fig.add_subplot(gs[1, 1]); self.qq_ploter(data, pbt=pbt, color='tab:pink', prob_fun='EXP', label='Exponential'); self.qq_plot_labels()
                if 'GPD' in prob_funs: ax = fig.add_subplot(gs[1, 0]); self.qq_ploter(data, pbt=pbt, color='tab:orange', prob_fun='GPD', label='Generalized Patero'); self.qq_plot_labels()
                if 'NOR' in prob_funs: ax = fig.add_subplot(gs[1, 0]); self.qq_ploter(data, pbt=pbt, color='tab:orange', prob_fun='NOR', label='Normal'); self.qq_plot_labels()
                
                self.legend_probability(bbox_to_anchor=[0.5, 0.01], fig=fig, general_dist=True)
                plt.savefig(f'./Figures/Statics/{vnf.sname}_{name}.png', bbox_inches='tight');plt.close()
        print(outDF.dropna(how='all', axis=1))
        outDF.dropna(how='all', axis=1).to_csv(f'./Results/Statics/MLT_test_{vnf.sname}.csv')
        return
    
    def qq_plot_labels(self,):
        plt.xlabel('Empirical Data'); plt.ylabel('Theoretical Data')
        
    def plot_qq_figure_paper(self, vnf=None, pbt=None, n_s:int=2, prob_fun:str="LOG", startUp:bool=True):
        """
        Plot three subplots for pbt.static_table_title variables. Each subplot presents q-q plot for the variable for the given prob_fun
        """
        self.check_class(vnf, 'vnf', 'VNF')
        self.check_class(pbt, 'pbt', 'ProbClass')

        staticTable = pd.DataFrame(index=pd.Index(pbt.static_table_title, name='paramter'), columns=pd.Index(['cmvtest', 'kstest'], name='test'))
        fig = plt.figure(figsize=(12, 4)); gs = GridSpec(1, 3, wspace=0.3)

        for i_, (df, label, name) in enumerate(zip([vnf.gas_filter_start, vnf.vnf_filter_start, vnf.flareEpisodes_filter_start], pbt.static_table_title, ['GAS', 'VNF', 'LENGTH'])):
            # Select, clean, sort data
            if name in ['VNF', 'GAS']:
                data = df.values[:, :n_s].flatten() if startUp else df.values.flatten()
                data = np.sort(pbt.rm_nan_fl(data)); data = data[data>0]
            else:
                data = df.loc[df.index.get_level_values(1) <= n_s, 'length'].values.flatten() if startUp else df['length'].values.flatten()
                data = np.sort(pbt.rm_nan_fl(data))

            staticTable.at[label, 'cmvtest'] = pbt.cvmtest_distribution(data, prob_fun)
            staticTable.at[label, 'kstest'] = pbt.kstest_distribution(data, prob_fun)
            
            ax = fig.add_subplot(gs[i_])
            self.qq_ploter(data, pbt=pbt, color='tab:red', prob_fun=prob_fun, label=f'{label}\n{pbt.static_table_units[i_]}') #; self.setting_ylim(name)
            self.qq_plot_labels()
        self.legend_probability(bbox_to_anchor=[0.5, -0.1], fig=fig)
        plt.savefig(f'./FIgures/Statics/QQplot_{vnf.sname}.png', bbox_inches='tight'); plt.close()
        staticTable.to_csv(f'./Results/Statics/GOF_{vnf.sname}.csv')
        return staticTable

    def legend_probability(self, fig=None, bbox_to_anchor=[0.5, -0.1], general_dist:bool=False, label_fitted:str='Fitted Log-Normal Estimates'):
        if general_dist: label_fitted = 'Fitted Distribution Estimates'
        fig.legend(handles=[Line2D([0], [0], linestyle='--', linewidth=2, color='tab:red', label='95% Confidence Intervals'),                   \
                            Line2D([0], [0], linestyle='-',  linewidth=2, color='tab:red', label=label_fitted),                                 \
                            Line2D([0], [0], markersize=10, marker='o', markerfacecolor='tab:blue', linestyle='', label='Empirical Estimates')],\
                    bbox_to_anchor=bbox_to_anchor, ncols=3, loc='center', fancybox=True)
        return
        
    def setting_ylim(self, name):
        if name == 'VNF': plt.ylim(-10, 365); return
        if name == 'GAS': plt.ylim(-0.05, 0.3); return
        plt.ylim(-5, 30); return
    
    def qq_ploter(self, flattened_data:np.ndarray, pbt=None, color:str='tab:red', prob_fun:str='GUM', label:str='Gumbel', return_periods=np.array([1.001, 2, 5, 50])):
        """
        This function plots q-q in a given ax input baed on the input prob_fun
        Available prob_fun:
            GUM: Gumbel
            GEV: Generalized Extreme Value
            LOG: Log-Normal
            GPD: Generalized Perato
            EXP: Exponential
            NOR: Normal
        Input:
            Flattend_data: 1D numpy array. the np.nan values should be removed.
        """
        self.check_class(pbt, 'pbt', 'ProbClass')
        if len(flattened_data) < 1: return
        # Fixed variables
        return_periods_prob = 1 - 1 / return_periods
        n = len(flattened_data)
        plotting_positions = np.arange(1, n+1) / (n+1)

        # Find the correct parhat and qq posions based on prob_fun
        yy1, y_low, y_high, gumbel_dict = pbt.boot_correct_probability(flattened_data, return_periods_prob, plotting_positions, prob_fun=prob_fun)
        
        # Plot the Q-Q plot of distribution of confidence intervals
        plt.plot(gumbel_dict['Q'], yy1, "-", color=color, lw=0.9, label='Fitted Distribution')
        plt.plot(gumbel_dict['Q'], y_low, "--", color=color, lw=0.8)
        plt.plot(gumbel_dict['Q'], y_high, "--", color=color, lw=0.8)
        # Scatter Plot the Experical estimates and adjust the Xticks
        plt.scatter(gumbel_dict['Q'], flattened_data, s=3, facecolor=None, edgecolor='tab:blue', label='Emperical Estimates')
        # plt.xticks(gumbel_dict['T'], 1 / return_periods, rotation=45, ha='right')
        vmin, vmax = min(y_low.min(), gumbel_dict['Q'].min()) - 1, min(y_high.max(), gumbel_dict['Q'].max()) + 1
        plt.xlim(vmin, vmax); plt.ylim(vmin, vmax)
        plt.grid(True); plt.title(label) #; plt.legend(loc='best')
        return

if __name__ == '__main__':
    pbt = PloterClass()
