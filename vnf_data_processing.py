import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from glob import glob
from subprocess import check_call
from scipy.spatial import cKDTree
from copy import deepcopy

class VNF:
    pd.set_option('future.no_silent_downcasting', True)
    """
    The VNF class is initialized based on the complete dataset of daily number of flaring, annual volume of gas flared, and length of flaring events.
    After initialization, the object can be used filter and plot data. it can uses ProbClass from probability_theory to apply statistical analysis.
    Input:
        temperature_threshold:int = 1400 (default:Kelvin)
        bufferSizeVNF:int = 750 (default:meter)
        bufferSizeGas:int = 2500 (default:meter)
    Output:
        VNF class = object
    The object has many functions including
        filter
        plot_compare_vnf_wb
        check_different_distribution_start
    """
    def __init__(self, temperature_threshold:int=1400, bufferSizeVNF:int=750, bufferSizeGas:int=2500, quickloading:bool=False, could_mask:bool=True, saveFilterData:bool=True, adjustedVNFcount:bool=False):
        self.createOutputDirectories()
        self.quickloading = quickloading
        
        self.years = pd.Index([f'{i}' for i in range(2012, 2023)], name='Years', dtype='str')
        self.status, self.capacity = self.getCapacityStatus()
        self.wb = self.getWBgas(self.capacity.index)
        
        if quickloading:
            self.QuickLoad(index=self.capacity.index)
        else:
            self.getGasProcessVNF(self.capacity.index, bufferSize=bufferSizeGas, saveFilterData=saveFilterData) # Fix capacity of Sabine Pass and Yemen in the input dataset
            if adjustedVNFcount:
                vnf_all = self.vnfAnnualProcess(self.capacity.index, Temp_thres=temperature_threshold, bufferSize=bufferSizeVNF, could_mask=False, saveFilterData=False)
                vnf_clear = self.vnfAnnualProcess(self.capacity.index, Temp_thres=temperature_threshold, bufferSize=bufferSizeVNF, could_mask=True, saveFilterData=False)
                vnf_adjusted = vnf_clear + (vnf_all - vnf_clear) * vnf_clear / 365
                vnf_adjusted = vnf_adjusted.where(vnf_adjusted <= 365, 365)
                self.vnf = vnf_adjusted.where(vnf_adjusted <= vnf_all, vnf_all).round(0)
            else:
                self.vnf = self.vnfAnnualProcess(self.capacity.index, Temp_thres=temperature_threshold, bufferSize=bufferSizeVNF, could_mask=could_mask, saveFilterData=saveFilterData)
            # self.getVNFgas() # Read the preprocessed Volume of gas flared from VNF dataset. 
        
    def filter_zeros_data(self,):
        self.vnf = self.vnf.where(self.vnf != 0, np.nan)
        self.gas = self.gas.where(self.gas != 0, np.nan)
        self.wb =  self.wb.where(self.wb   != 0, np.nan)
    
    def filter_zeros_dataframe(self, dfI:pd.DataFrame) -> pd.DataFrame:
        return dfI.where(dfI>0, np.nan)

    def control_data(self, ):
        mask =  ~np.logical_or(self.gas.sum(axis=1) == 0, self.vnf.sum(axis=1) == 0) # ~(self.vnf.sum(axis=1) == 0)#
        self.capacity         = self.capacity.loc[mask]
        self.vnf              = self.vnf.loc[mask]
        self.wb               = self.wb.loc[mask]
        self.gas              = self.gas.loc[mask]
        self.status           = self.status.loc[mask]
        self.capacity         = self.capacity.loc[mask]
        self.detection        = self.detection.loc[mask]
        self.cleanobservation = self.cleanobservation.loc[mask]
        self.temperature      = self.temperature.loc[mask]
        self.flareEpisodes    = self.flareEpisodes.loc[self.flareEpisodes.index.get_level_values(0).isin(self.capacity.index)]
        
    def create_summary_tables(self, pbt=None):
        state = self.status.copy().sort_values(by=['IM_STATUS', 'StartYear'])
        indexes = state.index
        capacity = self.capacity.copy().loc[indexes]
        vnf = self.vnf.copy().loc[indexes]
        gas = self.gas.copy().loc[indexes]
        vnf = self.filter_zeros_dataframe(vnf)
        gas = self.filter_zeros_dataframe(gas)
        gasPerCap = gas / capacity * 100
        capacity['cap'] = capacity[self.years].min(axis=1).astype(str) + ' - ' + capacity[self.years].max(axis=1).astype(str)

        pd.DataFrame({'Facility Name':indexes, 'Country':state['COUNTRY'], 'Onshore/Offshore':state['Shore'],                       \
                      'Start Year':state['StartYear'], 'Capacity range since 2012 (mtpa)':capacity['cap'],                          \
                      'Average number of flaring days per year since opening':vnf.mean(axis=1).round(0),                \
                      'Average volume of gas flared (bcm)':gas.mean(axis=1).round(4),                                               \
                      'Average volume of gas flared per capacity of the facility (% - bcm/bcm)':gasPerCap.mean(axis=1).round(2),    \
                      }).reset_index(drop=True).fillna('N/A').to_csv('./Results/WholeData/SummaryTablePaperPerFacility.csv', index=False)

        capacity.drop(columns=['cap'], inplace=True)
        state_c = state.groupby('COUNTRY')
        state_c_g = state_c.agg(min_date=('StartYear', 'min'), max_date=('StartYear', 'max'))
        state_c_g['cy'] = state_c_g['min_date'].astype(str) + ' - ' + state_c_g['max_date'].astype(str)
        country_list = state_c_g.index
        on_off_count = state_c['Shore'].value_counts().unstack(fill_value=0)
        operation_count = state_c['IM_STATUS'].value_counts().unstack(fill_value=0)
        vnf['cy'], gas['cy'], capacity['cy'], gasPerCap['cy'] = state['COUNTRY'], state['COUNTRY'], state['COUNTRY'], state['COUNTRY']
        
        capacity_c = capacity.groupby('cy').sum()
        capacity_c['cy'] = capacity_c.min(axis=1).astype(str) + ' - ' + capacity_c.max(axis=1).astype(str)
        for country, group in vnf.groupby('cy'):
            values = pbt.rm_nan_fl(group[self.years].mean(axis=0).values)
            if len(values) == 0: continue
            me_, lo_, hi_ = pbt.boot_emperical_function(values, np.mean, nboot=2000)
            capacity_c.at[country, 'flare'] = f'{me_:.0f} ({lo_:.0f} - {hi_:.0f})'
        
        for country, group in gasPerCap.groupby('cy'):
            values = pbt.rm_nan_fl(group[self.years].mean(axis=0).values)
            if len(values) == 0: continue
            me_, lo_, hi_ = pbt.boot_emperical_function(values, np.mean, nboot=2000)
            capacity_c.at[country, 'gaspc'] = f'{me_:.2f} ({lo_:.2f} - {hi_:.2f})'
        
        # for country, group in gas.groupby('cy'):
        #     values = pbt.rm_nan_fl(group[self.years].sum(axis=1).values)
        #     if len(values) == 0: continue
        #     me_, lo_, hi_ = pbt.boot_emperical_function(values, np.mean, nboot=2000)
        #     capacity_c.at[country, 'gas'] = f'{me_:.3f} ({lo_:.4f} - {hi_:.4f})'
        
        pd.DataFrame({'# Onshore':on_off_count['On'], '# Offshore':on_off_count['Off'], 'LNG Construction Year': state_c_g['cy'],           \
                        '# Operational':operation_count['Operational'], '# Inactive':operation_count['Inactive'],                           \
                        'Total Capacity':capacity_c['cy'], 'Average Number of Annual Flaring Days': capacity_c['flare'],                    \
                        'Average Volume of Gas Flared Per Capacity':capacity_c['gaspc'],    \
                        }, index=country_list).reset_index().fillna('N/A').to_csv('./Results/WholeData/SummaryTablePaperPerCountry.csv', index=False)
        
    def saveFilter(self,startup:bool=True):
        if not hasattr(self, 'sname') and not hasattr(self, 'vnf_filter'): raise ValueError("The saveFilter function needs the vnf_filtered object not a VNF class")
        if startup:
            self.vnf_filter_start.to_csv(f'./Results/Data/Filter_startup/AnnualDailyFlaresActivity_{self.sname}.csv')
            self.gas_filter_start.to_csv(f'./Results/Data/Filter_startup/AnnualVolumeOfGasFlaredPerCap_{self.sname}.csv')
            self.detection_filter_start.to_csv(f'./Results/Data/Filter_startup/AnnualDetection_{self.sname}.csv')
            self.cleanobservation_filter_start.to_csv(f'./Results/Data/Filter_startup/AnnualCleanObservation_{self.sname}.csv')
            self.temperature_filter_start.to_csv(f'./Results/Data/Filter_startup/AnnualAverageTemperature_{self.sname}.csv')
            self.flareEpisodes_filter_start.to_csv(f'./Results/Data/Filter_startup/FlareEpisodes_{self.sname}.csv')
            return
        self.vnf_filter.to_csv(f'./Results/Data/Filter/AnnualDailyFlaresActivity_{self.sname}.csv')
        self.gas_filter.to_csv(f'./Results/Data/Filter/AnnualVolumeOfGasFlaredPerCap_{self.sname}.csv')
        self.detection_filter.to_csv(f'./Results/Data/Filter/AnnualDetection_{self.sname}.csv')
        self.cleanobservation_filter.to_csv(f'./Results/Data/Filter/AnnualCleanObservation_{self.sname}.csv')
        self.temperature_filter.to_csv(f'./Results/Data/Filter/AnnualAverageTemperature_{self.sname}.csv')
        self.flareEpisodes_filter.to_csv(f'./Results/Data/Filter/FlareEpisodes_{self.sname}.csv')
    
    def copy(self,):
        """
        it returns deepcopy of self [VNF] object
        """
        return deepcopy(self)

    def savePreprocessed(self,):
        """
        Check if quickloading is not True, save all data that requires preprocessing.
        """
        if self.quickloading: return
        self.gas.to_csv('./Results/WholeData/gas_prep.csv')
        self.detection.to_csv('./Results/WholeData/detection_prep.csv')
        self.cleanobservation.to_csv('./Results/WholeData/cleanobs.csv')
        self.temperature.to_csv('./Results/WholeData/temperature.csv')
        self.vnf.to_csv('./Results/WholeData/VNF_records.csv')
        self.flareEpisodes.to_csv('./Results/WholeData/FlareEpisodes.csv')

    def QuickLoad(self, index):
        self.gas:pd.DataFrame = pd.read_csv('./Results/WholeData/gas_prep.csv', index_col='FAC_NAME').loc[index][self.years]
        self.detection:pd.DataFrame = pd.read_csv('./Results/WholeData/detection_prep.csv', index_col='FAC_NAME').loc[index][self.years]
        self.cleanobservation:pd.DataFrame = pd.read_csv('./Results/WholeData/cleanobs.csv', index_col='FAC_NAME').loc[index][self.years]
        self.temperature:pd.DataFrame = pd.read_csv('./Results/WholeData/temperature.csv', index_col='FAC_NAME').loc[index][self.years]
        self.vnf:pd.DataFrame = pd.read_csv('./Results/WholeData/VNF_records.csv', index_col='FAC_NAME').loc[index][self.years]
        self.flareEpisodes:pd.DataFrame = pd.read_csv('./Results/WholeData/FlareEpisodes.csv', index_col=['FAC_NAME', 'Year', 'episode_id'])

    def createOutputDirectories(self,):
        check_call(f'mkdir -p Results/Data/Filter Results/Data/Filter_startup Results/Statics Results/Distributions Results/WholeData', shell=True)
        check_call(f'mkdir -p Figures/Output Figures/Statics Figures/General', shell=True)

    def filter(self, name:str='report', expImpType:str='Export', startUp:int=1700, onShore:str='All', imStatus:str='All', facStatus:str='All', \
                        onlyRegularOperation:bool=False, numStartUpRegular:int=1, column_reorder:bool=False, removeKiyanly:bool=True, onlyKiyanly:bool=False):
        if onlyRegularOperation:
            if numStartUpRegular < 1 or numStartUpRegular > 10: raise ValueError('Invalid numStartUpRegular when using onlyRegularOperation. Use in range 1 and 10')
            if imStatus != 'Operational': raise ValueError('Invalid imStatus when using onlyRegularOperation. Use Operational')
        
        # Create a deepcopy of filter so I can create new objects from filter       
        new_obj = deepcopy(self)
        new_obj.sname = name
        new_obj.status_filter = self.status.copy()
        new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.StartYear >= startUp-numStartUpRegular+1]    

        # Filter for LNG type (Export/Import/All)
        if expImpType == 'All': pass
        elif expImpType == 'Export': new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.FAC_TYPE == 'Export']
        elif expImpType != 'Export': new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.FAC_TYPE != 'Export']
        else: raise ValueError("Invalid exp_imp_type. Use Export, All, Import")

        # Filter for LNG IM_STATUS (Operational/Inactive/Abandoned)
        if imStatus == 'All': pass
        elif imStatus == 'Operational': new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.IM_STATUS == 'Operational']
        elif imStatus == 'Inactive': new_obj.status_filter = new_obj.status_filter.loc[np.logical_or(new_obj.status_filter.IM_STATUS == 'Inactive', new_obj.status_filter.IM_STATUS == 'Abandoned')]
        else: raise ValueError("Invalid imStatus. Use Operational, Inactive")

        # Filter for LNG location (Onshore/Offshore)
        if onShore == 'All': pass
        elif onShore == 'On': new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.Shore == 'On']
        elif onShore == 'Off': new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.Shore == 'Off']
        else: raise ValueError("Invalid onShore type. Use All, On, Off")

        # Filter for FAC_STATUS ('Operating/Retired' 'Operating' 'Mothballed' 'Idle' 'Retired' 'Operating/Mothballed')
        if facStatus == 'All': pass
        elif facStatus == 'Operating': new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.FAC_STATUS == 'Operating']
        elif facStatus == 'notOperating':
            new_obj.status_filter = new_obj.status_filter.loc[np.logical_or( \
                                                                np.logical_or(np.logical_or(new_obj.status_filter.FAC_STATUS == 'Operating/Retired', new_obj.status_filter.FAC_STATUS == 'Mothballed'), \
                                                                            np.logical_or(new_obj.status_filter.FAC_STATUS == 'Idle', new_obj.status_filter.FAC_STATUS == 'Retired')), \
                                                                new_obj.status_filter.FAC_STATUS == 'Operating/Mothballed')]
        else: raise ValueError("Invalid facStatys type. Use Operating/Retired Operating Mothballed Idle Retired Operating/Mothballed")

        # Selecting the filtered data
        if onlyKiyanly: new_obj.status_filter = new_obj.status_filter.loc[new_obj.status_filter.index == 'Kiyanly LNG']
        elif removeKiyanly and 'Kiyanly LNG' in new_obj.status_filter.index: new_obj.status_filter = new_obj.status_filter.drop(['Kiyanly LNG'])
        
        new_obj.index_filter = new_obj.status_filter.index
        new_obj.capacity_filter = new_obj.capacity.loc[new_obj.index_filter]
        new_obj.vnf_filter = new_obj.vnf.loc[new_obj.index_filter]
        new_obj.gas_filter = new_obj.gas.loc[new_obj.index_filter] / new_obj.capacity_filter * 100
        new_obj.detection_filter = new_obj.detection.loc[new_obj.index_filter]
        new_obj.cleanobservation_filter = new_obj.cleanobservation.loc[new_obj.index_filter]
        new_obj.temperature_filter = new_obj.temperature.loc[new_obj.index_filter]
        new_obj.flareEpisodes_filter = new_obj.flareEpisodes.loc[new_obj.flareEpisodes.index.get_level_values(0).isin(new_obj.index_filter)]

        # Filter for only Regular Operation
        if onlyRegularOperation:
            for index, year in zip(new_obj.status_filter.index, new_obj.status_filter.StartYear):
                for j in range(numStartUpRegular):
                    if year+j < 2012 or year+j>2022: continue

                    new_obj.vnf_filter.at[index, f'{year+j}'] = np.nan
                    new_obj.gas_filter.at[index, f'{year+j}'] = np.nan
                    new_obj.detection_filter.at[index, f'{year+j}'] = np.nan
                    new_obj.cleanobservation_filter.at[index, f'{year+j}'] = np.nan
                    new_obj.temperature_filter.at[index, f'{year+j}'] = np.nan
                    new_obj.flareEpisodes_filter.loc[np.logical_and(
                                new_obj.flareEpisodes_filter.index.get_level_values(0) == index,
                                new_obj.flareEpisodes_filter.index.get_level_values(1) == f'{year+j}'), 'length'] = np.nan
            new_obj.flareEpisodes_filter = new_obj.flareEpisodes_filter.dropna(subset=['length'])

        # Create the startUp dataframe (i.e., columns are change to the year of operating: 1-11)
        if column_reorder:
            new_obj.vnf_filter_start:pd.DataFrame              = new_obj.process_start_filter(new_obj.vnf_filter)
            new_obj.gas_filter_start:pd.DataFrame              = new_obj.process_start_filter(new_obj.gas_filter)
            new_obj.detection_filter_start:pd.DataFrame        = new_obj.process_start_filter(new_obj.detection_filter)
            new_obj.cleanobservation_filter_start:pd.DataFrame = new_obj.process_start_filter(new_obj.cleanobservation_filter)
            new_obj.temperature_filter_start:pd.DataFrame      = new_obj.process_start_filter(new_obj.temperature_filter)
            new_obj.flareEpisodes_filter_start:pd.DataFrame    = new_obj.process_start_filter_eventlength(new_obj.flareEpisodes_filter)
        else:
            new_obj.vnf_filter_start:pd.DataFrame              = new_obj.vnf_filter
            new_obj.gas_filter_start:pd.DataFrame              = new_obj.gas_filter
            new_obj.detection_filter_start:pd.DataFrame        = new_obj.detection_filter
            new_obj.cleanobservation_filter_start:pd.DataFrame = new_obj.cleanobservation_filter
            new_obj.temperature_filter_start:pd.DataFrame      = new_obj.temperature_filter
            new_obj.flareEpisodes_filter_start:pd.DataFrame    = new_obj.flareEpisodes_filter

        return new_obj

    def process_start_filter(self, dfIn:pd.DataFrame) -> pd.DataFrame:
        df = pd.DataFrame(index=self.index_filter, columns=[f'{i}' for i in range(1, len(self.years)+1)])
        for index, year in zip(self.status_filter.index, self.status_filter.StartYear):
            for i, iYear in enumerate(range(year, 2023)):
                if iYear < 2012: continue
                df.at[index, f'{i+1}'] = dfIn.at[index, f'{iYear}']
        return df

    def process_start_filter_eventlength(self, dfIn:pd.DataFrame) -> pd.DataFrame:
        df = dfIn.reset_index()
        for index in df['FAC_NAME'].unique():
            year, mask = self.status.at[index, 'StartYear'], df['FAC_NAME'] == index
            df.loc[mask, 'year_int'] = df.loc[mask, 'year_int'] - year + 1
        return df.set_index(['FAC_NAME', 'year_int', 'episode_id'])

    def getVNFgas(self, index):
        self.gas = self.adjust_start_year(pd.read_csv('./gas/ExtractedFromCSV/GasVNF_BCM.csv', index_col=['FAC_NAME']).loc[index][self.years].fillna(0))

    def getWBgas(self, index):
        return self.adjust_start_year(pd.read_csv('./WorldBank/AllLNG_new.csv', index_col=['FAC_NAME']).loc[index][self.years].fillna(0) / 1000)

    def getCapacityStatus(self,):
        df = pd.read_csv('./WorldBank/AllCapacities_new.csv', index_col='FAC_NAME')
        return df[['COUNTRY', 'FAC_STATUS', 'FAC_TYPE', 'IM_STATUS', 'Shore', 'StartYear', 'IdleYear']], df[self.years]

    def check_column(self, colI):
        return colI.__contains__('BCM') or colI.__contains__('Detec') or colI.__contains__('Clear') or colI.__contains__('Avg')
        
    def extract_location(self, fname='./WorldBank/AllCordsVNF_new', bufferSize=750, createBuffer=False):
        df = pd.read_csv(f'{fname}.csv')
        df['geometry'] = [Point(lon, lat) for lon, lat in zip(df['lon'], df['lat'])]
        gdf = gpd.GeoDataFrame(data=df, geometry='geometry', crs='EPSG:4326')
        gdf['utm_epsg'] = self.get_utm_zone(gdf).astype(np.int32)
        
        if createBuffer:
            buffered_gdfs = []
            # Group by UTM zone
            for epsg_code, group in gdf.groupby("utm_epsg"):
                # Reproject to local UTM
                local_proj = group.to_crs(epsg=epsg_code)        
                # Create 1 km buffer (1500 meters)
                local_proj['geometry'] = local_proj.geometry.buffer(bufferSize)
                # Reproject back to WGS84 for merging
                buffered_wgs84 = local_proj.to_crs(epsg=4326)
                buffered_gdfs.append(buffered_wgs84)
            # Merge all buffered subsets
            return pd.concat(buffered_gdfs).reset_index(drop=True) #.drop(columns='utm_epsg')
        return gdf

    def get_utm_zone(self, gdfIn):
        return gdfIn.geometry.apply(lambda geom: self.get_utm_epsg(geom.y, geom.x))

    # Function to get UTM zone EPSG code (Northern Hemisphere)
    def get_utm_epsg(self, lat, lon):
        zone = (((lon + 180) // 6) % 60) + 1
        if lat >= 0: return 32600 + zone  # Northern Hemisphere
        else: return 32700 + zone  # Southern Hemisphere

    def getGasProcessVNF(self, index, bufferSize=1500, saveFilterData:bool=True):
        # Reading files
        otherNearby = self.extract_location(createBuffer=False, fname='./AllOilGasSHP/Nearby_facilities_cleaned')
        otherNearby = otherNearby.loc[otherNearby.CATEGORY != 'Natural gas flaring detections']
        otherNearby = otherNearby.loc[otherNearby.CATEGORY != 'Oil and natural gas wells']
        Final_out1 = pd.DataFrame(0, index=index, columns=self.years); Final_out2, Final_out3, Final_out4 = Final_out1.copy(), Final_out1.copy(), Final_out1.copy()
        group_df_list = []
        for item in [2012, 2017, 2018, 2019, 2020, 2021, 2022]:
            locations = self.extract_location(createBuffer=True, bufferSize=bufferSize)
            df = pd.concat([pd.read_csv(i) for i in sorted(glob(f'./gas/gas/*{item}*.csv'))], ignore_index=True) #pd.read_csv(glob(f'./gas/gas/*{item}*.csv')[0]) #pd.concat([pd.read_csv(i) for i in sorted(glob(f'./gas/gas/*{item}*.csv'))], ignore_index=True)
            # Creating geometry in EPSG:4326
            df['geometry'] = [Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
            gdf = gpd.GeoDataFrame(data=df, geometry='geometry', crs='EPSG:4326')
            # Filtering VNF records to 750m of the points
            out = gpd.sjoin(locations, gdf, how='right', predicate='contains').dropna(subset='FAC_NAME').drop(columns=['index_left', 'COUNTRY', 'lat', 'lon', 'FAC_NAME']).drop_duplicates().reset_index(drop=True)
            out['utm_epsg'] = out['utm_epsg'].astype(np.int32)
            locations['geometry'] = [Point(lon, lat) for lon, lat in zip(locations['lon'], locations['lat'])]
            
            all_locations = pd.concat([locations, otherNearby]) # locations.copy() #
            # Clustering in appropriate UTM zone
            clustered_flares = []
            for utm_zone in sorted(locations['utm_epsg'].unique()):
                out_utm = out.loc[out.utm_epsg == utm_zone].to_crs(epsg=utm_zone)
                locations_utm = all_locations.loc[all_locations.utm_epsg == utm_zone].to_crs(epsg=utm_zone)
                
                # Build KDTree on facility coordinates ---
                fac_coords = np.vstack([locations_utm.geometry.x, locations_utm.geometry.y]).T
                tree = cKDTree(fac_coords)

                # Clustering using cKDTree. Assign nearest facility for each flare record ---
                flare_coords = np.vstack([out_utm.geometry.x, out_utm.geometry.y]).T
                _, nearest_facility_idx = tree.query(flare_coords, k=1)
                out_utm['facility_index'] = nearest_facility_idx
                out_utm['FAC_NAME'] = locations_utm.iloc[nearest_facility_idx].reset_index(drop=True)['FAC_NAME'].values
                clustered_flares.append(out_utm.to_crs(epsg=4326))
            cols = [col for col in clustered_flares[0].columns if self.check_column(col)] + ['FAC_NAME', 'Latitude', 'Longitude']
            clustered_flares_df = pd.concat(clustered_flares, ignore_index=True).reset_index(drop=True)[cols]
            group_df_list.append(clustered_flares_df)

            clustered_groupby_sum = clustered_flares_df.groupby(['FAC_NAME']).sum()
            clustered_groupby_mean = clustered_flares_df.groupby(['FAC_NAME']).mean()
            
            cols_bcm = [col for col in clustered_groupby_sum.columns if col.__contains__('BCM')]
            cols_temp = [col for col in clustered_groupby_sum.columns if col.__contains__('Avg')]
            cols_year = [col[-4:] for col in clustered_groupby_sum.columns if col.__contains__('BCM')]
            cols_dete = [col for col in clustered_groupby_sum.columns if col.__contains__('Detec')]
            cols_clea = [col for col in clustered_groupby_sum.columns if col.__contains__('Clear')]
            temp1 = pd.DataFrame(0, index=clustered_groupby_mean.index, columns=cols_year); temp2, temp3, temp4 = temp1.copy(), temp1.copy(), temp1.copy()
            
            for year, bcm, dete, clea in zip(cols_year, cols_bcm, cols_dete, cols_clea):
                temp1[year] = clustered_groupby_sum[bcm]
                temp2[year] = clustered_groupby_mean[dete]
                temp3[year] = clustered_groupby_mean[clea]
                temp4[year] = clustered_groupby_mean[cols_temp]
            
            Final_out1 = Final_out1.add(temp1, fill_value=0)
            Final_out2 = Final_out2.add(temp2, fill_value=0)
            Final_out3 = Final_out3.add(temp3, fill_value=0)
            Final_out4 = Final_out4.add(temp4, fill_value=0)

        group_df = pd.concat(group_df_list, ignore_index=True)
        if saveFilterData: group_df.to_csv('./Results/WholeData/Filtered_cleaned_gas.csv')
        self.gas:pd.DataFrame = self.adjust_start_year(Final_out1.loc[index].fillna(0)[self.years])
        self.detection:pd.DataFrame = self.adjust_start_year(Final_out2.loc[index].fillna(0)[self.years])
        self.cleanobservation:pd.DataFrame = self.adjust_start_year(Final_out3.loc[index].fillna(0)[self.years])
        self.temperature:pd.DataFrame = self.adjust_start_year(Final_out4.loc[index].fillna(0)[self.years])
        return
    
    def getGasProcessVNF_KML(self, index, bufferSize=1500, saveFilterData:bool=True):
        # Reading files
        otherNearby = self.extract_location(createBuffer=False, fname='./AllOilGasSHP/Nearby_facilities_cleaned')
        otherNearby = otherNearby.loc[otherNearby.CATEGORY != 'Natural gas flaring detections']
        otherNearby = otherNearby.loc[otherNearby.CATEGORY != 'Oil and natural gas wells']
        Final_out1 = pd.DataFrame(0, index=index, columns=self.years); Final_out2, Final_out3, Final_out4 = Final_out1.copy(), Final_out1.copy(), Final_out1.copy()
        group_df_list = []

        locations = self.extract_location(createBuffer=True, bufferSize=bufferSize)
        df = pd.concat([pd.read_csv(i) for i in glob(f'./gas/ExtractedFromKML/CSV_FILES/*.csv')])[['Year', 'BCM_flaring', 'Clear_obs', 'Clear_pct', 'T_avg_K', 'Latitude', 'Longitude']]

        # Creating geometry in EPSG:4326
        df['geometry'] = [Point(lon, lat) for lon, lat in zip(df['Longitude'], df['Latitude'])]
        gdf = gpd.GeoDataFrame(data=df, geometry='geometry', crs='EPSG:4326')
        # Filtering VNF records to 750m of the points
        out = gpd.sjoin(locations, gdf, how='right', predicate='contains').dropna(subset='FAC_NAME').drop(columns=['index_left', 'COUNTRY', 'lat', 'lon', 'FAC_NAME']).drop_duplicates().reset_index(drop=True)
        out['utm_epsg'] = out['utm_epsg'].astype(np.int32)
        locations['geometry'] = [Point(lon, lat) for lon, lat in zip(locations['lon'], locations['lat'])]
        
        all_locations = pd.concat([locations, otherNearby]) # locations.copy() #
        # Clustering in appropriate UTM zone
        clustered_flares = []
        for utm_zone in sorted(locations['utm_epsg'].unique()):
            out_utm = out.loc[out.utm_epsg == utm_zone].to_crs(epsg=utm_zone)
            locations_utm = all_locations.loc[all_locations.utm_epsg == utm_zone].to_crs(epsg=utm_zone)
            
            # Build KDTree on facility coordinates ---
            fac_coords = np.vstack([locations_utm.geometry.x, locations_utm.geometry.y]).T
            tree = cKDTree(fac_coords)

            # Clustering using cKDTree. Assign nearest facility for each flare record ---
            flare_coords = np.vstack([out_utm.geometry.x, out_utm.geometry.y]).T
            _, nearest_facility_idx = tree.query(flare_coords, k=1)
            out_utm['facility_index'] = nearest_facility_idx
            out_utm['FAC_NAME'] = locations_utm.iloc[nearest_facility_idx].reset_index(drop=True)['FAC_NAME'].values
            clustered_flares.append(out_utm.to_crs(epsg=4326))
        clustered_flares_df = pd.concat(clustered_flares, ignore_index=True).reset_index(drop=True)[['FAC_NAME', 'Latitude', 'Longitude', 'Year', 'BCM_flaring', 'Clear_obs', 'Clear_pct', 'T_avg_K']]
        clustered_flares_df['Year'] = clustered_flares_df['Year'].astype(str)
        
        clustered_grouped = clustered_flares_df.groupby(['FAC_NAME', 'Year'])
        clustered_groupby_sum = clustered_grouped.sum().reset_index()
        clustered_groupby_max = clustered_grouped.max().reset_index()
        clustered_groupby_mean = clustered_grouped.mean().reset_index()

        Final_out1 = clustered_groupby_sum.pivot(index='FAC_NAME', columns='Year', values='BCM_flaring').dropna(how='all', axis=1).fillna(0).astype(float)
        Final_out2 = clustered_groupby_max.pivot(index='FAC_NAME', columns='Year', values='Clear_pct').dropna(how='all', axis=1).fillna(0).astype(float)
        Final_out3 = clustered_groupby_max.pivot(index='FAC_NAME', columns='Year', values='Clear_obs').dropna(how='all', axis=1).fillna(0).astype(int)
        Final_out4 = clustered_groupby_mean.pivot(index='FAC_NAME', columns='Year', values='T_avg_K').dropna(how='all', axis=1).fillna(0).astype(int)
        
        for yy in self.years:
            if not yy in Final_out1.columns:
                Final_out1.loc[:, yy], Final_out2.loc[:, yy], Final_out3.loc[:, yy], Final_out4.loc[:, yy] = np.nan, np.nan, np.nan, np.nan
        
        for fa in index:
            if not fa in Final_out1.index:
                Final_out1.loc[fa, :], Final_out2.loc[fa, :], Final_out3.loc[fa, :], Final_out4.loc[fa, :] = np.nan, np.nan, np.nan, np.nan
        
        if saveFilterData: clustered_flares_df.to_csv('./Results/WholeData/Filtered_cleaned_gas.csv')
        self.gas:pd.DataFrame = self.adjust_start_year(Final_out1.loc[index].fillna(0)[self.years])
        self.detection:pd.DataFrame = self.adjust_start_year(Final_out2.loc[index].fillna(0)[self.years])
        self.cleanobservation:pd.DataFrame = self.adjust_start_year(Final_out3.loc[index].fillna(0)[self.years])
        self.temperature:pd.DataFrame = self.adjust_start_year(Final_out4.loc[index].fillna(0)[self.years])
        return

    def findTheEpisodeLength(self, df:pd.DataFrame) -> pd.DataFrame:
        # Ensure datetime is in proper format
        df['Date-time'] = pd.to_datetime(df['Date-time'])
        df['Year_int'] = df['Year'].astype(int)
        df = df.loc[df['Year_int'] < int(self.years[-1])]
        # Sort by FAC_NAME and Date-time
        df = df.sort_values(['FAC_NAME', 'Year', 'Date-time'])
        # Create a column with date only (ignore time)
        df['Date'] = df['Date-time'].dt.date
        # Calculate gap from previous day
        df['prev_date'] = df.groupby('FAC_NAME')['Date'].shift()
        df['gap'] = pd.to_timedelta(df['Date'] - df['prev_date']).dt.days
        # New episode if gap != 1 day
        df['episode_flag'] = (df['gap'] != 1)
        # Create episode_id per FAC_NAME and Year by cumsum within group
        df['episode_id'] = df.groupby(['FAC_NAME', 'Year'])['episode_flag'].cumsum()
        # Group by FAC_NAME and episode_id to find episode lengths
        episode_lengths = df.groupby(['FAC_NAME', 'Year', 'episode_id']).agg(start_date=('Date', 'min'),
                                                                             end_date=('Date', 'max'),
                                                                             length=('Date', 'count'),
                                                                             year_int=('Year_int','first')) #.reset_index()

        return episode_lengths[['start_date', 'year_int', 'length']]
        
    def vnfAnnualProcess(self, index, could_mask:bool=True, Temp_thres:int=1400, bufferSize:int=750, saveFilterData:bool=True):
        # Reading files
        locations = self.extract_location(createBuffer=True, bufferSize=bufferSize)
        otherNearby = self.extract_location(createBuffer=False, fname='./AllOilGasSHP/Nearby_facilities_cleaned')
        otherNearby = otherNearby.loc[otherNearby.CATEGORY != 'Natural gas flaring detections']
        otherNearby = otherNearby.loc[otherNearby.CATEGORY != 'Oil and natural gas wells']
        df = pd.read_csv('./vnf/Extracted.csv')[['Date_Mscan', 'Lon_GMTCO', 'Lat_GMTCO', 'Temp_BB', 'RHI','Cloud_Mask']]

        # Applying filters on temperature and cloudiness
        if could_mask: df = df.loc[np.logical_and(df['Cloud_Mask'] == 0, df['Temp_BB'] > Temp_thres)]
        else: df = df.loc[df['Temp_BB'] > Temp_thres]
        
        # Formatting Date_Mscan column to %Y-%m-%d
        df['Date-time'] = pd.to_datetime(pd.to_datetime(df['Date_Mscan']).dt.strftime('%Y-%m-%d'))
        df['Year'] = df['Date-time'].dt.year.astype(str)
        df.drop(columns=['Date_Mscan'], inplace=True)
        
        # Creating geometry in EPSG:4326
        df['geometry'] = [Point(lon, lat) for lon, lat in zip(df['Lon_GMTCO'], df['Lat_GMTCO'])]
        gdf = gpd.GeoDataFrame(data=df, geometry='geometry', crs='EPSG:4326')
        
        # Filtering VNF records to 750m of the points
        out = gpd.sjoin(locations, gdf, how='right', predicate='contains').dropna(subset='FAC_NAME').drop(columns=['index_left', 'COUNTRY', 'lat', 'lon', 'FAC_NAME']).drop_duplicates().reset_index(drop=True)
        out['utm_epsg'] = out['utm_epsg'].astype(np.int32)
        locations['geometry'] = [Point(lon, lat) for lon, lat in zip(locations['lon'], locations['lat'])]
        
        all_locations = pd.concat([locations, otherNearby]) # locations.copy() #
        # Clustering in appropriate UTM zone
        clustered_flares = []
        for utm_zone in sorted(locations['utm_epsg'].unique()):
            out_utm = out.loc[out.utm_epsg == utm_zone].to_crs(epsg=utm_zone)
            locations_utm = all_locations.loc[all_locations.utm_epsg == utm_zone].to_crs(epsg=utm_zone)
            
            # Build KDTree on facility coordinates ---
            fac_coords = np.vstack([locations_utm.geometry.x, locations_utm.geometry.y]).T
            tree = cKDTree(fac_coords)

            # Clustering using cKDTree. Assign nearest facility for each flare record ---
            flare_coords = np.vstack([out_utm.geometry.x, out_utm.geometry.y]).T
            _, nearest_facility_idx = tree.query(flare_coords, k=1)
            out_utm['facility_index'] = nearest_facility_idx
            out_utm['FAC_NAME'] = locations_utm.iloc[nearest_facility_idx].reset_index(drop=True)['FAC_NAME'].values

            clustered_flares.append(out_utm.to_crs(epsg=4326))
        clustered_flares_df = pd.concat(clustered_flares).reset_index(drop=True)
        if saveFilterData: clustered_flares_df.to_csv('./Results/WholeData/Filtered_cleaned_VNF.csv', index=False)
        self.clustered_flares_df:pd.DataFrame = clustered_flares_df

        # Count daily flares per facility. If more than 1 flares is detected, counted as 1 flare ---
        daily_multipl_stack = clustered_flares_df.loc[clustered_flares_df.groupby(['Date-time', 'Year', 'FAC_NAME', 'facility_index'])['RHI'].idxmax()]
        daily_counts = daily_multipl_stack.groupby(['Date-time', 'Year', 'FAC_NAME']).agg({'RHI':'sum', 'Cloud_Mask':'min', 'Temp_BB':'max'}).reset_index()
        
        # Find the episodes and then create a dataframe for only length of flaring episodes 
        self.flareEpisodes = self.findTheEpisodeLength(daily_counts)
        
        # Count annual flares per facility ---
        annual_counts = daily_counts.groupby(['Year', 'FAC_NAME']).size().reset_index(name='annual_flares')

        # Create a wide summary table (optional) ---
        wide_summary = annual_counts.pivot(index='FAC_NAME', columns='Year', values='annual_flares').dropna(how='all', axis=1).fillna(0).astype(int)

        # Create Correct formated Output
        corrected_flare_counts = pd.DataFrame(index=index, columns=wide_summary.columns)
        for i in corrected_flare_counts.index:
            if i in wide_summary.index: corrected_flare_counts.loc[i] = wide_summary.loc[i]
        
        return self.adjust_start_year(corrected_flare_counts.fillna(0).infer_objects(copy=False)[self.years])
        
    def adjust_start_year(self, inpDf:pd.DataFrame) -> pd.DataFrame:
        for index, year in zip(self.status.index, self.status.StartYear):
            if year > 2012:
                for i in range(2012, year):
                    inpDf.at[index, f'{i}'] = np.nan
        inpDf.columns = self.years
        return inpDf

if __name__ == '__main__':
    vnf = VNF()
