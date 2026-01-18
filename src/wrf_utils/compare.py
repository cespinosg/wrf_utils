import json
import pathlib

import matplotlib.pyplot as plt
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr

from wrf_utils import helper_functions as hf


plt.rcdefaults()
plt.rc('lines', markersize=6)
plt.rc('markers', fillstyle='none')
plt.rc('axes', grid=True)
plt.rc('legend', framealpha=0.7)
plt.rc('savefig', dpi=200, bbox='tight')
plt.rc('figure.constrained_layout', use=True)
plt.close('all')


class Results:
    '''
    Base class for all the other classes that contain results.
    '''

    def filter_by_date(self, start=None, end=None):
        '''
        Filters the DataFrame by date.
        '''
        self.df = filter_df_by_date(self.df, 't', start, end)

    def get_field_data(self, field, start=None, end=None):
        '''
        Returns the field values measured by all the sensors during the given
        period.
        '''
        df = filter_df_by_date(self.df, 't', start, end)
        columns = [f's{s+1:02}_{field}' for s in range(self.n_sensors)]
        data = df[columns].values
        return data

    def get_field_mean(self, field, start=None, end=None):
        '''
        Calculates the variability for all the sensors at each hour.
        '''
        data = self.get_field_data(field, start, end)
        return np.mean(data, axis=1)

    def get_field_std(self, field, start=None, end=None):
        '''
        Calculates the variability for all the sensors at each hour.
        '''
        data = self.get_field_data(field, start, end)
        return np.std(data, axis=1)


def filter_df_by_date(df, time_key, start=None, end=None):
    '''
    Filters the given DataFrame by date.
    '''
    if start is not None:
        mask = df[time_key] >= start
        df = df[mask]
    if end is not None:
        mask = df[time_key] <= end
        df = df[mask]
    return df


class Sensors(Results):
    '''
    Stores the sensors data.
    '''

    def __init__(self, folder_path):
        self.folder = pathlib.Path(folder_path)
        self._read()
        self._gather_sensors_data()

    def _read(self):
        '''
        Reads the json file.
        '''
        with open(self.folder / 'locations.json', 'r') as fin:
            self.locations = json.load(fin)
        self.n_sensors = len(self.locations)
        self.lats = [l['lat'] for l in self.locations]
        self.lons = [l['lon'] for l in self.locations]

    def _gather_sensors_data(self):
        '''
        Gathers data from all the sensors.
        '''
        self.csv_files = [f for f in self.folder.iterdir() if f.suffix == '.csv']
        self.csv_files = sorted(self.csv_files)
        self.df = pd.DataFrame()
        for i in range(self.n_sensors):
            self._read_sensor(i)

    def _read_sensor(self, index):
        '''
        Reads the given sensor data.
        '''
        csv_file_path = self.csv_files[index]
        print(f'Reading {csv_file_path}')
        df = pd.read_csv(csv_file_path, sep=';', decimal=',',
            parse_dates=['date'])
        if 't' not in self.df.columns:
            self.df['t'] = df['date']
        self.df[f's{index+1:02}_tc'] = df['Temp']
        self.df[f's{index+1:02}_rh'] = df['Hum']


class WRFSurface(Results):
    '''
    Represents the WRF surface results.
    '''

    def __init__(self, nc_file_path, lats, lons):
        self.nc_file_path = pathlib.Path(nc_file_path)
        self.lats = lats
        self.lons = lons
        self.folder = self.nc_file_path.parent
        self.name = self.nc_file_path.parent.name
        self.ds = xr.open_dataset(self.nc_file_path)
        self.ds['T2'] -= 273.15
        self._interpolate_data_at_sensors_locations()

    def _interpolate_data_at_sensors_locations(self):
        '''
        Interpolates the data at the sensors locations.
        '''
        self.df = pd.DataFrame({'t': self.ds['XTIME'].values})
        for i in range(len(self.lats)):
            self.df[f's{i+1:02}_tc'] = self.get_temp(self.lons[i], self.lats[i])
            self.df[f's{i+1:02}_rh'] = self.get_rh(self.lons[i], self.lats[i])

    def get_temp(self, lon, lat):
        '''
        Returns the temperature in the given location during the given period.
        '''
        return self.get_value('T2', lon, lat)

    def get_rh(self, lon, lat):
        '''
        Returns the relative humidity in the given location during the given
        period.
        '''
        p = self.get_value('PSFC', lon, lat)*units.Pa
        temp = self.get_value('T2', lon, lat)*units.degC
        q2 = self.get_value('Q2', lon, lat)
        rh = relative_humidity_from_mixing_ratio(p, temp, q2)
        rh = rh.to('percent').m
        return rh

    def get_value(self, field, lon, lat):
        '''
        Returns the values of the given field at the given coordinates.
        '''
        dist = (self.ds['XLONG'][0]-lon)**2+(self.ds['XLAT'][0]-lat)**2
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        field_values = self.ds[field].sel(south_north=i, west_east=j)
        field_values = field_values.values
        return field_values


class WRFSensors(Results):
    '''
    Contains the WRF results at the sensors locations.
    '''

    def __init__(self, folder):
        self.folder = pathlib.Path(folder)
        self.name = self.folder.name
        self._read_csvs()

    def _read_csvs(self):
        '''
        Reads the csv files.
        '''
        self._read_tc_df()
        self._read_rh_df()
        self._merge_dfs()

    def _read_tc_df(self):
        '''
        Reads the csv file with the temperature data.
        '''
        self.tc_csv_fp = f'{self.folder}/{self.name}-sensors-tc.csv'
        print(f'Reading {self.tc_csv_fp}')
        self.tc_df = pd.read_csv(self.tc_csv_fp, parse_dates=['t'])

    def _read_rh_df(self):
        '''
        Reads the csv file with the relative humidity data.
        '''
        self.rh_csv_fp = f'{self.folder}/{self.name}-sensors-rh.csv'
        print(f'Reading {self.rh_csv_fp}')
        self.rh_df = pd.read_csv(self.rh_csv_fp, parse_dates=['t'])

    def _merge_dfs(self):
        '''
        Merges the temperature and relative humidity DataFrames.
        '''
        self.n_sensors = len(self.tc_df.columns)-1
        self.df = pd.DataFrame({'t': self.tc_df['t']})
        for s in range(1, self.n_sensors+1):
            self.df[f's{s:02}_tc'] = self.tc_df[f's{s:02}']
            self.df[f's{s:02}_rh'] = self.rh_df[f's{s:02}']


LABELS = {
    'temp': 'Temperature',
    'rh': 'Relative humidity',
}
UNITS = {
    'temp': 'ºC',
    'rh': '%',
}


class Comparator:
    '''
    Compares the sensor data with WRF results.
    '''

    def __init__(self, sensors, wrf, start=None, end=None):
        self.sensors = sensors
        self.wrf = wrf
        self.start = start
        self.end = end
        self._select_date()
        self._compare()

    def _select_date(self):
        '''
        Selects the data by date.
        '''
        if self.start is None:
            self.start = self.wrf.df['t'].iloc[0]
        if self.end is None:
            self.end = self.wrf.ds['t'].iloc[-1]
        self.sensors.filter_by_date(self.start, self.end)
        self.wrf.filter_by_date(self.start, self.end)
        self.t = self.sensors.df['t'].values

    def _compare(self):
        '''
        Compares all the sensors.
        '''
        self.stats = {
                'name': [s['name'] for s in self.sensors.locations],
                'temp_rmse': np.zeros(self.sensors.n_sensors),
                'temp_bias': np.zeros(self.sensors.n_sensors),
                'rh_rmse': np.zeros(self.sensors.n_sensors),
                'rh_bias': np.zeros(self.sensors.n_sensors),
                }
        for i in range(self.sensors.n_sensors):
            self._compare_temp(i)
            self._compare_rh(i)
        self._write_stats()

    def _compare_temp(self, index):
        '''
        Compares the temperature for the given sensor.
        '''
        sensor_tc = self.sensors.df[f's{index+1:02}_tc'].values
        wrf_tc = self.wrf.df[f's{index+1:02}_tc'].values
        rmse, bias = compare_series(sensor_tc, wrf_tc)
        self.stats['temp_rmse'][index] = rmse
        self.stats['temp_bias'][index] = bias
        stats = {'rmse': rmse, 'bias': bias}
        # self.plot('temp', sensor_tc, wrf_tc, stats, index)

    def _compare_rh(self, index):
        '''
        Compares the relative humidity for the given sensor.
        '''
        sensor_rh = self.sensors.df[f's{index+1:02}_rh'].values
        wrf_rh = self.wrf.df[f's{index+1:02}_rh'].values
        rmse, bias = compare_series(sensor_rh, wrf_rh)
        self.stats['rh_rmse'][index] = rmse
        self.stats['rh_bias'][index] = bias
        stats = {'rmse': rmse, 'bias': bias}
        # self.plot('rh', sensor_rh, wrf_rh, stats, index)

    def plot(self, field, sensor_data, wrf_data, stats, index):
        '''
        Plots the given field of the given sensor index.
        '''
        fig, ax = plt.subplots()
        ax.plot(self.t, sensor_data, label='Sensor')
        ax.plot(self.t, wrf_data, label='WRF')
        ax.legend()
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
        units = UNITS[field]
        ax.set_ylabel(LABELS[field]+f' [{units}]')
        code = self.sensors.locations[index]['code']
        name = self.sensors.locations[index]['name']
        title = ((f'{self.wrf.name} {code:02}-{name}\n'
            f'RMSE = {stats["rmse"]:.2f} [{units}], '
            f'bias = {stats["bias"]:.2f} [{units}]'))
        ax.set_title(title)
        file_name = f'{self.wrf.name}-{field}-{code:02}.png'
        png_file_path = self.wrf.folder / 'comparison' / file_name
        hf.save_fig(fig, png_file_path)

    def _write_stats(self):
        '''
        Writes the statistics of the error between the measures and the
        simulation.
        '''
        df = pd.DataFrame(self.stats)
        csv_file_path = f'{self.wrf.folder}/stats/{self.wrf.name}-stats.csv'
        csv_file_path = pathlib.Path(csv_file_path)
        if not csv_file_path.parent.is_dir():
            csv_file_path.parent.mkdir(parents=True)
        print(f'Writing statistics to {csv_file_path}')
        df.to_csv(csv_file_path, index=False)
        stats = Stats(csv_file_path, self.start, self.end, self.wrf.name)


def compare_series(measure, simulation):
    '''
    Compares the given time series.
    '''
    error = simulation-measure
    rmse = np.sqrt(np.mean(error**2))
    bias = np.mean(error)
    return rmse, bias


class Stats:
    '''
    Reads and plots the given statistics.
    '''

    def __init__(self, csv_file_path, start, end, name):
        self.csv_file_path = pathlib.Path(csv_file_path)
        self.start = start
        self.end = end
        self.name = name
        self.folder = self.csv_file_path.parent
        self.df = pd.read_csv(self.csv_file_path)
        self._plot('rmse', 'temp')
        self._plot('bias', 'temp')
        self._plot('rmse', 'rh')
        self._plot('bias', 'rh')

    def _plot(self, var, field):
        '''
        Plots the RMSE of the given field.
        '''
        fig, ax = plt.subplots()
        ax.hist(self.df[f'{field}_{var}'])
        label = LABELS[field]
        units = UNITS[field]
        name = {
            'rmse': 'RMSE',
            'bias': 'Bias',
        }[var]
        ax.set_xlabel(f'{name} [{units}]')
        mean = np.mean(self.df[f'{field}_{var}'])
        std = np.std(self.df[f'{field}_{var}'])
        start = np.datetime_as_string(self.start, unit='h')
        end = np.datetime_as_string(self.end, unit='h')
        ax.set_title((
            f'{self.name} - {label}\n'
            f'Mean = {mean:.2f} [{units}], '
            f'Standard deviation = {std:.2f} [{units}]\n'
            f'{start} - {end}'
            ))
        fig.savefig(f'{self.folder}/{self.name}-{field}-{var}.png')
        plt.close(fig.number)

