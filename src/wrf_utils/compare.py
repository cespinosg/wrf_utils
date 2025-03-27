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


class Sensors:
    '''
    Stores the sensors data.
    '''

    def __init__(self, folder_path):
        self.folder = pathlib.Path(folder_path)
        self._read()

    def _read(self):
        '''
        Reads the csv file.
        '''
        with open(self.folder / 'locations.json', 'r') as fin:
            self.locations = json.load(fin)
        self.n_sensors = len(self.locations)
        self.csv_files = [f for f in self.folder.iterdir() if f.suffix == '.csv']
        self.csv_files = sorted(self.csv_files)
        self.list = [self._create_sensor(i) for i in range(self.n_sensors)]

    def _create_sensor(self, index):
        '''
        Creates the sensor object that stores the data.
        '''
        csv_file_path = self.csv_files[index]
        sensor = Sensor(self.locations[index]['lon'],
                self.locations[index]['lat'],
                self.locations[index]['code'],
                self.locations[index]['name'],
                csv_file_path)
        return sensor

    def get_field_data(self, field, start, end):
        '''
        Returns the field values measured by all the sensors during the given
        period.
        '''
        for i in range(self.n_sensors):
            self.list[i].filter_by_date(start, end)
        data = np.array([s.ds[field].values for s in self.list])
        return data

    def get_field_mean(self, field, start, end):
        '''
        Calculates the variability for all the sensors at each hour.
        '''
        data = self.get_field_data(field, start, end)
        return np.mean(data, axis=0)

    def get_field_std(self, field, start, end):
        '''
        Calculates the variability for all the sensors at each hour.
        '''
        data = self.get_field_data(field, start, end)
        return np.std(data, axis=0)


class Sensor:
    '''
    Stores the data of a sensor.
    '''

    def __init__(self, lon, lat, code, name, csv_file_path):
        self.lon = lon
        self.lat = lat
        self.code = code
        self.name = name
        self.csv_file_path = csv_file_path
        self._read()

    def _read(self):
        '''
        Reads the data.
        '''
        print(f'Reading {self.csv_file_path}')
        df = pd.read_csv(self.csv_file_path, sep=';', decimal=',',
            parse_dates=['date'])
        self.ds = xr.Dataset(
            {
                'temp': ('time', df['Temp']),
                'rh': ('time', df['Hum']),
            },
            coords = {'time': df['date']},
        )

    def filter_by_date(self, start_date, end_date):
        '''
        Returns the subset of the data that is between the given dates.
        '''
        indices = hf.dates_filter(self.ds['time'], start_date, end_date)
        self.ds = self.ds.isel(time=indices)


class Results:
    '''
    Represents the WRF results.
    '''

    def __init__(self, nc_file_path):
        self.nc_file_path = pathlib.Path(nc_file_path)
        self.name = self.nc_file_path.parent.name
        self.ds = xr.open_dataset(self.nc_file_path)
        self.ds['T2'] -= 273.15

    def get_value(self, field, lon, lat, start, end):
        '''
        Returns the values of the given field at the given coordinates.
        '''
        dist = (self.ds['XLONG'][0]-lon)**2+(self.ds['XLAT'][0]-lat)**2
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        t = hf.dates_filter(self.ds['XTIME'], start, end)
        field_values = self.ds[field].sel(Time=t, south_north=i, west_east=j)
        field_values = field_values.values
        return field_values

    def get_temp(self, lon, lat, start, end):
        '''
        Returns the temperature in the given location during the given period.
        '''
        return self.get_value('T2', lon, lat, start, end)

    def get_rh(self, lon, lat, start, end):
        '''
        Returns the relative humidity in the given location during the given
        period.
        '''
        p = self.get_value('PSFC', lon, lat, start, end)*units.Pa
        temp = self.get_value('T2', lon, lat, start, end)*units.degC
        q2 = self.get_value('Q2', lon, lat, start, end)
        rh = relative_humidity_from_mixing_ratio(p, temp, q2)
        rh = rh.to('percent').m
        return rh

    def get_field_data(self, field, locations, start, end):
        '''
        Returns the field in the given locations during the given period.
        '''
        func = {
            'rh': self.get_rh,
            'temp': self.get_temp,
        }[field]
        data = [func(l['lon'], l['lat'], start, end) for l in locations]
        data = np.array(data)
        return data

    def get_field_mean(self, field, locations, start, end):
        '''
        Returns the standard deviation of the given field among the given
        locations.
        '''
        data = self.get_field_data(field, locations, start, end)
        return np.mean(data, axis=0)

    def get_field_std(self, field, locations, start, end):
        '''
        Returns the standard deviation of the given field among the given
        locations.
        '''
        data = self.get_field_data(field, locations, start, end)
        return np.std(data, axis=0)


LABELS = {
    'temp': 'Temperature',
    'rh': 'Relative humidity',
}
UNITS = {
    'temp': 'C',
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
        self._set_folder()
        self._select_date()
        self._compare()

    def _set_folder(self):
        '''
        Creates the post-processing folder if it does not exist.
        '''
        self.folder = self.wrf.nc_file_path.parent
        if not self.folder.is_dir():
            self.folder.mkdir()

    def _select_date(self):
        '''
        Selects the data by date.
        '''
        self.start = self.wrf.ds['XTIME'][0] if self.start is None else self.start
        self.end = self.wrf.ds['XTIME'][-1] if self.end is None else self.end
        for i in range(self.sensors.n_sensors):
            self.sensors.list[i].filter_by_date(self.start, self.end)

    def _compare(self):
        '''
        Compares all the sensors.
        '''
        self.stats = {
                'name': [s.name for s in self.sensors.list],
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
        sensor = self.sensors.list[index]
        temp_wrf = self.wrf.get_value('T2', sensor.lon, sensor.lat, self.start,
            self.end)
        rmse, bias = compare_series(sensor.ds['temp'].values, temp_wrf)
        self.stats['temp_rmse'][index] = rmse
        self.stats['temp_bias'][index] = bias
        stats = {'rmse': rmse, 'bias': bias}
        self.plot(sensor, 'temp', temp_wrf, stats)

    def _compare_rh(self, index):
        '''
        Compares the relative humidity for the given sensor.
        '''
        sensor = self.sensors.list[index]
        rh_wrf = self.wrf.get_rh(sensor.lon, sensor.lat, self.start, self.end)
        rmse, bias = compare_series(sensor.ds['rh'].values, rh_wrf)
        self.stats['rh_rmse'][index] = rmse
        self.stats['rh_bias'][index] = bias
        stats = {'rmse': rmse, 'bias': bias}
        self.plot(sensor, 'rh', rh_wrf, stats)

    def plot(self, sensor, field, wrf_values, stats):
        '''
        Plots the given field.
        '''
        fig, ax = plt.subplots()
        ax.plot(sensor.ds['time'], sensor.ds[field], label='Sensor')
        ax.plot(sensor.ds['time'], wrf_values, label='WRF')
        ax.legend()
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
        units = UNITS[field]
        ax.set_ylabel(LABELS[field]+' '+units)
        title = ((f'{self.wrf.name} {sensor.code:02}-{sensor.name}\n'
            f'RMSE = {stats["rmse"]:.2f} [{units}], '
            f'bias = {stats["bias"]:.2f} [{units}]'))
        ax.set_title(title)
        file_name = f'{self.wrf.name}-{field}-{sensor.code:02}.png'
        png_file_path = self.folder / 'comparison' / file_name
        hf.save_fig(fig, png_file_path)

    def _write_stats(self):
        '''
        Writes the statistics of the error between the measures and the
        simulation.
        '''
        df = pd.DataFrame(self.stats)
        csv_file_path = f'{self.folder}/stats/{self.wrf.name}-stats.csv'
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
    error = measure-simulation
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
        ax.set_xlabel(f'{var} [{units}]')
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

