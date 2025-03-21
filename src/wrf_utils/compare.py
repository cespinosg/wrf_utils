import json
import pathlib

import matplotlib.pyplot as plt
from metpy.calc import relative_humidity_from_mixing_ratio
from metpy.units import units
import numpy as np
import pandas as pd
import xarray as xr


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
        sensor = Sensor(self.locations[index]['lat'],
                self.locations[index]['lon'],
                self.locations[index]['code'],
                self.locations[index]['name'],
                csv_file_path)
        return sensor


class Sensor:
    '''
    Stores the data of a sensor.
    '''

    def __init__(self, lat, lon, code, name, csv_file_path):
        self.lat = lat
        self.lon = lon
        self.code = code
        self.name = name
        self.csv_file_path = csv_file_path
        self._read()

    def _read(self):
        '''
        Reads the data.
        '''
        print(f'Reading {self.csv_file_path}')
        df = pd.read_csv(self.csv_file_path, sep=';', decimal=',', parse_dates=['date'])
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
        mask_1 = self.ds['time'] >= start_date
        mask_2 = self.ds['time'] <= end_date
        mask = mask_1*mask_2
        indices = np.where(mask)[0]
        self.ds = self.ds.isel(time=indices)


class Results:
    '''
    Represents the WRF results.
    '''

    def __init__(self, nc_file_path):
        self.nc_file_path = pathlib.Path(nc_file_path)
        self.ds = xr.open_dataset(self.nc_file_path)

    def get_value(self, field, lat, lon, start, end):
        '''
        Returns the values of the given field at the given coordinates.
        '''
        dist = (self.ds['XLAT'][0]-lat)**2+(self.ds['XLONG'][0]-lon)**2
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        mask_1 = self.ds['XTIME'] >= start
        mask_2 = self.ds['XTIME'] <= end
        mask = mask_1*mask_2
        time_indices = np.where(mask)[0]
        return self.ds[field].sel(Time=time_indices, south_north=i, west_east=j)


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
        self.rmse = {
                'name': [s.name for s in self.sensors.list],
                'temp': np.zeros(self.sensors.n_sensors),
                'rh': np.zeros(self.sensors.n_sensors)
                }
        for i in range(self.sensors.n_sensors):
            self._compare_temp(i)
            self._compare_rh(i)
        self._write_rmse()

    def _compare_temp(self, index):
        '''
        Compares the temperature for the given sensor.
        '''
        sensor = self.sensors.list[index]
        wrf = self.wrf.get_value('T2', sensor.lat, sensor.lon, self.start, self.end)
        wrf -= 273.15
        error = wrf.values-sensor.ds['temp'].values
        rmse = np.sqrt(np.mean(error**2))
        self.rmse['temp'][index] = rmse
        title = f'{sensor.code:02} {sensor.name} RMSE = {rmse:.2f} [C]'
        png_file_path = self.folder / f'temp/sensor-{sensor.code:02}-temp.png'
        self.plot(sensor.ds['time'], sensor.ds['temp'], wrf, 'Temperature [C]',
            title, png_file_path)

    def _compare_rh(self, index):
        '''
        Compares the relative humidity for the given sensor.
        '''
        sensor = self.sensors.list[index]
        p = self.wrf.get_value('PSFC', sensor.lat, sensor.lon, self.start, self.end)
        temp = self.wrf.get_value('T2', sensor.lat, sensor.lon, self.start, self.end)
        q2 = self.wrf.get_value('Q2', sensor.lat, sensor.lon, self.start, self.end)
        p = p.values*units.Pa
        temp = temp.values*units.degK
        rh = relative_humidity_from_mixing_ratio(p, temp, q2.values)
        rh = rh.to('percent')
        error = rh.m-sensor.ds['rh'].values
        rmse = np.sqrt(np.mean(error**2))
        self.rmse['rh'][index] = rmse
        title = f'{sensor.code:02} {sensor.name} RMSE = {rmse:.2f} [%]'
        png_file_path = self.folder / f'rh/sensor-{sensor.code:02}-rh.png'
        self.plot(sensor.ds['time'], sensor.ds['rh'], rh,
            'Relative humidity [%]', title, png_file_path)

    def plot(self, time, sensor, wrf, y_label, title, png_file_path):
        '''
        Plots the given field.
        '''
        fig, ax = plt.subplots()
        ax.plot(time, sensor, label='Sensor')
        ax.plot(time, wrf, label='WRF')
        ax.legend()
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
        ax.set_ylabel(y_label)
        ax.set_title(title)
        print(f'Writing {png_file_path}')
        if not png_file_path.parent.is_dir():
            png_file_path.parent.mkdir(parents=True)
        fig.savefig(png_file_path)
        plt.close(fig.number)

    def _write_rmse(self):
        '''
        Writes the RMSE.
        '''
        df = pd.DataFrame(self.rmse)
        csv_file_path = pathlib.Path(f'{self.folder}/rmse/rmse.csv')
        if not csv_file_path.parent.is_dir():
            csv_file_path.parent.mkdir(parents=True)
        print(f'Writing RMSE data to {csv_file_path}')
        df.to_csv(csv_file_path, index=False)
        rmse = RMSE(csv_file_path, self.start, self.end)


class RMSE:
    '''
    Reads and plots the given RMSE.
    '''

    label = {
        'temp': 'Temperature',
        'rh': 'Relative humidity',
    }
    units = {
        'temp': 'C',
        'rh': '%',
    }

    def __init__(self, csv_file_path, start, end):
        self.csv_file_path = pathlib.Path(csv_file_path)
        self.start = start
        self.end = end
        self.name = [p for p in self.csv_file_path.parts if 'bochorno' in p][0]
        self.folder = self.csv_file_path.parent
        self.df = pd.read_csv(self.csv_file_path)
        self._plot('temp')
        self._plot('rh')

    def _plot(self, field):
        '''
        Plots the RMSE of the given field.
        '''
        fig, ax = plt.subplots()
        ax.hist(self.df[field])
        label = self.label[field]
        units = self.units[field]
        ax.set_xlabel(f'RMSE [{units}]')
        mean = np.mean(self.df[field])
        std = np.std(self.df[field])
        start = np.datetime_as_string(self.start, unit='h')
        end = np.datetime_as_string(self.end, unit='h')
        ax.set_title((
            f'{self.name} - {label}\n'
            f'Mean = {mean:.2f} [{units}], '
            f'Standard deviation = {std:.2f} [{units}]\n'
            f'{start} - {end}'
            ))
        fig.savefig(f'{self.folder}/rmse-{field}.png')
        plt.close(fig.number)

