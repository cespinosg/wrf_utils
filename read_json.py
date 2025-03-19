import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcdefaults()
plt.rc('lines', markersize=6)
plt.rc('markers', fillstyle='none')
plt.rc('axes', grid=True)
plt.rc('legend', framealpha=0.7)
plt.rc('savefig', dpi=200, bbox='tight')
plt.rc('figure.constrained_layout', use=True)
plt.close('all')


class Reader:
    '''
    Reads the given json file.
    '''

    def __init__(self, json_file_path):
        self.json_file_path = pathlib.Path(json_file_path)
        self.name = self.json_file_path.stem
        self.folder = self.json_file_path.parent
        self._read()
        self._convert_data_to_np()
        self._convert_temp_to_celsius()
        self._find_hottest_time()

    def _read(self):
        '''
        Reads the json file.
        '''
        print(f'Reading {self.json_file_path}')
        with open(self.json_file_path, 'r') as fin:
            self.data = json.load(fin)

    def _convert_data_to_np(self):
        '''
        Converts the data to numpy arrays.
        '''
        self.data['time'] = [np.datetime64(t, 'ns') for t in self.data['time']]
        self.data['time'] = np.array(self.data['time'])
        for i in range(len(self.data['sensors'])):
            for (key, value) in self.data['sensors'][i].items():
                self.data['sensors'][i][key] = np.array(value)

    def _convert_temp_to_celsius(self):
        '''
        Converts the temperature to Celsius.
        '''
        for i in range(len(self.data['sensors'])):
            self.data['sensors'][i]['T2'] -= 273.15

    def _find_hottest_time(self):
        '''
        Finds the hottest time for all the sensors.
        '''
        indices = [s['COMF_90'].argmax() for s in self.data['sensors']]
        self.comfort = pd.DataFrame({
            'sensor': [s['name'] for s in self.data['sensors']],
            'code': [s['code'] for s in self.data['sensors']],
            'time': [self.data['time'][i] for i in indices],
            'comf_90': [s['COMF_90'].max() for s in self.data['sensors']],
        })
        self.comfort.sort_values('comf_90', ascending=False, ignore_index=True,
            inplace=True)

    def plot(self, field, sensor_index):
        '''
        Plots the given field and sensor.
        '''
        fig, ax = plt.subplots()
        sensor_data = self.data['sensors'][sensor_index]
        ax.plot(self.data['time'], sensor_data[field])
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
        ax.set_ylabel(field)
        ax.set_title(f'{sensor_data["code"]} - {sensor_data["name"]}')
        fig.show()


class Comparator:
    '''
    Compares the given results.
    '''

    def __init__(self, results):
        self.results = results

    def plot(self, sensor_index, field):
        '''
        Plots the data for the given sensor and field.
        '''
        fig, ax = plt.subplots()
        for r in self.results:
            ax.plot(r.data['time'], r.data['sensors'][sensor_index][field],
                label=r.json_file_path.stem)
        ax.legend()
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
        ax.set_ylabel(field)
        sensor_data = self.results[0].data["sensors"][sensor_index]
        ax.set_title(f'{sensor_data["code"]} - {sensor_data["name"]}')
        fig.show()

    def plot_diff(self, sensor_index, field):
        '''
        Plots the data for the given sensor and field.
        '''
        fig, ax = plt.subplots()
        diff = self.results[0].data['sensors'][sensor_index][field]-\
            self.results[1].data['sensors'][sensor_index][field]
        max_diff_id = abs(diff).argmax()
        print(f'Maximum difference at {self.results[0].data["time"][max_diff_id]}')
        ax.plot(self.results[0].data['time'], diff)
        ax.set_xlabel('Time')
        ax.tick_params(axis='x', labelrotation=45)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment('right')
        ax.set_ylabel(field+' difference')
        sensor_data = self.results[0].data["sensors"][sensor_index]
        ax.set_title((f'{sensor_data["code"]} - {sensor_data["name"]}\n'
            f'{self.results[0].json_file_path.stem} - {self.results[1].json_file_path.stem}'))
        fig.show()


if __name__ == '__main__':
    results = [
        # 'bochorno-v01/bochorno-v01.json',
        # 'bochorno-v02-01/sensors/bochorno-v02-01.json',
        # 'bochorno-v02-02/bochorno-v02-02.json',
        'bochorno-v03-01/bochorno-v03-01.json',
    ]
    readers = [Reader(r) for r in results]
    [print(f'{r.name}\n{r.comfort}\n') for r in readers]
    # comparator = Comparator(readers)
    # comparator.plot(0, 'T2')
    # comparator.plot_diff(0, 'T2')
    # comparator.plot(0, 'Q2')
    # comparator.plot_diff(0, 'Q2')
    # comparator.plot(0, 'PSFC')
    # comparator.plot_diff(0, 'PSFC')

