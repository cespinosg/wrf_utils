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

    def __init__(self, csv_file_path):
        self.csv_file_path = pathlib.Path(csv_file_path)
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
        ax.set_title((
            f'{self.name} - {label}\n'
            f'Mean = {mean:.2f} [{units}], '
            f'Standard deviation = {std:.2f} [{units}]\n'
            ))
        fig.savefig(f'{self.folder}/rmse-{field}.png')
        plt.close(fig.number)


if __name__ == '__main__':
    # rmse = RMSE('bochorno-v01/rmse/rmse.csv')
    rmse = RMSE('bochorno-v02-01/rmse/rmse.csv')
    # rmse = RMSE('bochorno-v02-02/rmse/rmse.csv')
    # rmse = RMSE('bochorno-v03-01/rmse/rmse.csv')

