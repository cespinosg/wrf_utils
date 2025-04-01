from concurrent.futures import ProcessPoolExecutor
import pathlib
import time

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


plt.rcdefaults()
plt.rc('lines', markersize=6)
plt.rc('markers', fillstyle='none')
plt.rc('axes', grid=True)
plt.rc('legend', framealpha=0.7)
plt.rc('savefig', dpi=200, bbox='tight')
plt.rc('figure.constrained_layout', use=True)
plt.close('all')


class SurfacePlotter:
    '''
    Plots surface data.
    '''

    def __init__(self, nc_file_path):
        self.nc_file_path = pathlib.Path(nc_file_path)
        self._read()
        self._set_folder()

    def _read(self):
        '''
        Reads the nc file.
        '''
        self.ds = xr.open_dataset(self.nc_file_path)
        self.times = self.ds['XTIME'].values.astype('<M8[h]')
        self.ds['wind'] = np.sqrt(self.ds['U10']**2+self.ds['V10']**2)
        self.ds['Q2'] *= 1e3
        self.ds['T2'] -= 273.15

    def _set_folder(self):
        '''
        Sets the folder where the images will be saved.
        '''
        self.folder = self.nc_file_path.parent / self.nc_file_path.stem
        if not self.folder.is_dir():
            self.folder.mkdir()

    def plot(self, start_time=None, end_time=None):
        '''
        Plots all the dates.
        '''
        self.start_time = start_time
        self.end_time = end_time
        self._set_indices()
        start = time.perf_counter()
        for i in self.indices:
            self._plot_time(i)
        duration = time.perf_counter()-start
        print(f'--> Spent {duration:.2f} [s] plotting all the dates sequentially')

    def plot_parallel(self, start_time=None, end_time=None):
        '''
        Plots all the dates in parallel.
        https://realpython.com/python-concurrency
        '''
        self.start_time = start_time
        self.end_time = end_time
        self._set_indices()
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=4) as executor:
            executor.map(self._plot_time, self.indices)
        duration = time.perf_counter()-start
        print(f'--> Spent {duration:.2f} [s] plotting all the dates in parallel')

    def _set_indices(self):
        '''
        Sets the indices of the times to plot.
        '''
        if self.start_time is None:
            self.start_time = self.ds['XTIME'].values[0]
        if self.end_time is None:
            self.end_time = self.ds['XTIME'].values[-1]
        start_condition = self.ds['XTIME'] >= self.start_time
        end_condition = self.ds['XTIME'] <= self.end_time
        mask = np.logical_and(start_condition, end_condition)
        self.indices = np.where(mask)[0]

    def _plot_time(self, time_index):
        '''
        Plots the wind field.
        '''
        self.time_index = time_index
        self._init_plot()
        self._plot_data()
        self._set_title()
        self._write()

    def _init_plot(self):
        '''
        Initialises the plot.
        '''
        self.fig = plt.figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
        self.ax.coastlines()
        self.ax.gridlines(draw_labels=True, dms=True)
        self._set_colormap()

    def _set_colormap(self):
        '''
        Sets the colormap used for plotting.
        https://discourse.matplotlib.org/t/single-color-transparent-colormap/18954/6
        https://stackoverflow.com/a/51601850
        https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.LinearSegmentedColormap.html
        '''
        cm_dict = {
            'red': [
                (0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0),
                ],
            'green':[
                (0.0, 1.0, 1.0),
                (1.0, 0.0, 0.0),
                ],
            'blue': [
                (0.0, 1.0, 1.0),
                (1.0, 1.0, 1.0),
                ],
            'alpha': [
                (0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0),
                ],
           }
        self.my_cm = LinearSegmentedColormap('my_cm', cm_dict)

    def _set_title(self):
        '''
        Sets the plot title.
        '''
        self.time = self.times[self.time_index].item()
        self.ax.set_title(self.time.strftime('%d/%m/%Y %H UTC'))

    def _write(self):
        '''
        Writes the png file with the plot.
        '''
        time_stamp = self.time.strftime('%y-%m-%d-%H')
        png_file_path = self.folder / f'{self.field}-{time_stamp}.png'
        print(f'Writing {png_file_path}')
        self.fig.savefig(png_file_path)
        plt.close(self.fig.number)


class CityPlotter(SurfacePlotter):
    '''
    Plots the surface data in the city. An OSM is added to the background.
    '''

    def _init_plot(self):
        '''
        Initialises the plot.
        https://scitools.org.uk/cartopy/docs/latest/gallery/scalar_data/eyja_volcano.html
        '''
        self.fig = plt.figure(figsize=(7, 7))
        imagery = OSM(cache=True)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=imagery.crs)
        self.ax.set_extent((self.ds['XLONG'][0].min(), self.ds['XLONG'].max(),
            self.ds['XLAT'][0].min(), self.ds['XLAT'][0].max()),
            crs=ccrs.Geodetic())
        self.ax.add_image(imagery, 10)
        self.ax.gridlines(draw_labels=True, dms=True)
        self._set_colormap()


class Wind(SurfacePlotter):
    '''
    Plots the wind distribution over the simulation domain.
    '''

    field = 'wind-10m'
    n_regrid = 50

    def _plot_data(self):
        '''
        Plots the wind speed and direction.
        '''
        self._plot_wind_speed()
        self._plot_wind_direction()

    def _plot_wind_speed(self):
        '''
        Plots the wind speed magnitude in a contour filled plot.
        '''
        levels = np.linspace(0, 10, 11)
        cs = self.ax.contourf(self.ds['XLONG'][0], self.ds['XLAT'][0],
            self.ds['wind'][self.time_index],
            transform=ccrs.PlateCarree(), levels=levels, extend='max',
            cmap=self.my_cm)
        self.fig.colorbar(cs, location='bottom', label='Wind speed [m/s] at 10 m',
            shrink=0.75)

    def _plot_wind_direction(self):
        '''
        Plots the wind direction.
        https://scitools.org.uk/cartopy/docs/latest/gallery/vector_data/regridding_arrows.html
        '''
        self.ax.quiver(self.ds['XLONG'][0].values,
            self.ds['XLAT'][0].values,
            self.ds['U10'][self.time_index].values,
            self.ds['V10'][self.time_index].values,
            transform=ccrs.PlateCarree(),
            angles='xy',
            regrid_shape=self.n_regrid)


class WaterVapour(SurfacePlotter):
    '''
    Plots the water vapour.
    '''

    field = 'q2'

    def _plot_data(self):
        '''
        Plots the water vapour content.
        '''
        levels = np.linspace(0, 20, 9)
        self._set_colormap()
        cs = self.ax.contourf(self.ds['XLONG'][0], self.ds['XLAT'][0],
            self.ds['Q2'][self.time_index],
            transform=ccrs.PlateCarree(), levels=levels, extend='max',
            cmap=self.my_cm)
        self.fig.colorbar(cs, location='bottom', label='Specific humidity [g/kg] at 2 m',
            shrink=0.75)


class Temperature(SurfacePlotter):
    '''
    Plots the temperature.
    '''

    field = 't2'

    def _plot_data(self):
        '''
        Plots the temperature.
        '''
        levels = np.linspace(15, 45, 31)
        self._set_colormap()
        cs = self.ax.contourf(self.ds['XLONG'][0], self.ds['XLAT'][0],
            self.ds['T2'][self.time_index], transform=ccrs.PlateCarree(),
            levels=levels, extend='both', cmap='coolwarm', alpha=0.5)
        self.fig.colorbar(cs, location='bottom', label='Temperature [C] at 2 m',
            shrink=0.75)


class CityWind(CityPlotter, Wind):
    '''
    Plots the wind in the city. An OSM is added to the background.
    '''

    n_regrid = 15


class CityWaterVapour(CityPlotter, WaterVapour):
    '''
    Plots the water vapour in the city.
    '''


class CityTemperature(CityPlotter, Temperature):
    '''
    Plots the temperature in the city.
    '''


def plot_data(nc_file_path, start, end, plotters):
    '''
    Plots the given data.
    '''
    for plotter in plotters:
        my_plotter = plotter(nc_file_path)
        my_plotter.plot_parallel(start_time=start, end_time=end)

