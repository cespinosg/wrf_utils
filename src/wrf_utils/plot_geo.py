import pathlib

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Proj
import xarray as xr


plt.rcdefaults()
plt.rc('lines', markersize=6)
plt.rc('markers', fillstyle='none')
plt.rc('axes', grid=True)
plt.rc('legend', framealpha=0.7)
plt.rc('savefig', dpi=200, bbox='tight')
plt.rc('figure.constrained_layout', use=True)
plt.close('all')


class GeoPlotter:
    '''
    Plots the data in the given geo_em file.
    '''

    def __init__(self, geo_em_file_path, city=False):
        self.file_path = pathlib.Path(geo_em_file_path)
        self.city = city
        self._read()

    def _read(self):
        '''
        Reads the data in the geo_em file.
        '''
        self.ds = xr.open_dataset(self.file_path)

    def get_value(self, field, lat, lon):
        '''
        Returns the values of the given field at the given coordinates.
        '''
        dist = (self.ds['XLAT_M'][0]-lat)**2+(self.ds['XLONG_M'][0]-lon)**2
        i, j = np.unravel_index(np.argmin(dist.values), dist.values.shape)
        return self.ds[field].sel(south_north=i, west_east=j).values.tolist()

    def plot(self, field, index=None):
        '''
        Plots the given field.
        '''
        self.field = field
        self.index = index
        self._init_plot()
        if self.index is None:
            z = self.ds[self.field][0]
        else:
            z = self.ds[self.field][0, self.index]
        cs = self.ax.contourf(self.ds['XLONG_M'][0], self.ds['XLAT_M'][0], z,
            transform=ccrs.PlateCarree(), cmap=self.my_cm)
        description = self.ds[self.field].description
        units = self.ds[self.field].units
        self.fig.colorbar(cs, location='bottom',
            label=f'{description} [{units}]', shrink=0.75)
        # self._zoom_in()
        self._write()

    def _init_plot(self):
        '''
        Initialises the plot.
        '''        
        self.fig = plt.figure(figsize=(7, 7))
        if self.city:
            self._init_plot_city()
        else:
            self._init_plot_map()
        self._set_colormap()
        self.ax.gridlines(draw_labels=True, dms=True)

    def _init_plot_map(self):
        '''
        Initialises the map plot.
        '''
        self.ax = self.fig.add_subplot(1, 1, 1, projection=ccrs.EqualEarth())
        self.ax.coastlines()

    def _init_plot_city(self):
        '''
        Initialises the city plot.
        '''
        imagery = OSM(cache=True)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=imagery.crs)
        self.ax.set_extent(
            (
                self.ds['XLONG_C'][0].min(), self.ds['XLONG_C'][0].max(),
                # 41.4479, 42.0587
                41.4479, 42.01
                # self.ds['XLAT_C'][0].min(), self.ds['XLAT_C'][0].max()
            ),
            crs=ccrs.Geodetic()
        )
        self.ax.add_image(imagery, 10)

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
            'green': [
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

    def _zoom_in(self):
        '''
        Zooms in.
        '''
        left, right = self.ax.get_xlim()
        delta_x = right-left
        self.ax.set_xlim(left+0.45*delta_x, right-0.45*delta_x)
        bottom, top = self.ax.get_ylim()
        delta_y = top-bottom
        self.ax.set_ylim(bottom+0.45*delta_y, top-0.45*delta_y)

    def _write(self):
        '''
        Writes the plot.
        '''
        folder = self.file_path.parent
        if self.index is None:
            name = self.file_path.stem+f'-{self.field}.pdf'
        else:
            name = self.file_path.stem+f'-{self.field}-{self.index}.pdf'
        png_file_path = folder / name
        print(f'Writing {png_file_path}')
        self.fig.savefig(png_file_path)


def get_field_at_sensors(plotter, sensors, field):
    '''
    Prints the given field at the sensor locations.
    '''
    print(f'Printing {field} for all sensors')
    for s in sensors.locations:
        value = plotter.get_value(field, s['lat'], s['lon'])
        print(f'{s["code"]} {s["name"]} {value[0]}')


class TopoPlotter:
    '''
    Plots the topography.
    '''

    def __init__(self, nc_file_path):
        self.nc_file_path = nc_file_path
        self.ds = xr.open_dataset(self.nc_file_path)
        self._set_coords()
        self._plot()

    def _set_coords(self):
        '''
        Sets the coordinates for plotting.
        '''
        self.p = Proj('urn:ogc:def:crs:EPSG::25830')
        self._set_centre()
        self.x, self.y = self.p(self.ds['XLONG_M'][0], self.ds['XLAT_M'][0])
        self.x -= self.x_c
        self.y -= self.y_c
        self.x *= 1e-3
        self.y *= 1e-3

    def _set_centre(self):
        '''
        Sets the centre of the map.
        '''
        lat_c, lon_c = 41.65606, -0.87734
        self.x_c, self.y_c = self.p(lon_c, lat_c)

    def _plot(self):
        '''
        Plots the topography.
        '''
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(self.x, self.y, self.ds['HGT_M'][0])
        # ax.set_aspect('equal')
        fig.show()

