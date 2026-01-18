import pathlib

import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_map_utils import north_arrow
import numpy as np
from pyproj import Geod
import xarray as xr


plt.rcdefaults()
plt.rc('lines', markersize=6)
plt.rc('markers', fillstyle='none')
plt.rc('axes', grid=True)
plt.rc('legend', framealpha=0.7)
plt.rc('savefig', dpi=200, bbox='tight')
plt.rc('figure.constrained_layout', use=True)
plt.close('all')


class UrbanPlotter:
    '''
    Plots the urban parameters.
    '''

    def __init__(self, ds, label, field, units, time_index, offset,
            image_resolution=13, scale_bar_length=1000, levels=None):
        self.ds = ds
        self.label = label
        self.field = field
        self.units = units
        self.time_index = time_index
        self.offset = offset
        self.image_resolution = image_resolution
        self.scale_bar_length = scale_bar_length
        self.levels = levels
        self._plot()

    def plot(self):
        '''
        Plots the given field.
        '''
        self._init_plot()
        z = self.ds[self.field]
        if 'Time' in self.ds.coords.sizes:
            lon = self.ds['XLONG'][self.time_index]
            lat = self.ds['XLAT'][self.time_index]
            z = z[time_index]
        else:
            lon = self.ds['XLONG']
            lat = self.ds['XLAT']
        if levels is None:        
            levels = np.linspace(z.min(), z.max(), 11)
        norm = mcolors.BoundaryNorm(levels, ncolors=plt.cm.viridis.N)
        cs = self.ax.pcolormesh(lon, lat, z,
            shading='nearest', transform=ccrs.PlateCarree(), alpha=0.5,
            cmap='coolwarm', norm=norm)
        # self.ax.scatter(self.ds['XLONG'], self.ds['XLAT'],
        #     transform=ccrs.PlateCarree())
        self.ax.scatter([-0.9491011503598326], [41.59621897116297],
            marker='x', transform=ccrs.PlateCarree())
        self.fig.colorbar(cs, location='bottom',
            label=f'{self.field} [{units}]', shrink=0.75)
        self._add_scalebar()
        self._set_title()
        self.fig.show()
        self._write()

    def _init_plot(self):
        '''
        Initialises the plot.
        '''
        self.fig = plt.figure(figsize=(7, 7))
        imagery = OSM(cache=True)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=imagery.crs)
        north_arrow(
            self.ax, location="upper left", scale=0.25,
            rotation={"crs": self.ax.projection, "reference": "center"}
        )
        self._set_extent()
        self.ax.add_image(imagery, self.image_resolution)

    def _set_extent(self):
        '''
        Sets the plot extent.
        '''
        n = len(self.ds.south_north)
        start = int(n/2)-self.offset
        end = int(n/2)+self.offset-1
        self.ds = self.ds.isel(south_north=slice(start, end),
            west_east=slice(start, end))
        if 'Time' not in self.ds.coords.sizes:
            lon = self.ds['XLONG'].values[0]
            lat = self.ds['XLAT'].values[:, 0]
        else:
            lon = self.ds['XLONG'].values[0, 0]
            lat = self.ds['XLAT'].values[0, :, 0]
        bounds = (
            lon[0]-0.5*(lon[1]-lon[0]),
            lon[-1]+0.5*(lon[-1]-lon[-2]),
            lat[0]-0.5*(lat[1]-lat[0]),
            lat[-1]+0.5*(lat[-1]-lat[-2])
        )
        self.ax.set_extent(
            bounds,
            crs=ccrs.Geodetic()
        )

    def _add_scalebar(self, location=(0.9, 0.95)):
        '''
        Draws a scale bar on the map in axes-relative coordinates.
        
        Parameters
        ----------
        ax : matplotlib axes with a cartopy projection
        length : float
            Length of the scale bar in meters
        location : tuple (float, float)
            Center of the scale bar in axes fraction coordinates (0–1)
        linewidth : int
            Line width of the scale bar
        color : str
            Color of the scale bar
        '''
        extent = self.ax.get_extent(ccrs.PlateCarree())
        lon_centre = extent[0] + (extent[1] - extent[0]) * location[0]
        lat_centre = extent[2] + (extent[3] - extent[2]) * location[1]
        geod = Geod(ellps="WGS84")
        half = 0.5*self.scale_bar_length
        lon1, lat1, _ = geod.fwd(lon_centre, lat_centre, 90, half)
        lon2, lat2, _ = geod.fwd(lon_centre, lat_centre, 270, half)
        delta_lon = lon1-lon2
        rectangle = mpatches.Rectangle((lon2-0.1*delta_lon, lat2-0.1*delta_lon),
            1.2*delta_lon, 0.3*delta_lon, facecolor='lightgrey',
            transform=ccrs.PlateCarree())
        self.ax.add_patch(rectangle)
        self.ax.plot([lon1, lon2], [lat1, lat2],
                transform=ccrs.PlateCarree(), color='black', linewidth=3)
        self.ax.text(lon_centre, lat_centre, f'{self.scale_bar_length} m',
            transform=ccrs.PlateCarree(), ha='center', va='bottom')

    def _set_title(self):
        '''
        Sets the title.
        '''
        title = self.label
        if 'Time' in self.ds.coords.sizes:
            time = self.ds['XTIME'].values[self.time_index]
            time = np.datetime64(time, 's')
            time = str(time).replace('T', ' ')
            title += f'\n{time}'
        self.ax.set_title(title)

    def _write(self):
        '''
        Writes the plot.
        '''
        png_file_path = self.label+f'-{self.field}-{self.time_index}.png'
        print(f'Writing {png_file_path}')
        self.fig.savefig(png_file_path)

