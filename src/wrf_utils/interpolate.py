import numpy as np
import pandas as pd


class Interpolator:
    '''
    Interpolates the dataset at the sensor locations.
    '''

    def __init__(self, sensors, ds, lat_key, lon_key):
        self.sensors = sensors
        self.ds = ds
        self.lat_key = lat_key
        self.lon_key = lon_key
        self.n_sensors = len(self.sensors.locations)
        self._find_indices()

    def _find_indices(self):
        '''
        Finds the indices that correspond to each sensor.
        '''
        self.i = [0 for i in range(self.n_sensors)]
        self.j = [0 for i in range(self.n_sensors)]
        for (k, s) in enumerate(self.sensors.locations):
            dist = (self.ds[self.lat_key][0]-s['lat'])**2+\
                (self.ds[self.lon_key][0]-s['lon'])**2
            self.i[k], self.j[k] = np.unravel_index(dist.argmin(), dist.shape)

    def get_field(self, field):
        '''
        Returns the given field at all the sensors locations.
        '''
        values = [0 for i in range(self.n_sensors)]
        for k in range(self.n_sensors):
            values[k] = self.ds[field].sel(south_north=self.i[k],
                west_east=self.j[k]).values
        return values

