import numpy as np
import pandas as pd


class Interpolator:
    '''
    Interpolates the dataset at the sensor locations.
    '''

    def __init__(self, sensors, ds, lat, lon):
        self.sensors = sensors
        self.ds = ds
        self.lat = lat
        self.lon = lon
        self._find_indices()

    def _find_indices(self):
        '''
        Finds the indices that correspond to each sensor.
        '''
        self.i = [0 for i in range(self.sensors.n)]
        self.j = [0 for i in range(self.sensors.n)]
        for (k, s) in enumerate(self.sensors.locations):
            dist = (self.ds[self.lat][0]-s['lat'])**2+\
                (self.ds[self.lon][0]-s['lon'])**2
            self.i[k], self.j[k] = np.unravel_index(dist.argmin(), dist.shape)

    def get_field(self, field):
        '''
        Returns the given field at all the sensors locations.
        '''
        values = [0 for i in range(self.sensors.n)]
        for k in range(self.sensors.n):
            values[k] = self.ds[field].sel(south_north=self.i[k],
                west_east=self.j[k]).values
        return np.array(values)

