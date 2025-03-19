import json
import pathlib

import folium
import geojson
import numpy as np
from pyproj import Proj
import xarray as xr

# https://stackoverflow.com/q/30052990
# https://python-visualization.github.io/folium/latest/getting_started.html
# https://pyproj4.github.io/pyproj/stable/api/proj.html


class Sensors:
    '''
    Plots the sensors locations on a map and saves them to a json file.
    '''

    def __init__(self, geojson_file_path):
        self.geojson_file_path = geojson_file_path
        self._read()
        self._set_locations()

    def _read(self):
        '''
        Reads the geojson file.
        '''
        with open(self.geojson_file_path, 'r') as fin:
            self.data = geojson.load(fin)

    def _set_locations(self):
        '''
        Sets the locations.
        '''
        self.locations = []
        p = Proj('urn:ogc:def:crs:EPSG::25830')
        for f in self.data['features']:
            x, y = f['geometry']['coordinates']
            props = f['properties']
            lon, lat = p(x, y, inverse=True)
            self.locations.append({
                'code': f['properties']['COD_OBS'],
                'name': f['properties']['NOMBRE'],
                'lon': lon,
                'lat': lat,
            })
        codes = [l['code'] for l in self.locations]
        indices = np.argsort(codes)
        self.locations = [self.locations[i] for i in indices]


class JsonMesh:
    '''
    Reads the mesh coordinates from the json file.
    '''

    def __init__(self, json_file_path):
        self.json_file_path = pathlib.Path(json_file_path)
        self._read()

    def _read(self):
        '''
        Reads the json file.
        '''
        with open(self.json_file_path, 'r') as fin:
            self.coords = json.load(fin)

    def get_mesh_extent(self):
        '''
        Returns the mesh extent.
        '''
        self.names = sorted(self.coords.keys())
        extent = {}
        for name in self.names:
            extent[name] = [
                [[lat, coords['lon'][0]] for lat in coords['lat']],
                [[coords['lat'][-1], lon] for lon in coords['lon']],
                [[lat, coords['lon'][-1]] for lat in coords['lat']],
                [[coords['lat'][0], lon] for lon in coords['lon']],
            ]
        return extent

    def get_mesh(self):
        '''
        Returns the meshes.
        '''
        meshes = {}
        for name in sorted(self.coords.keys()):
            lons = self.coords[name]['lon']
            lats = self.coords[name]['lat']
            horizontal = [[[lat, lon] for lon in lons] for lat in lats]
            vertical = [[[lat, lon] for lat in lats] for lon in lons]
            meshes[name] = horizontal+vertical
        return meshes


class GeoMesh:
    '''
    Reads the mesh from the given geo_em file.
    '''

    def __init__(self, geo_em_file_path):
        self.geo_em_file_path = pathlib.Path(geo_em_file_path)
        self._read()

    def _read(self):
        '''
        Reads the data in the geo_em file.
        '''
        self.ds = xr.open_dataset(self.geo_em_file_path)

    def get_mesh(self):
        '''
        Returns the mesh lines for plotting.
        '''
        lons = self.ds['XLONG_C'].values[0]
        lats = self.ds['XLAT_C'].values[0]
        h = [0 for j in range(len(lons))]
        for j in range(len(lons)):
            h[j] = [[lat, lon] for (lat, lon) in zip(lats[:, j], lons[:, j])]
        v = [0 for i in range(len(lats))]
        for i in range(len(lats)):
            v[i] = [[lat, lon] for (lat, lon) in zip(lats[i, :], lons[i, :])]
        mesh = {self.geo_em_file_path.name: h+v}
        return mesh


class CellCentres:
    '''
    Stores the cell centres of each sensor.
    '''

    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        self._read()
        self._set_locations()

    def _read(self):
        '''
        Reads the json file.
        '''
        with open(self.json_file_path, 'r') as fin:
            self.data = json.load(fin)

    def _set_locations(self):
        '''
        Sets the locations for each sensor.
        '''
        self.locations = []
        for sensor_data in self.data['sensors']:
            self.locations.append({
                'code': sensor_data['code'],
                'name': sensor_data['name']+' cell centre',
                'lon': sensor_data['XLONG'][0],
                'lat': sensor_data['XLAT'][0],
            })


class Map:
    '''
    Writes a map with the sensors and mesh coordinates.
    '''

    def __init__(self):
        self._init_map()

    def _init_map(self):
        '''
        Initialises the map.
        '''
        self.map = folium.Map(
                location=(41.65606, -0.87734),
                control_scale=True,
                zoom_start=12,
            )

    def add_mesh(self, mesh):
        '''
        Adds the mesh coordinates to the map.
        '''
        for (name, lines) in mesh.items():
            mesh_group = folium.FeatureGroup(name).add_to(self.map)
            for line in lines:
                poly_line = folium.PolyLine(locations=line).add_to(mesh_group)

    def add_sensors(self, sensors, name, color='blue'):
        '''
        Adds the sensor locations to the map.
        '''
        sensors_group = folium.FeatureGroup(name).add_to(self.map)
        for l in sensors:
            name = f'{l["code"]} {l["name"]}'
            marker = folium.Marker((l['lat'], l['lon']), popup=f'{name}',
                icon=folium.Icon(color))
            marker.add_to(sensors_group)

    def write(self, html_file_path):
        '''
        Writes the map with the sensor locations.
        '''
        folium.LayerControl().add_to(self.map)
        print(f'Writing {html_file_path}')
        self.map.save(html_file_path)


if __name__ == '__main__':
    zaragoza = Map()
    # mesh = JsonMesh('v02/mesh-coords.json')
    mesh = GeoMesh('v03/geo_em.d03.nc')
    zaragoza.add_mesh(mesh.get_mesh())
    sensors = Sensors('../02-cierzo-bochorno/mapa/sensores_islas.geojson')
    zaragoza.add_sensors(sensors.locations, 'Sensors')
    cell_centres = CellCentres('../04-wrf/bochorno-v03-01/bochorno-v03-01.json')
    zaragoza.add_sensors(cell_centres.locations, 'Cell centres', 'red')
    zaragoza.write('v03/mesh-with-sensors.html')

