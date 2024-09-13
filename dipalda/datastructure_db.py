import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shapely
import shapely.ops
import geopy.distance
import pyproj


class Point:
    def __init__(self, x, y, srid):
        self.geometry = shapely.Point((x, y))
        self.srid = srid
        self.crs = pyproj.CRS(srid)
        self.x = self.geometry.x
        self.y = self.geometry.y

    def transform(self, to_srid):
        t = Point(self.x, self.y, self.srid)
        if t.srid != to_srid:
            project = pyproj.Transformer.from_crs(t.srid, pyproj.CRS(to_srid), always_xy=True).transform
            t.geometry = shapely.ops.transform(project, t.geometry)
            t.x = t.geometry.x
            t.y = t.geometry.y
            t.srid = to_srid
            t.crs = pyproj.CRS(to_srid)
        return t

    @staticmethod
    def get_distance(point_a, point_b):
        point_a_t = point_a.transform(to_crs='EPSG:4326')
        point_b_t = point_b.transform(to_crs='EPSG:4326')
        return geopy.distance.distance([point_a_t.x, point_a_t.y], [point_b_t.x, point_b_t.y]).m


class Rectangle:
    """
    A rectangle class. Location data is handled internally with EPSG:32632 and transformed into EPSG:4326 only for
    geometry representation.
    """

    def __init__(self, bounds, srid):
        # EPSG:4326 for plotting purposes
        self.srid_plotting = 4326
        # UTM (Universal Transverse Mercator) zone, EPSG:32632, for planar measurements in meters in Europe
        self.srid_calculation = 32632
        
        self.srid = srid
        self.bounds = bounds

        self.bbox_df = self.create_bbox()
        self.__transform_to_srid(self.srid_calculation)

    def vertices(self):
        return [self.lower_left(), self.top_left(), self.top_right(), self.lower_right()]

    def create_bbox(self):
        return gpd.GeoSeries(self.vertices(), crs='EPSG:' + str(self.srid))
    
    def geometry(self):
        """Return shapely geometry of this rectangle in EPSG:4326."""
        self.__transform_to_srid(srid=self.srid_plotting)
        geometry = shapely.Polygon(self.vertices())
        self.__transform_to_srid(srid=self.srid_calculation)
        return geometry

    def postgis_geometry(self):
        """Return postgis polygon of this rectangle in EPSG:4326."""
        self.__transform_to_srid(srid=self.srid_plotting)
        vertices = self.vertices() + [self.vertices()[0]]
        geometry = "'POLYGON((" + ','.join([str(p.x) + " " + str(p.y) for p in vertices]) + "))'::geometry"
        self.__transform_to_srid(srid=self.srid_calculation)
        return geometry

    def transformed_postgis_geometry(self, to_srid):
        return "ST_Transform(ST_SetSRID(" + self.postgis_geometry() + "," + str(self.srid_plotting) + \
            ")," + str(to_srid) + ")"

    def __transform_to_srid(self, srid):
        self.srid = srid
        self.bbox_df = self.bbox_df.to_crs(crs='EPSG:' + str(srid))
        self.bounds = self.bbox_df.total_bounds

    def width(self):
        minx, _, maxx, _ = self.bounds
        return maxx - minx

    def width_meters(self):
        minx, _, maxx, _ = self.bbox_df.to_crs(crs='EPSG:' + str(self.srid_calculation)).total_bounds
        return maxx - minx

    def height(self):
        _, miny, _, maxy = self.bounds
        return maxy - miny

    def height_meters(self):
        _, miny, _, maxy = self.bbox_df.to_crs(crs='EPSG:' + str(self.srid_calculation)).total_bounds
        return maxy - miny

    def max_extension(self):
        return max(self.width(), self.height())

    def max_extension_meters(self):
        return max(self.width_meters(), self.height_meters())

    def surface_area_square_meters(self):
        return self.max_extension_meters()**2

    def lower_left(self):
        minx, miny, _, _ = self.bounds
        return shapely.Point(minx, miny)

    def top_left(self):
        minx, _, _, maxy = self.bounds
        return shapely.Point(minx, maxy)

    def top_right(self):
        _, _, maxx, maxy = self.bounds
        return shapely.Point(maxx, maxy)

    def lower_right(self):
        _, miny, maxx, _ = self.bounds
        return shapely.Point(maxx, miny)

    def center(self):
        minx, miny, _, _ = self.bounds
        cx = minx + self.width() / 2.
        cy = miny + self.height() / 2.
        return shapely.Point(cx, cy)

    def make_bbox_square(self):
        minx, miny, maxx, maxy = self.bounds
        max_extension = self.max_extension()
        
        minx = (minx + maxx - max_extension) / 2.
        maxx = minx + max_extension
        miny = (miny + maxy - max_extension) / 2.
        maxy = miny + max_extension
        
        self.bounds = minx, miny, maxx, maxy
        self.bbox_df = self.create_bbox()

    def sub_rectangle(self, direction):
        """
        Create a new sub-rectangle at the indicated direction.
        :param direction: {'nw', 'ne', 'se', 'sw'}
            The direction at which the sub-rectangle is located.
        :return: Rectangle
            A sub-rectangle with one fourth of this rectangle's size at the indicated direction and with one edge being
            this rectangle's center.
        """
        bounds = None
        minx, miny, maxx, maxy = self.bounds
        c = self.center()
        if direction == 'nw':
            bounds = minx, c.y, c.x, maxy
        elif direction == 'ne':
            bounds = c.x, c.y, maxx, maxy
        elif direction == 'se':
            bounds = c.x, miny, maxx, c.y
        elif direction == 'sw':
            bounds = minx, miny, c.x, c.y
        return Rectangle(bounds, srid=self.srid)


def calculate_root_epsilon(epsilon, privacy_ratio, max_level):
    return (privacy_ratio - 1.) * epsilon/(np.power(privacy_ratio, max_level + 1.) - 1.)


class Tree:
    """A class implementing a Tree. Datastructure is saved in a database."""

    def __init__(self, bbox, tree_table_name, db):
        self.tree_table_name = tree_table_name
        self.db = db
        self.bbox = bbox
        self.root = Node(bbox, self)

    def is_empty(self):
        empty = True
        if self.db.exists(self.tree_table_name):
            sql = "SELECT * FROM " + self.tree_table_name
            df = pd.read_sql(sql, self.db.engine)
            empty = len(df) == 0
        return empty

    def draw_points_on_leafs(self):
        DPI = 72
        plt.figure(figsize=(700/DPI, 500/DPI), dpi=DPI)
        ax = plt.subplot()
        self.root.draw_points_on_leafs(ax)
        plt.show()

    def draw_points_on_level(self, level):
        DPI = 72
        plt.figure(figsize=(700 / DPI, 500 / DPI), dpi=DPI)
        ax = plt.subplot()
        self.root.draw_points_on_level(level, ax)
        plt.show()

    def find(self, point):
        return self.root.find(point)


class QuadTree(Tree):
    """A class implementing a Quadtree."""

    def __init__(self, bbox, tree_table_name, db):
        super().__init__(bbox, tree_table_name, db)

    def insert(self, points_table_name, k_per_level, p_per_level, max_level, save_points=True,
               always_divide=False):
        self.root.insert(points_table_name, k_per_level, p_per_level, max_level, save_points,
                         always_divide)


class DPQuadTree(Tree):
    """A class implementing a Quadtree with noisy counts according to the approach in
    Liu et al. (2022): Differential privacy location data release based on quadtree in mobile edge computing"""
    def __init__(self, bbox, tree_table_name, db):
        super().__init__(bbox, tree_table_name, db)

    def insert(self, points_table_name, max_level, epsilon, privacy_ratio, save_points=True,
               always_divide=False):
        self.root.insert(points_table_name, k_per_level=None, p_per_level=None, max_level=max_level,
                         save_points=save_points, always_divide=always_divide)
        self.root.add_laplace_noise(epsilon, privacy_ratio)

    def adjust_count(self, t, theta):
        self.root.adjust_count(t, theta)


class Node:
    """A class implementing a node of a Quadtree."""

    def __init__(self, bbox, tree, level=0, position=0, name='root'):
        """Initialize this node of the tree."""
        self.tree = tree
        self.bbox = bbox
        self.k_per_level = None
        self.p_per_level = None
        self.k = None
        self.p = None
        self.max_level = None
        self.is_k_valid = None
        self.is_p_valid = None
        self.surface_area_m2 = bbox.surface_area_square_meters()
        self.population_count = None
        self.sensitive_users_count = None
        self.valid_sensitive_users_count = None
        self.sensitive_users_ratio = 0.
        self.points_geometries = []    # geometries of sensitive users filtered by privacy parameters
        self.rectangle_geometry = None
        self.level = level
        self.position = position
        self.name = name
        self.is_divided = False
        self.nw = None
        self.ne = None
        self.se = None
        self.sw = None

    def divide(self):
        """Divide (branch) this node by spawning four children nodes."""
        level = self.level + 1
        position = np.max([0, self.position * 4])
        self.nw = Node(self.bbox.sub_rectangle('nw'), self.tree, level, position, name='nw')
        self.ne = Node(self.bbox.sub_rectangle('ne'), self.tree, level, position + 1, name='ne')
        self.se = Node(self.bbox.sub_rectangle('se'), self.tree, level, position + 2, name='se')
        self.sw = Node(self.bbox.sub_rectangle('sw'), self.tree, level, position + 3, name='sw')
        self.is_divided = True

    def find(self, point):
        # todo: use Rectangle's contains class
        found = False
        level = self.level
        position = self.position
        sql = "SELECT * FROM " + self.tree.tree_table_name
        df = pd.read_sql(sql, self.tree.engine)
        vertex_exists = len(df) > 0 and len(df[(df['level'] == level) & (df['position'] == position)]) != 0
        if self.bbox.contains(point) and vertex_exists:
            found = True
            for child in self.children():
                c_found, c_level, c_position = child.find(point)
                if c_found:
                    found, level, position = c_found, c_level, c_position
                    break
        return found, level, position

    def insert(self, points_table_name, k_per_level, p_per_level, max_level, save_points,
               always_divide):
        """
        Insert points of type Point into this node. If save_points is True, the inserted points are saved in tree_items.

        @:param save_points: boolean
            If true, save user locations in tree table. Necessary to plot location data but uses additional resources.
        """
        self.k_per_level = k_per_level
        self.p_per_level = p_per_level
        self.k = 0. if k_per_level is None else self.k_per_level[self.level]
        self.p = 1. if p_per_level is None else self.p_per_level[self.level]
        self.max_level = max_level

        table_exists = self.tree.db.exists(self.tree.tree_table_name)
        node_exists = False
        if table_exists:
            sql = "SELECT * FROM " + self.tree.tree_table_name + " WHERE level = " + str(self.level) +\
                  " AND position = " + str(self.position) + ";"
            df_item = pd.read_sql(sql, self.tree.db.engine)
            node_exists = len(df_item) != 0
            if node_exists:
                self.is_k_valid = df_item['is_k_valid'].values[0]
                self.is_p_valid = df_item['is_p_valid'].values[0]
                self.sensitive_users_count = df_item['sensitive_users_count'].values[0]
                self.valid_sensitive_users_count = df_item['valid_sensitive_users_count'].values[0]
                self.population_count = df_item['population_count'].values[0]

        if not node_exists:
            sql = '''SELECT COUNT(*) FROM ''' + points_table_name + ''' WHERE ST_Intersects(user_location,''' + \
                  self.bbox.transformed_postgis_geometry(3035) + ''');'''
            df_population_count = pd.read_sql(sql, self.tree.db.engine)
            self.population_count = df_population_count['count'].values[0]

            sql = "SELECT COUNT(*) FROM " + points_table_name + " WHERE ST_Intersects(user_location, " + \
                  self.bbox.transformed_postgis_geometry(3035) + ") AND covid = TRUE;"
            df_sensitive_users_count = pd.read_sql(sql, self.tree.db.engine)
            self.sensitive_users_count = df_sensitive_users_count['count'].values[0]

            self.sensitive_users_ratio = 0. if self.population_count == 0 else \
                (float(self.sensitive_users_count) / self.population_count)

            # filter sensitive users whose privacy requirements are in line with the negotiated values
            if always_divide:
                self.valid_sensitive_users_count = self.sensitive_users_count
            else:
                sql = "SELECT COUNT(*) FROM " + points_table_name + " WHERE ST_Intersects(user_location, " + \
                      self.bbox.transformed_postgis_geometry(3035) + ") AND covid = TRUE AND " + str(self.level) + \
                      " <= max_level AND " + str(self.k) + " >= k AND " + str(self.p) + "<= p;"
                df_valid_sensitive_users_count = pd.read_sql(sql, self.tree.db.engine)
                self.valid_sensitive_users_count = df_valid_sensitive_users_count['count'].values[0]

            # check for the node if the sum of valid sensitive users fulfills the negotiated privacy requirement
            self.is_k_valid = self.k <= self.valid_sensitive_users_count
            self.is_p_valid = self.valid_sensitive_users_count <= self.p * self.population_count
            self.points_geometries = []
            if save_points and self.is_k_valid and self.is_p_valid:
                sql = "SELECT user_location FROM " + points_table_name + " WHERE ST_Intersects(user_location, " + \
                      self.bbox.transformed_postgis_geometry(3035) + ") AND covid = TRUE AND " + str(self.level) + \
                      " <= max_level AND " + str(self.k) + " >= k AND " + str(self.p) + "<= p;"
                df_valid_sensitive_users = pd.read_sql(sql, self.tree.db.engine)
                self.points_geometries = df_valid_sensitive_users['user_location'].values.tolist()
            self.rectangle_geometry = self.bbox.geometry()

        # divide only nodes that meet the privacy criteria
        if always_divide or (self.is_k_valid and self.is_p_valid and self.valid_sensitive_users_count > 0):
            if self.level < self.max_level:
                self.divide()
                for child in self.children():
                    child.insert(points_table_name, self.k_per_level, self.p_per_level, self.max_level,
                                 save_points, always_divide)

            # write node data to db, if not exists
            if not node_exists:
                node_item = {'level': [self.level], 'position': [self.position], 'name': [self.name],
                             'is_divided': [self.is_divided], 'k': [self.k], 'p': [self.p],
                             'is_k_valid': [self.is_k_valid],
                             'is_p_valid': [self.is_p_valid], 'population_count': [self.population_count],
                             'sensitive_users_count': [self.sensitive_users_count],
                             'valid_sensitive_users_count': [self.valid_sensitive_users_count],
                             'sensitive_users_ratio': [self.sensitive_users_ratio],
                             'surface_area_m2': [self.surface_area_m2], 'points': [self.points_geometries],
                             'rectangle': [self.rectangle_geometry]}
                gdf = gpd.GeoDataFrame(node_item, geometry='rectangle', crs=self.bbox.srid)
                gdf.to_crs(4236)
                gdf.to_postgis(name=self.tree.tree_table_name, con=self.tree.db.engine, if_exists='append', index=False)

        print('node_exists', node_exists, 'level', self.level, 'pos', self.position)

    def add_laplace_noise(self, epsilon, privacy_ratio):
        """Add Laplace noise on counts per tree level."""
        self.sensitive_users_count += np.random.laplace(loc=0, scale=1./epsilon)
        if self.is_divided:
            for child in self.children():
                child.add_laplace_noise(privacy_ratio * epsilon, privacy_ratio)

    def adjust_count(self, t, theta):
        if self.is_divided:
            for child in self.children():
                child.adjust_count(t, theta)
            if self.sensitive_users_count <= t:
                self.delete_children()
                self.is_divided = False
            else:
                is_on_second_last_level = True
                for child in self.children():
                    if child.is_divided:
                        is_on_second_last_level = False
                        break
                if is_on_second_last_level:
                    sum_of_counts = np.sum([child.sensitive_users_count for child in self.children()])
                    avg_count = sum_of_counts / len(self.children())
                    sum_of_diffs = np.sum([np.abs(child.sensitive_users_count - avg_count) for child in self.children()])
                    if sum_of_diffs / sum_of_counts <= theta:
                        self.delete_children()
                        self.is_divided = False

    def children(self):
        return [self.nw, self.ne, self.se, self.sw] if self.is_divided else []

    def delete_children(self):
        if self.is_divided:
            for child in self.children():
                idx = [i for (i, item) in enumerate(self.tree.tree_items) if item['level'] == child.level and
                       item['position'] == child.position and item['k'] == child.k and item['p'] == child.p]
                if len(idx) > 0:
                    del self.tree.tree_items[idx[0]]
                child.delete_children()

    def draw_bbox(self, ax):
        """Draw a representation of the Node on Matplotlib Axes ax without points."""
        self.bbox.draw(ax)
        if not self.is_divided:
            ax.scatter([p.x for p in self.points_geometries], [p.y for p in self.points_geometries], s=4)
        if self.is_divided:
            for child in self.children():
                child.draw_bbox(ax)

    def draw_points_on_leafs(self, ax):
        """Draw tree leafs with points."""
        self.bbox.draw(ax)
        if not self.is_divided:
            ax.scatter([p.x for p in self.points_geometries], [p.y for p in self.points_geometries], s=10)
        if self.is_divided:
            for child in self.children():
                child.draw_points_on_leafs(ax)

    def draw_points_on_level(self, level, ax):
        """Draw tree on certain level with points."""
        self.bbox.draw(ax)
        if level > self.level:
            if self.is_divided:
                for child in self.children():
                    child.draw_points_on_level(level, ax)
        else:
            ax.scatter([p.x for p in self.points_geometries], [p.y for p in self.points_geometries], s=10)

    def draw_heatmap_infection_rate(self, ax):
        """Draw heatmap based on infection rate in the leaf nodes."""
        self.bbox.draw(ax)
        if not self.is_divided:
            infection_rate = float(self.sensitive_users_count)/self.population_count

        if self.is_divided:
            for child in self.children():
                child.draw_points_on_leafs(ax)
