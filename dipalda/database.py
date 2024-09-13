import geopandas as gpd
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import csv
from psycopg2 import pool


class Database:

    def __init__(self, host, port, database, user, password):
        self.connection_pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=2, database=database, user=user,
                                                                  password=password, host=host, port=port)
        self.engine = create_engine(
            url='postgresql+psycopg2://' + user + ':' + password + '@' + host + ':' + port + '/' + database,
            pool_size=2, max_overflow=2, pool_pre_ping=True)

    def close_connection_pool(self):
        if self.connection_pool:
            self.connection_pool.closeall()

    def close_connection(self, conn):
        if self.connection_pool:
            self.connection_pool.putconn(conn)

    def get_new_connection(self):
        conn = self.connection_pool.getconn()
        conn.autocommit = True
        return conn

    def execute_with_new_connection(self, sql):
        # get new connection and return it to the pool after usage
        conn = self.get_new_connection()
        try:
            conn.cursor().execute(sql)
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while connecting to PostgreSQL", error)
            conn.rollback()
        finally:
            self.connection_pool.putconn(conn)

    def execute_with_existing_connection(self, sql, conn):
        # use existing connection and keep it alive after usage
        try:
            if conn:
                conn.cursor().execute(sql)
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error while connecting to PostgreSQL", error)
            conn.rollback()

    def read_data_sets_into_db(self):
        conn = self.get_new_connection()

        # Read Covid-19 data - updated on 15 Sep 2022
        # https://www.ecdc.europa.eu/en/publications-data/subnational-14-day-notification-rate-covid-19
        sql = '''CREATE TABLE covid_rates(country text, region_name text, nuts_code text, date date, 
        rate_14_day_per_100k text, source text);'''
        self.execute_with_existing_connection(sql, conn)
        with open('resources/covid-19.csv', 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                conn.cursor().execute('''INSERT INTO covid_rates VALUES (%s, %s, %s, %s, %s, %s)''', row)
        sql = '''ALTER TABLE covid_rates ALTER COLUMN rate_14_day_per_100k TYPE numeric 
        USING COALESCE(NULLIF(rate_14_day_per_100k, '')::NUMERIC, 0)'''
        self.execute_with_existing_connection(sql, conn)

        # Read subregions by NUTS Code (EPSG 3035)
        # https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts
        nuts = gpd.read_file('resources/NUTS_RG_20M_2021_3035.shp')
        nuts.to_postgis('nuts', self.engine, index=False, if_exists='replace')

        # Read population data (ETRS89 / LAEA): European LAEA (Lambert Azimuthal Equal Area) projection, EPSG 3035
        # https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/population-distribution-demography/geostat#geostat11
        population = gpd.read_file('resources/ESTAT_Census_2011_V1-0.gpkg')
        population.to_postgis('population', self.engine, index=False, if_exists='replace')

        sql = '''ALTER TABLE population RENAME COLUMN "geometry" TO geom;'''
        self.execute_with_existing_connection(sql, conn)

        sql = '''ALTER TABLE population RENAME COLUMN "OBS_VALUE_T" TO obs_value_t;'''
        self.execute_with_existing_connection(sql, conn)

        sql = '''ALTER TABLE population RENAME COLUMN "GRD_ID" TO grd_id;'''
        self.execute_with_existing_connection(sql, conn)

        # Create table covid_countries:
        # Selects only countries for which data is available in covid_rates table
        sql = '''SELECT DISTINCT "CNTR_CODE" INTO covid_countries 
                 FROM nuts n INNER JOIN covid_rates c ON c.nuts_code=n."NUTS_ID"'''
        self.execute_with_existing_connection(sql, conn)

        self.close_connection(conn)

    def data_pre_processing(self, p):
        conn = self.get_new_connection()

        ####### Data preprocessing #########
        # 1. Create table locations:
        # Create random user locations in each grid cell from table 'population' according to the population count in the cell.
        # Choose only countries, where covid statistics are available.
        country_constraint = ''
        for i, country in enumerate(p.countries):
            country_constraint += '''c."CNTR_CODE" = \'''' + country + '''\''''
            if i < len(p.countries) - 1:
                country_constraint += ''' OR '''
        sql = '''SELECT population_points.*
                 INTO locations_tmp 
                 FROM (
                     /* Generate population points per grid cell in countries where Covid data is available */
                     SELECT pop_filtered.grd_id, pop_filtered.grid_geom, 
                     (ST_Dump(ST_GeneratePoints(ST_Intersection(pop_filtered.grid_geom, pop_filtered.country_geom), 
                     cast(pop_filtered.obs_value_t as int), 
                     seed := 1))).geom AS user_location 
                     FROM (
                        /* Population data per subregion in selected countries where Covid data is available */
                        SELECT p.grd_id, p.geom as grid_geom, p.obs_value_t, country_geom
                        FROM population p, (
                            /* Geometries of selected countries where Covid data is available */
                            select c."CNTR_CODE", geometry as country_geom
                            from nuts n inner join covid_countries c on n."CNTR_CODE" = c."CNTR_CODE"
                            where n."LEVL_CODE" = 0 and (''' + country_constraint + ''')
                            ) AS countries_filtered
                        WHERE st_intersects(st_centroid(p.geom), country_geom) /* where grid cell lies in country */
                     ) AS pop_filtered
                  ) AS population_points
                 ;'''
        self.execute_with_existing_connection(sql, conn)
        sql = '''ALTER TABLE locations_tmp ADD COLUMN id SERIAL PRIMARY KEY;'''
        self.execute_with_existing_connection(sql, conn)

        # 2. Create table locations:
        # Map each user location to its country and subregion (only level code 3 to partition the region)
        sql = '''SELECT l.*, n."CNTR_CODE", n."NUTS_ID", n."URBN_TYPE", n.geometry as sub_geom
                 INTO locations
                 FROM locations_tmp l, nuts n
                 WHERE "LEVL_CODE" = 3 AND ST_Intersects(l.user_location, n.geometry);'''
        self.execute_with_existing_connection(sql, conn)

        sql = '''DROP TABLE locations_tmp;'''
        self.execute_with_existing_connection(sql, conn)

        # 3. Create table population_count:
        # Calculate population count per subregion
        sql = '''SELECT l."CNTR_CODE", l."NUTS_ID", COUNT(*) as cnt
                 INTO population_count
                 FROM locations l 
                 GROUP BY l."CNTR_CODE", l."NUTS_ID"'''
        self.execute_with_existing_connection(sql, conn)

        # 4. Create table covid_cnt_per_subregion:
        # Calculate population count and map covid cases per subregion on specific dates
        sql = '''SELECT date, nuts_code, pop_cnt.cnt, c.rate_14_day_per_100k, 
                        cast((pop_cnt.cnt * c.rate_14_day_per_100k)/100000 as int) as covid_cases
                 INTO covid_cnt_per_subregion
                 FROM covid_rates c inner join population_count pop_cnt ON c."nuts_code" = pop_cnt."NUTS_ID"
                 WHERE date in (''' + str(p.dates).replace('[', '').replace(']', '') + ''');'''
        self.execute_with_existing_connection(sql, conn)

        # 5. Create table locations_covid:
        # Set user locations randomly to be covid positive depending on the number of covid cases per subregion
        for date in p.dates:
            sql = '''CREATE TABLE locations_covid_tmp AS
                     SELECT id, date
                        FROM (
                            SELECT "NUTS_ID", id, row_number() OVER (PARTITION BY "NUTS_ID" ORDER BY random()) as rn
                            FROM locations
                        ) l
                        INNER JOIN covid_cnt_per_subregion c ON l."NUTS_ID" = c.nuts_code
                        WHERE l.rn <= c.covid_cases and c.date = \'''' + date + '''\';'''
            self.execute_with_existing_connection(sql, conn)

            # Set covid attribute according to table locations_covid
            sql = '''CREATE TABLE locations_''' + self.get_date_name(date) + ''' AS
                 SELECT l.*, \'''' + str(date) + '''\' as date, 
                 case when c.id is null then FALSE else TRUE end as covid
                 FROM locations l LEFT JOIN locations_covid_tmp c ON l.id = c.id;'''
            self.execute_with_existing_connection(sql, conn)

            sql = '''DROP TABLE locations_covid_tmp;'''
            self.execute_with_existing_connection(sql, conn)

        ####### Define urban and rural areas #########

        # Create table surface_area_per_subregion:
        # Calculate surface area per subregion (only level code 3 to partition the region)
        sql = '''SELECT n.*, ST_Area(n.geometry)/(1000*1000.) as surface_area_km2, 
                 (ST_XMax(n.geometry) - ST_XMin(n.geometry))/1000. AS length_x, 
                 (ST_YMax(n.geometry) - ST_YMin(n.geometry))/1000. AS length_y
                 INTO surface_area_per_subregion
                 FROM nuts n
                 WHERE n."LEVL_CODE" = 3
                '''
        self.execute_with_existing_connection(sql, conn)

        # Create table population_density:
        # Calculate population density from surface area and population count
        sql = '''SELECT s.*, p.cnt as pop_cnt, 
                 (p.cnt/cast(s.surface_area_km2 as float)) as pop_density, 'NONE' as area_type_name, 0 as area_type_code
                 INTO population_density
                 FROM surface_area_per_subregion s inner join population_count p on s."NUTS_ID" = p."NUTS_ID"
                '''
        self.execute_with_existing_connection(sql, conn)

        # Update table population_density
        # Set area type (urban centre, urban cluster, or rural) based on population density
        # (see https://ghsl.jrc.ec.europa.eu/degurbaDefinitions.php)
        sql = '''UPDATE population_density
                 SET area_type_name =
                         CASE WHEN pop_density > 1500
                             THEN 'urban centre'
                         WHEN pop_density > 300
                             THEN 'urban cluster'
                         ELSE 'rural'
                         END,
                     area_type_code =
                         CASE WHEN pop_density > 1500
                             THEN 0
                         WHEN pop_density > 300
                             THEN 1
                         ELSE 2
                         END
                '''
        self.execute_with_existing_connection(sql, conn)

        self.close_connection(conn)

    def get_date_name(self, date):
        return date.replace('-', '_')

    def create_points_table(self, points_table_name, list_region_alias, date_name, user_parameters):
        h_min, h_max, k_min, k_max, p_min, p_max = user_parameters
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''"NUTS_ID" LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = '''SELECT * INTO TABLE ''' + points_table_name + \
              ''' FROM (SELECT id, user_location, covid, cast(random() * ''' + \
              str(h_max - h_min) + ''' + ''' + str(h_min) + ''' as int) AS max_level, cast(random() * ''' + \
              str(k_max - k_min) + ''' + ''' + str(k_min) + ''' as int) AS k, cast(random() * ''' + \
              str(p_max - p_min) + ''' + ''' + str(p_min) + ''' as float) AS p 
              FROM locations_''' + date_name + ''' WHERE ''' + var + ''') AS a;'''
        self.execute_with_new_connection(sql)

    def create_parameters_table(self, user_types):
        params = [{'type': u['type'], 
                   'k': (u['k']['upper'] + u['k']['lower'])/2.,
                   'p': (u['p']['upper'] + u['p']['lower'])/2.,
                   'h': (u['h']['upper'] + u['h']['lower'])/2.,
                   'k_name': self.get_param_name('k', u),
                   'p_name': self.get_param_name('p', u),
                   'h_name': self.get_param_name('h', u)
                   } for u in user_types]
        df = pd.DataFrame(params)
        df.to_sql('parameters', self.engine)

    def create_region_parameters_table(self, regions):
        params = [(u['name'], u['type']) for u in regions]
        params = [(i, item[0], item[1]) for i, item in enumerate(params)]
        nuts_ids = [(nutsid, u['name']) for u in regions for nutsid in u['nuts_id']]
        params = [{'region_id': p[0], 'name': p[1], 'type': p[2], 'nuts_id': n[0]}
                  for p in params for n in nuts_ids if p[1] == n[1]]
        df = pd.DataFrame(params)
        df.to_sql('parameters_regions', self.engine)

    def get_covid_rate_in_regions(self):
        sql = '''SELECT date, name, avg(rate_14_day_per_100k)/100 FROM covid_rates c, parameters_regions p 
                WHERE nuts_code like p.nuts_id group by date, name'''
        df = pd.read_sql(sql, self.engine)
        return df

    def get_param_name(self, param, user_type):
        lower = user_type[param]['lower']
        upper = user_type[param]['upper']
        if param == 'p':
            lower = str(int(lower * 100.))
            upper = str(int(upper * 100.))
        name = str(lower) if lower == upper else (str(lower) + '-' + str(upper))
        if param == 'p':
            name += "%"
        return name

    def get_points(self, list_region_alias, date):
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''"NUTS_ID" LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''

        sql = '''SELECT id, user_location, "NUTS_ID", covid 
        FROM locations_covid_''' + self.get_date_name(date) + ''' WHERE ''' + var
        df = gpd.read_postgis(sql, self.engine, geom_col='user_location')
        df = df.to_crs(epsg=4326)
        return df

    def get_nr_points(self, list_region_alias):
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''"NUTS_ID" LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = '''SELECT COUNT(*) FROM locations WHERE ''' + var
        df = pd.read_sql(sql, self.engine)
        count = df['count'].values[0]
        return count

    def get_region_stats(self, list_region_alias):
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''"NUTS_ID" LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''

        sql = '''SELECT "NUTS_ID", "CNTR_CODE", "NAME_LATN", area_type_name, surface_area_km2, pop_cnt, 
                pop_density FROM population_density WHERE ''' + var
        df_stats = pd.read_sql(sql, self.engine)
        sql = '''SELECT SUM(surface_area_km2) as surface_area_km2_sum, SUM(pop_cnt) as pop_cnt_sum, 
                COUNT(*) as nr_sub_regions, (SUM(pop_cnt)/cast(SUM(surface_area_km2) as float)) as pop_density 
                FROM population_density WHERE ''' + var
        df_sum = pd.read_sql(sql, self.engine)
        return df_stats, df_sum

    def get_all_region_stats(self):
        sql = '''select r.region_id, r.name, c.date, count(p."NUTS_ID") as nr_subareas, 
        sum(p.pop_cnt)/(st_area(ST_Union(p.geometry))/1000000.) as pop_per_km2,
        sum(p.pop_cnt) as pop_cnt, sum(c.covid_cases) as covid_cases, 
        (sum(c.covid_cases)/sum(p.pop_cnt))*100000. as covid_rate, 
        st_area(ST_Union(p.geometry))/1000000. as surface_area_km2, 
        (ST_XMax(ST_Union(p.geometry)) - ST_XMin(ST_Union(p.geometry)))/1000. AS length_x, 
        (ST_YMax(ST_Union(p.geometry)) - ST_YMin(ST_Union(p.geometry)))/1000. AS length_y
        from population_density p inner join covid_cnt_per_subregion c on p."NUTS_ID"=c.nuts_code, parameters_regions r
        where p."NUTS_ID" like r.nutsid
        group by r.name, r.region_id, c.date
        order by c.date, r.region_id, r.name'''
        df = pd.read_sql(sql, self.engine)
        return df

    def get_all_subregion_stats(self):
        sql = '''select r.region_id, r.name, p."NUTS_ID", p."NAME_LATN", c.date, 
        sum(p.pop_cnt)/(st_area(ST_Union(p.geometry))/1000000.) as pop_per_km2,
        sum(p.pop_cnt) as pop_cnt, sum(c.covid_cases) as covid_cases, 
        (sum(c.covid_cases)/sum(p.pop_cnt))*100000. as covid_rate, 
        st_area(ST_Union(p.geometry))/1000000. as surface_area_km2, 
        (ST_XMax(ST_Union(p.geometry)) - ST_XMin(ST_Union(p.geometry)))/1000. AS length_x, 
        (ST_YMax(ST_Union(p.geometry)) - ST_YMin(ST_Union(p.geometry)))/1000. AS length_y
        from population_density p inner join covid_cnt_per_subregion c on p."NUTS_ID"=c.nuts_code, parameters_regions r
        where p."NUTS_ID" like r.nutsid
        group by r.name, r.region_id, c.date, p."NUTS_ID", p."NAME_LATN"
        order by c.date, r.region_id, r.name, p."NUTS_ID";'''
        df = pd.read_sql(sql, self.engine)
        return df

    def get_covid_stats(self, list_region_alias, date):
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''nuts_code LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''

        sql = '''SELECT SUM(cnt) as cnt, SUM(covid_cases) as covid_cases, SUM(cast(covid_cases as float))/SUM(cnt) 
        AS covid_rate FROM covid_cnt_per_subregion WHERE date = ''' + str(date) + ''' and ''' + var
        df = pd.read_sql(sql, self.engine)
        return df

    def get_covid_stats_per_subregion(self, list_region_alias, date):
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''nuts_code LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = '''SELECT min(cast(covid_cases as float)/cnt) as min_covid_rate, 
        max(cast(covid_cases as float)/cnt) as max_covid_rate, 
        variance(cast(covid_cases as float)/cnt) as var_covid_rate 
        FROM covid_cnt_per_subregion WHERE date = ''' + str(date) + ''' and ''' + var
        df = pd.read_sql(sql, self.engine)
        return df

    def get_total_bounds(self, list_region_alias, srid):
        var = ''
        for i, region in enumerate(list_region_alias):
            var += '''"NUTS_ID" LIKE \'''' + region + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''

        sql = '''SELECT st_union(n.geometry) as geometry FROM nuts n WHERE ''' + var
        df = gpd.read_postgis(sql, self.engine, geom_col='geometry')
        df = df.to_crs(epsg=srid)
        return df.total_bounds

    def write_tree(self, tree, table_name):
        df = tree.read_tree_df_per_node()
        df.to_sql(name=table_name, con=self.engine)

    def copy(self, table_name, table_name_copy):
        sql = '''SELECT * INTO ''' + table_name_copy + ''' FROM ''' + table_name + ''';'''
        self.execute_with_new_connection(sql)

    def write_df(self, df, table_name):
        # todo: test if append works
        df.to_sql(name=table_name, con=self.engine, if_exists='append', index=False)

    def get_tree_df(self, table_name):
        sql = '''SELECT * FROM ''' + str(table_name)
        df = pd.read_sql(sql, self.engine) if self.exists(table_name) else None
        return df

    def exists(self, table_name):
        sql = '''SELECT EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND 
        tablename = \'''' + table_name + '''\');'''
        df = pd.read_sql(sql, self.engine)
        return df['exists'].values[0]

    def drop(self, table_name):
        sql = '''DROP TABLE ''' + table_name + ''';'''
        self.execute_with_new_connection(sql)
