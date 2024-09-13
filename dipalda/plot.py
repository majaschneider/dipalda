import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import contextily as ctx
import xyzservices.providers as xyz
import pandas as pd
import geopandas as gpd
import seaborn as sns
from shapely.geometry import Point
import numpy as np
import matplotlib.colors as mcolors


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def get_region_name_beautiful(region_name):
    region_names_beautiful = {'luneburgarea': 'Lüneburg area',
                              'colognearea': 'Cologne area',
                              'saxony': 'Saxony',
                              'berlin': 'Berlin',
                              'firenzearea': 'Florence area',
                              'jylland': 'Jylland area',
                              'milano': 'Milan'}
    return region_names_beautiful[region_name]


def get_column_name_beautiful(column_name):
    column_names_beautiful = {'fscore_dens': 'F1-Score', 'fscore_ratio': 'F1-Score',
                              'rel_infection_count_error': 'Relative Error', 'h_name': 'h', 'p_name': 'p',
                              'k_name': 'k'}
    return column_names_beautiful[column_name]


def param_wo_spec_char(parameter_name):
    return parameter_name.replace('%', 'pct')


class Plot:
    def __init__(self, db, base_path, parameters):
        self.db = db
        self.base_path = base_path
        self.p = parameters
        self.title_fontsize = 18
        self.axes_label_fontsize = 18
        self.tick_params_labelsize = 14
        self.dpi = 200
        self.figsize_heatmap = (10, 10)
        self.default_figsize_square = (6, 6)
        self.default_figsize_rect = (6, 4)
        self.legend_fontsize = 14
        self.legend = True
        self.colors = ['dodgerblue', 'gold', 'tomato', 'mediumpurple', 'mediumseagreen', 'peru', 'teal']

    def get_user_type_codes(self, user_types):
        return [e['type'] for e in user_types]

    def get_utility_table_name(self, detail_level):
        return 'utility_' + detail_level
        # todo: move function to parameters and replace func in experiment as well

    def create_density_maps(self):
        inverted_cmap = plt.cm.get_cmap('viridis').reversed()
        for region in self.p.regions:
            self.density_map(region, column_name='pop_density', maximum=4500, cmap=inverted_cmap)

    def create_area_type_maps(self):
        for region in self.p.regions:
            self.density_map(region, column_name='area_type_code')

    def create_infection_rate_maps(self):
        inverted_cmap = plt.cm.get_cmap('viridis').reversed()
        for region in self.p.regions:
            for date in self.p.dates:
                self.infection_rate_map(region, date, max=10_000, cmap=inverted_cmap)

    def create_utility_vs_pop_density_plot(self):
        path = self.base_path + 'utility_area_type/'
        mkdir(path)
        sql = '''select region, cast(population_count/surface_area_km2 as float) as pop_per_km2, 
        case when infection_count_true = 0 then 0. else cast(infection_count_error/infection_count_true as float) end 
        as rel_infection_count_error 
        from utility_node where user_type<30 and infection_count_true > 0;'''

        df = pd.read_sql(sql, self.db.engine)
        df = df[df['pop_per_km2'] != 0]
        df = df.sort_values(by=['pop_per_km2'], ascending=True)

        bins = [0, 300, 1500, 25000]
        df['pop_bin'] = pd.cut(df['pop_per_km2'], bins)
        labels = [str(e) for e in df['pop_bin'].unique()]
        x = []
        for i, bin in enumerate(pd.unique(df['pop_bin'])):
            x.append(df[df['pop_bin'] == bin]['rel_infection_count_error'].values)
        plt.figure(figsize=self.default_figsize_rect)
        plt.hist(x, bins=25, density=True, histtype='bar', stacked=True, color=self.colors[:len(x)],
                 label=labels)
        plt.xlabel('Relative Error', fontsize=self.axes_label_fontsize)
        plt.ylabel('Frequency', fontsize=self.axes_label_fontsize)
        plt.tick_params(axis='both', labelsize=self.tick_params_labelsize)
        if self.legend:
            plt.legend(title_fontsize=self.legend_fontsize, fontsize=self.legend_fontsize, title='Population per km2',
                       loc='upper center')
        save_to_path = path + "rel_infection_count_error_per_pop_size.pdf"
        plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def create_parameter_comparison_plot(self, user_types=None, shared_parameter_approach='default'):
        if user_types is None:
            user_types = self.get_user_type_codes(self.p.user_types)
        path = self.base_path + 'parameter_comparison/'
        mkdir(path)
        if self.db.exists('parameters'):
            self.db.drop('parameters')
        self.db.create_parameters_table(self.p.user_types)
        sql = "SELECT * FROM " + self.get_utility_table_name('total') + \
              " u INNER JOIN parameters p ON u.user_type = p.type WHERE " + \
              "shared_parameter_approach = '" + str(shared_parameter_approach) + "' and user_type in " + \
              str(user_types).replace('[', '(').replace(']', ')') + ";"
        df = pd.read_sql(sql, self.db.engine)
        df = df.sort_values(by=['h', 'p', 'k'], ascending=[False, False, True])
        for param_combination in [
            ('h_name', '7', 'p_name', '50-100%', 'k_name'),
            ('h_name', '7', 'k_name', '0-5', 'p_name'),
            # ('h_name', '4-7', 'p_name', '50-100%', 'k_name'),
            # ('h_name', '4-7', 'k_name', '0-5', 'p_name'),
        ]:
            param_1_name, param_1_value, param_2_name, param_2_value, param_x_name = param_combination
            # ymax = 0.5 if px == 'k' else None
            ymax = 1.1
            for column_name in ['fscore_ratio', 'rel_infection_count_error']:
                save_to_path = path + 'influence_params_' + column_name + '_' + \
                               '_'.join(list([param_wo_spec_char(str(e)) for e in param_combination])) + ".pdf"
                self.plot_column_vs_params(df, param_1_name, param_1_value, param_2_name, param_2_value, param_x_name,
                                           column_name, save_to_path, ymin=0, ymax=ymax)

    def create_utility_per_level(self, user_types=None, shared_parameter_approach='default'):
        if user_types is None:
            user_types = self.get_user_type_codes(self.p.user_types)
        for region in self.p.regions:
            for column_name in ['fscore_ratio', 'rel_infection_count_error']:
                path = self.base_path + 'utility_per_level/'
                mkdir(path)
                if self.db.exists('parameters'):
                    self.db.drop('parameters')
                self.db.create_parameters_table(self.p.user_types)
                # todo: eval just until certain level
                sql = "SELECT level, user_type, CONCAT('k=', p.k_name, ',h=', p.h_name, ',p=', p.p_name) as name, " + \
                      column_name + \
                      " FROM " + self.get_utility_table_name('level') + " u INNER JOIN parameters p ON " + \
                      "p.type = u.user_type " + \
                      " WHERE shared_parameter_approach = '" + str(shared_parameter_approach) + "' and " + \
                      "region = '" + region['name'] + "' and user_type in " + \
                      str(user_types).replace('[', '(').replace(']', ')') + ";"
                df = pd.read_sql(sql, self.db.engine)
                if len(df) > 0:
                    save_to_path = path + 'lineplot_' + region['name'] + '_' + column_name + ".pdf"

                    fig, ax = plt.subplots(figsize=self.default_figsize_rect)
                    df = df.sort_values(by=['name', 'level'])
                    markers = ['o', 'v', '*', 's', 'D', '<', '>', 'p', 'X', 'h']
                    i = 0
                    for name, group in df.groupby('name'):
                        group.plot(x='level', y=column_name, marker=markers[i % len(markers)], markersize=8,
                                   label=name, ax=ax, color=self.colors[i])
                        i += 1
                    if self.legend:
                        ax.legend(fontsize=self.legend_fontsize)
                    ax.set_xlabel('Level', fontsize=self.axes_label_fontsize)
                    ax.set_ylabel(get_column_name_beautiful(column_name), fontsize=self.axes_label_fontsize)
                    ax.set_ylim([0, 1.1])
                    plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
                    plt.close()

    def create_utility_per_level_all_regions(self, user_types=None, shared_parameter_approach='default'):
        if user_types is None:
            user_types = self.get_user_type_codes(self.p.user_types)
        path = self.base_path + 'utility_per_level/'
        mkdir(path)
        for column_name in ['fscore_ratio', 'rel_infection_count_error']:
            save_to_path = path + 'lineplot_' + column_name + "_all_regions.pdf"
            sql = "SELECT region, level, user_type, " + column_name + \
                  " FROM " + self.get_utility_table_name('level') + " u INNER JOIN parameters p ON " + \
                  "p.type = u.user_type " + \
                  " WHERE shared_parameter_approach = '" + str(shared_parameter_approach) + "' and " + \
                  " user_type in " + str(user_types).replace('[', '(').replace(']', ')') + ";"
            df = pd.read_sql(sql, self.db.engine)
            df = df.sort_values(by=['level'])
            fig, ax = plt.subplots(figsize=self.default_figsize_rect)
            markers = ['o', 'v', '*', 's', 'D', '<', '>', 'p', 'X', 'h']
            i = 0
            for region, group in df.groupby('region'):
                group.plot(x='level', y=column_name, marker=markers[i % len(markers)], markersize=8,
                           label=get_region_name_beautiful(region), ax=ax, color=self.colors[i])
                i += 1
            if self.legend:
                ax.legend(fontsize=self.legend_fontsize, loc='lower left')
            ax.set_xlabel('Level', fontsize=self.axes_label_fontsize)
            ax.set_ylabel(get_column_name_beautiful(column_name), fontsize=self.axes_label_fontsize)
            ax.set_ylim([0, 1.1])
            plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
            plt.close()

    def plot_fscore_versus_relative_error(self):
        path = self.base_path + 'utility_vs_error/'
        mkdir(path)
        save_to_path = path + 'relative_error_vs_fscore.pdf'
        sql = '''select p.*, u.region, u.rel_infection_count_error, u.fscore_ratio from utility_total u 
        left join parameters p on u.user_type=p.type where h=7 order by region, p desc;'''
        df = pd.read_sql(sql, self.db.engine)
        df = df.sort_values(['rel_infection_count_error'])
        unique_regions = df['region'].unique()
        for i, region in enumerate(unique_regions):
            subdf = df[(df['region'] == region)]
            plt.plot(subdf['rel_infection_count_error'], subdf['fscore_ratio'], 'o', markersize=8,
                     color=self.colors[i], label=get_region_name_beautiful(region))

        coefficients = np.polyfit(df['rel_infection_count_error'], df['fscore_ratio'], 2)
        polynomial = np.poly1d(coefficients)
        x_values = np.linspace(df['rel_infection_count_error'].min(), df['fscore_ratio'].max(), 100)
        y_values = polynomial(x_values)
        plt.plot(x_values, y_values, color=self.colors[len(unique_regions)], label='Polynomial Trendline')

        plt.xlabel('Relative Error', fontsize=self.axes_label_fontsize)
        plt.ylabel('F1-Score', fontsize=self.axes_label_fontsize)
        plt.legend(loc='lower left', fontsize=self.legend_fontsize)
        plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def create_utility_vs_privacy_plots(self, kind, user_types=None, shared_parameter_approach='default'):
        if user_types is None:
            user_types = self.get_user_type_codes(self.p.user_types)
        for region in self.p.regions:
            for column_name in ['fscore_ratio', 'rel_infection_count_error']:
                path = self.base_path + 'utility_vs_privacy/'
                mkdir(path)
                if self.db.exists('parameters'):
                    self.db.drop('parameters')
                self.db.create_parameters_table(self.p.user_types)
                sql = "SELECT * FROM " + self.get_utility_table_name('total') + \
                      " u INNER JOIN parameters p ON u.user_type = p.type " + \
                      " WHERE shared_parameter_approach = '" + str(shared_parameter_approach) + "' and " + \
                      "region = '" + region['name'] + "' and user_type in " + \
                      str(user_types).replace('[', '(').replace(']', ')') + ";"
                df = pd.read_sql(sql, self.db.engine)
                if len(df) > 0:
                    df = df.sort_values(by=['h', 'p', 'k'], ascending=[False, False, True])
                    ymax = 1.1  # if column_name == 'fscore_ratio' else 0.6
                    if kind == 'line':
                        save_to_path = path + 'lineplot_' + column_name + '_' + region['name'] + '.pdf'
                        self.create_line_plot(df, column_name, save_to_path, ymin=-0.05, ymax=ymax)
                    elif kind == 'bar':
                        save_to_path = path + 'barplot_' + column_name + '_' + region['name'] + '.pdf'
                        self.create_bar_plot(df, column_name, save_to_path, ymin=0, ymax=ymax)

    def create_utility_vs_privacy_box_plots(self):
        # Utility vs privacy box plot per region and h for fixed shared_parameter_approach
        for region in self.p.regions:
            for column_name in ['fscore_ratio', 'infection_count_error', 'infection_density_error',
                                'infection_ratio_error']:
                self.box_plot_utility_vs_privacy(column_name, region=region['name'],
                                                 shared_parameter_approach='default')

    def create_tree_heatmaps(self, shared_parameter_approach, level):
        sql = "DROP TABLE tree_list;"
        self.db.execute_with_new_connection(sql)
        sql = "SELECT table_name INTO tree_list " + \
              "FROM information_schema.tables " + \
              "WHERE table_name like 'tree%' and " + \
              "table_schema not in ('information_schema', 'pg_catalog') " + \
              "and table_name like '%_" + str(shared_parameter_approach) + "' " + \
              "and table_type = 'BASE TABLE';"
        self.db.execute_with_new_connection(sql)
        sql = "SELECT table_name FROM tree_list;"
        df = pd.read_sql(sql, self.db.engine)
        table_list = df['table_name'].values.tolist()
        table_list = [name for name in table_list if name.split('_')[3] in [str(e['type']) for e in self.p.user_types]]

        for tree_table_name in table_list:
            for column_name in ['sensitive_users_count', 'sensitive_users_ratio']:
                self.heatmap(tree_table_name, column_name, level)

    def heatmap(self, tree_table_name, column_name, level, cast_as_int=False):
        if column_name is None:
            return
        path = self.base_path + 'heatmaps/'
        mkdir(path)
        tree_params = tree_table_name.split('_')
        approach = tree_params[1]
        region = tree_params[2]
        date = tree_params[3] + '_' + tree_params[4] + '_' + tree_params[5]
        if approach == 'dipalda':
            user_type = tree_params[6]
            shared_parameter_approach = int(tree_params[7]) / 100.
        else:
            sql = "SELECT shared_parameter_approach, user_type FROM " + self.get_utility_table_name(
                'node') + " LIMIT 1;"
            df = pd.read_sql(sql, self.db.engine)
            user_type = df['user_type'].values[0]
            shared_parameter_approach = df['shared_parameter_approach'].values[0]
        save_to_path = path + tree_table_name + "_" + column_name + "_" + str(level) + ".pdf"
        column = "cast(" + column_name + " as int) as " + column_name if cast_as_int else column_name
        sql = "SELECT " + column + ", rectangle " + \
              "FROM tree_true_" + region + "_" + date + " t LEFT JOIN " + self.get_utility_table_name(
            'node') + " u ON " + \
              "t.level=u.level and t.position=u.position " + \
              "WHERE t.level = " + str(level) + " and u.region = '" + region + "'" + \
              " and u.shared_parameter_approach = '" + str(shared_parameter_approach) + "'" + \
              " and u.user_type = '" + str(user_type) + "';"
        gdf = gpd.read_postgis(sql, self.db.engine, geom_col='rectangle', crs='4236')
        if len(gdf) == 0:
            print("No data.")
        else:
            self.create_heatmap(gdf, column_name, tree_table_name, save_to_path)

    def density_map(self, region, column_name, maximum=None, cmap='viridis'):
        path = self.base_path + 'density_maps/'
        mkdir(path)
        list_region_alias = region['nuts_id']
        var = ''
        for i, reg in enumerate(list_region_alias):
            var += '''p."NUTS_ID" LIKE \'''' + reg + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = "SELECT p.* FROM population_density p WHERE " + var
        df = gpd.read_postgis(sql, self.db.engine, geom_col='geometry')
        df = df.to_crs(epsg=4326)
        fig, ax = plt.subplots(figsize=self.default_figsize_square)
        df.plot(column=column_name, legend=self.legend, cmap=cmap, vmin=0, vmax=maximum,
                figsize=self.default_figsize_square,
                ax=ax, legend_kwds={'shrink': 0.8})
        plt.tight_layout()
        plt.savefig(path + region['name'] + '_' + column_name + '.pdf', dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def population_count_map(self, region, maximum=None):
        path = self.base_path + 'population_count_maps/'
        mkdir(path)
        list_region_alias = region['nuts_id']
        var = ''
        for i, nutsid in enumerate(list_region_alias):
            var += '''"NUTS_ID" LIKE \'''' + nutsid + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = '''select count(*) as cnt, grd_id, st_union(grid_geom) as geometry
        from locations where ''' + var + ''' group by grd_id'''
        gdf = gpd.read_postgis(sql, self.db.engine, geom_col='geometry')
        gdf = gdf.to_crs(epsg=4326)

        inverted_cmap = plt.cm.get_cmap('viridis').reversed()
        fig, ax = plt.subplots(figsize=self.default_figsize_square)
        gdf.plot(column='cnt', legend=self.legend, cmap=inverted_cmap, vmin=0, vmax=maximum,
                 figsize=self.default_figsize_square,
                 ax=ax, legend_kwds={'shrink': 0.7})
        if self.legend:
            fig.axes[1].tick_params(labelsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        ax.set_box_aspect(1)
        plt.savefig(path + region['name'] + '_pop_count.pdf', dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

        # fig, ax = plt.subplots(figsize=self.figsize_heatmap)
        # gdf.plot(column='count', ax=ax, legend=True, cmap='Reds', alpha=0.5)
        # plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        # plt.close()

    def population_count_map_combined(self, regions, legend=True, maximum=None):
        path = self.base_path + 'population_count_maps/'
        mkdir(path)

        # Different width for last subplot with legend
        widths = [1] * (len(regions) - 1) + [1.25]
        fig, axes = plt.subplots(1, len(regions),
                                 figsize=(sum(widths) * self.default_figsize_square[0], self.default_figsize_square[1]),
                                 gridspec_kw={'width_ratios': widths})

        for i, (ax, region) in enumerate(zip(axes, regions)):
            list_region_alias = region['nuts_id']
            var = ''
            for j, nutsid in enumerate(list_region_alias):
                var += '''"NUTS_ID" LIKE \'''' + nutsid + '''\''''
                if j < len(list_region_alias) - 1:
                    var += ''' OR '''
            sql = '''select count(*) as cnt, grd_id, st_union(grid_geom) as geometry
            from locations where ''' + var + ''' group by grd_id'''
            gdf = gpd.read_postgis(sql, self.db.engine, geom_col='geometry')
            gdf = gdf.to_crs(epsg=4326)
            inverted_cmap = plt.cm.get_cmap('viridis').reversed()

            # Show legend only for the last subplot
            leg = False
            if i == len(regions) - 1:
                leg = legend

            gdf.plot(column='cnt', legend=leg, cmap=inverted_cmap, vmin=0, vmax=maximum,
                     figsize=self.default_figsize_square, ax=ax, legend_kwds={'shrink': 0.9})

            if leg:
                ax.figure.axes[-1].tick_params(labelsize=self.tick_params_labelsize)
            ax.set_title(get_region_name_beautiful(region['name']), fontsize=self.title_fontsize)
            ax.set_box_aspect(1)

        plt.tight_layout()
        plt.savefig(path + 'combined_pop_count.pdf', dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def area_type_map(self, region, column_name, maximum=None, cmap='viridis'):
        path = self.base_path + 'density_maps/'
        mkdir(path)
        list_region_alias = region['nuts_id']
        var = ''
        for i, reg in enumerate(list_region_alias):
            var += '''p."NUTS_ID" LIKE \'''' + reg + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = "SELECT p.* FROM population_density p WHERE " + var
        df = gpd.read_postgis(sql, self.db.engine, geom_col='geometry')
        df = df.to_crs(epsg=4326)
        unique_values = df[column_name].unique()
        unique_values.sort()
        min_value = 0
        max_value = max(max(unique_values), 1)
        plot = df.plot(column=column_name, legend=self.legend, cmap=cmap, vmin=0, vmax=maximum,
                       figsize=self.default_figsize_square)
        colors = plt.cm.get_cmap(cmap)
        if self.legend:
            legend_labels = {value: colors((value - min_value) / (max_value - min_value)) for value in unique_values}
            areas = [Patch(facecolor=color, label=label) for label, color in legend_labels.items()]
            plot.legend(handles=areas, fontsize=12)
        # plt.title(region['name'], fontsize=18)
        plt.savefig(path + region['name'] + '_' + column_name + '.pdf', dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def infection_rate_map(self, region, date, max=None, cmap='viridis'):
        path = self.base_path + 'covid_maps/'
        mkdir(path)

        list_region_alias = region['nuts_id']
        var = ''
        for i, reg in enumerate(list_region_alias):
            var += '''p."NUTS_ID" LIKE \'''' + reg + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''
        sql = "select * from covid_cnt_per_subregion c inner join population_density p " + \
              "on c.nuts_code=p.\"NUTS_ID\" WHERE " + var + ''' and date=\'''' + date + '''\';'''
        df = gpd.read_postgis(sql, self.db.engine, geom_col='geometry')
        df = df.to_crs(epsg=4326)

        fig, ax = plt.subplots(figsize=self.default_figsize_square)
        df.plot(column='rate_14_day_per_100k', legend=self.legend, cmap=cmap, vmin=0, vmax=max,
                figsize=self.default_figsize_square, ax=ax,
                legend_kwds={'shrink': 0.8})
        plt.savefig(path + region['name'] + '_' + date + '_covid_rate.pdf', dpi=self.dpi, format='pdf',
                    bbox_inches='tight')
        plt.close()

    def density_map_with_bounds(self, region, column_name):
        epsg_plotting = 4326
        epsg_calculation = 32632

        path = self.base_path + 'density_maps/'
        mkdir(path)

        list_region_alias = region['nuts_id']
        var = ''
        for i, reg in enumerate(list_region_alias):
            var += '''p."NUTS_ID" LIKE \'''' + reg + '''\''''
            if i < len(list_region_alias) - 1:
                var += ''' OR '''

        sql = "SELECT p.* FROM population_density p WHERE " + var
        df = gpd.read_postgis(sql, self.db.engine, geom_col='geometry')
        df = df.to_crs(epsg=epsg_calculation)
        minx, miny, maxx, maxy = df.total_bounds

        # Create points for the bounding box
        lower_left = (minx, miny)
        top_left = (minx, maxy)
        top_right = (maxx, maxy)
        lower_right = (maxx, miny)

        # Create bounding box
        bbox = gpd.GeoSeries([Point(lower_left), Point(top_left), Point(top_right), Point(lower_right)],
                             crs='EPSG:' + str(epsg_calculation))

        # Adjust to create a square bounding box
        width = maxx - minx
        height = maxy - miny
        max_extension = max(width, height)

        minx = (minx + maxx - max_extension) / 2
        maxx = minx + max_extension
        miny = (miny + maxy - max_extension) / 2
        maxy = miny + max_extension

        lower_left = (minx, miny)
        top_left = (minx, maxy)
        top_right = (maxx, maxy)
        lower_right = (maxx, miny)

        # Calculate square bounding box
        square_bbox = gpd.GeoSeries([Point(lower_left), Point(top_left), Point(top_right), Point(lower_right)],
                                    crs='EPSG:' + str(epsg_calculation))

        # draw region data
        df = df.to_crs(epsg=epsg_plotting)
        df.plot(column=column_name, legend=self.legend, cmap='viridis', vmin=0, vmax=None,
                figsize=self.default_figsize_square,
                legend_kwds={'shrink': 0.8})

        # draw bounding box in green
        bbox = bbox.to_crs(epsg=epsg_plotting)
        x_values = [point.x for point in bbox.geometry]
        y_values = [point.y for point in bbox.geometry]
        plt.scatter(x_values, y_values)
        plt.plot(x_values + [x_values[0]], y_values + [y_values[0]], color='green')

        # draw square bounding box in red
        square_bbox = square_bbox.to_crs(epsg=epsg_plotting)
        x_values = [point.x for point in square_bbox.geometry]
        y_values = [point.y for point in square_bbox.geometry]
        plt.scatter(x_values, y_values)
        plt.plot(x_values + [x_values[0]], y_values + [y_values[0]], color='red')

        plt.savefig(path + region['name'] + '_' + column_name + '.pdf', dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def box_plot_utility_vs_privacy(self, column_name, region, shared_parameter_approach):
        path = self.base_path + 'utility_vs_privacy/'
        sql = "SELECT " + column_name + ", region, user_type" + \
              " FROM " + self.get_utility_table_name('node') + " WHERE shared_parameter_approach = '" + \
              str(shared_parameter_approach) + "' and region = '" + region + "';"
        df = pd.read_sql(sql, self.db.engine).sample(n=1000)
        if len(df) > 0:
            fig, ax = plt.subplots(figsize=self.default_figsize_rect)
            df.boxplot(column=column_name, by=['user_type'], ax=ax, showfliers=True, color=self.colors)
            # ax.set_title(column_name + ' ' + region, fontsize=self.title_fontsize)
            ax.set_xlabel('User type', fontsize=self.axes_label_fontsize)
            ax.set_ylabel(column_name, fontsize=self.axes_label_fontsize)
            plt.xticks(rotation=45)
            plt.tight_layout()
            save_to_path = path + 'boxplot_' + column_name + '_' + region + '.pdf'
            plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
            plt.close()

    def create_line_plots(self, df, column_name, save_to_path, xmin=None, xmax=None, ymin=None, ymax=None):
        unique_groups_region = df['region'].unique()
        fig, axes = plt.subplots(ncols=len(unique_groups_region), nrows=1,
                                 figsize=(5 * len(unique_groups_region), 6))
        markers = ['o', 'v', '*', 's', 'D', '<', '>', 'p', 'X', 'h']
        for idx_r, region in enumerate(unique_groups_region):
            for i, p_name in enumerate(df['p_name'].unique()):
                subset = df[(df['region'] == region) & (df['p_name'] == p_name)]
                if len(unique_groups_region) > 1:
                    ax = axes[idx_r]
                else:
                    ax = axes
                ax.plot(subset['k'], subset[column_name], marker=markers[i % len(markers)], markersize=8,
                        label=f'p = {p_name}', color=self.colors[i])
                ax.tick_params(axis='both', labelsize=self.tick_params_labelsize)
                if self.legend:
                    ax.legend(fontsize=12)
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_title(get_region_name_beautiful(region) + ', h=8', fontsize=self.title_fontsize)
                ax.set_xlabel('k', fontsize=self.axes_label_fontsize)
                ax.set_ylabel(column_name, fontsize=self.axes_label_fontsize)
        plt.tight_layout()
        plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def create_line_plot(self, df, column_name, save_to_path, h_name='5', xmin=None, xmax=None, ymin=None, ymax=None):
        fig, ax = plt.subplots(figsize=self.default_figsize_square)
        markers = ['o', 'v', '*', 's', 'D', '<', '>', 'p', 'X', 'h']
        for i, p_name in enumerate(df['p_name'].unique()):
            subset = df[(df['h_name'] == h_name) & (df['p_name'] == p_name)]
            ax.plot(subset['k'], subset[column_name], marker=markers[i % len(markers)], markersize=8,
                    label=f'p = {p_name}', color=self.colors[i])
            ax.tick_params(axis='both', labelsize=self.tick_params_labelsize)
            if self.legend:
                ax.legend(fontsize=12)
            ax.set_xlim([xmin, xmax])
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel('k', fontsize=self.axes_label_fontsize)
            ax.set_ylabel(get_column_name_beautiful(column_name), fontsize=self.axes_label_fontsize)
        plt.tight_layout()
        plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def create_bar_plot(self, df, column_name, save_to_path, ymin=None, ymax=None):
        fig, ax = plt.subplots(figsize=self.default_figsize_square)
        bar_width = 0.15
        patterns = ['-', '.', '\\', '/', 'x', '|']
        unique_k_names = df['k_name'].unique()
        for i, p_name in enumerate(df['p_name'].unique()):
            subset = df[df['p_name'] == p_name]
            bar_positions = [unique_k_names.tolist().index(k_val) + i * bar_width for k_val in subset['k_name']]
            # bar_positions = [j + (i-1) * bar_width for j in subset['k']]
            hatch_pattern = patterns[i % len(patterns)] * 3
            ax.bar(bar_positions, subset[column_name], width=bar_width, label=f'p = {p_name}', hatch=hatch_pattern,
                   alpha=0.5, color=self.colors[i])
            ax.tick_params(axis='both', labelsize=self.tick_params_labelsize)
            x_offset = (len(unique_k_names) * bar_width) / 2.
            ax.set_xticks([x_offset + i for i in range(len(unique_k_names))])
            ax.set_xticklabels(unique_k_names)
            if self.legend:
                ax.legend(fontsize=self.legend_fontsize, loc='lower left')
            ax.set_xlim([-0.2, len(unique_k_names)])
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel('k', fontsize=self.axes_label_fontsize)
            ax.set_ylabel(get_column_name_beautiful(column_name), fontsize=self.axes_label_fontsize)
        plt.tight_layout()
        plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')

        # Create a new figure for the legend
        if not self.legend:
            handles, labels = ax.get_legend_handles_labels()
            fig_legend = plt.figure(figsize=(4, 2))
            ax_legend = fig_legend.add_subplot(111)
            ax_legend.legend(handles, labels, loc='center', ncol=len(handles))
            ax_legend.axis('off')  # Turn off the axis
            save_legend_to_path = save_to_path[:save_to_path.rfind('_')] + "_legend.pdf"
            plt.savefig(save_legend_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')

        plt.close()

    def create_bar_plots(self, df, column_name, save_to_path, h_name='8', xmin=None, xmax=None, ymin=None, ymax=None):
        unique_groups_region = df['region'].unique()
        fig, axes = plt.subplots(ncols=len(unique_groups_region), nrows=1,
                                 figsize=(5 * len(unique_groups_region), 6))
        bar_width = 1.5
        patterns = ['-', '.', '\\', '/', '+', 'x', 'o', 'O', '.', '*']
        for idx_r, region in enumerate(unique_groups_region):
            for i, p_name in enumerate(df['p_name'].unique()):
                subset = df[(df['region'] == region) & (df['h_name'] == h_name) & (df['p_name'] == p_name)]
                if len(unique_groups_region) > 1:
                    ax = axes[idx_r]
                else:
                    ax = axes

                bar_positions = [j + (i - 1) * bar_width for j in subset['k']]
                hatch_pattern = patterns[i % len(patterns)] * 3
                ax.bar(bar_positions, subset[column_name], width=bar_width, label=f'p = {p_name}',
                       hatch=hatch_pattern, color=self.colors)
                ax.set_title(get_region_name_beautiful(region), fontsize=self.title_fontsize)
                ax.tick_params(axis='both', labelsize=self.tick_params_labelsize)
                if self.legend:
                    ax.legend(self.legend_fontsize, loc='upper right')
                ax.set_xlim([xmin, xmax])
                ax.set_ylim([ymin, ymax])
                ax.set_xlabel('k', fontsize=self.axes_label_fontsize)
                ax.set_ylabel(get_column_name_beautiful(column_name), fontsize=self.axes_label_fontsize)

        # fig.suptitle(column_name + ' vs. k, p, and h', fontsize=18)
        plt.tight_layout()
        plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def plot_column_vs_params(self, df, param_1_name, param_1_value, param_2_name, param_2_value, param_x_name,
                              column_name,
                              save_to_path, xmin=None, xmax=None, ymin=None, ymax=None):
        unique_regions = df['region'].unique()
        unique_regions.sort()
        fig, ax = plt.subplots(figsize=self.default_figsize_rect)
        markers = ['o', 'v', '*', 's', 'D', '<', '>', 'p', 'X', 'h']
        nothing_to_plot = True
        for i, region in enumerate(unique_regions):
            subset = df[(df['region'] == region) & (df[param_1_name] == param_1_value) &
                        (df[param_2_name] == param_2_value)]
            if len(subset) > 0:
                nothing_to_plot = False
            ax.plot(subset[param_x_name], subset[column_name], marker=markers[i % len(markers)], markersize=8,
                    label=get_region_name_beautiful(region), color=self.colors[i])

        ax.set_xlabel(get_column_name_beautiful(param_x_name), fontsize=self.axes_label_fontsize)
        ax.set_ylabel(get_column_name_beautiful(column_name), fontsize=self.axes_label_fontsize)
        # ax.set_title(f'{column_name} vs. {param_x_name} for {param_1_name}={param_1_value} and '
        #              f'{param_2_name}={param_2_value}', fontsize=self.title_fontsize)
        ax.tick_params(axis='both', labelsize=self.tick_params_labelsize)
        if self.legend:
            ax.legend(fontsize=self.legend_fontsize, loc='lower left')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        plt.tight_layout()
        if not nothing_to_plot:
            plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.close()

    def create_bubble_plot(self, df):
        # todo
        sns.relplot(
            data=df,
            x='k', y='p',
            size='fscore', sizes=(40, 200),
            hue='fscore',
            palette='coolwarm',
        )
        plt.show()

    def create_heatmap(self, gdf, column_name, tree_table_name, save_to_path):
        if gdf.crs != "EPSG:3857":
            gdf = gdf.to_crs(epsg=3857)
        fig, ax = plt.subplots(figsize=self.figsize_heatmap)
        # cmap = mcolors.LinearSegmentedColormap.from_list("transparent_red", [(0, 0, 0, 0), (1, 0, 0, 1)])
        gdf.plot(column=column_name, ax=ax, legend=self.legend, cmap='Reds', alpha=0.5)
        ctx.add_basemap(ax, source=xyz.OpenStreetMap.DE.url)
        # ax.set_title(column_name + " from\n" + tree_table_name, fontsize=self.title_fontsize)
        # plt.savefig(save_to_path, dpi=self.dpi, format='pdf', bbox_inches='tight')
        plt.savefig(save_to_path, dpi=100, format='pdf', bbox_inches='tight')
        plt.close()
