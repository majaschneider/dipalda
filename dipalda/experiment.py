import numpy as np
import pandas as pd

import datastructure_db as ds


class Experiment:

    def __init__(self, db, date, region, user_type, shared_parameter_approach, max_level):
        self.db = db
        self.region = region
        srid = 32632
        bounds = self.db.get_total_bounds(self.get_list_region_alias(), srid)
        self.bounding_box = ds.Rectangle(bounds, srid)
        self.bounding_box.make_bbox_square()
        self.user_type = user_type
        self.shared_parameter_approach = shared_parameter_approach
        self.max_level = max_level
        self.date = date
        self.df_ground_truth_tree = self.get_tree_df(self.get_ground_truth_tree_table_name())
        self.df_dipalda_tree = self.get_tree_df(self.get_dipalda_tree_table_name())
        self.df_dp_tree = self.get_tree_df(self.get_dp_tree_table_name())

    def get_user_parameters(self):
        h_min = self.user_type['h']['lower']
        h_max = self.user_type['h']['upper']
        k_min = self.user_type['k']['lower']
        k_max = self.user_type['k']['upper']
        p_min = self.user_type['p']['lower']
        p_max = self.user_type['p']['upper']
        return h_min, h_max, k_min, k_max, p_min, p_max

    def get_date_name(self):
        return self.date.replace('-', '_')

    def get_list_region_alias(self):
        return self.region['nuts_id']

    def get_points_table_name(self):
        return 'points_' + self.region['name'] + "_" + self.get_date_name() + "_" + \
            str(self.user_type['type'])

    def get_ground_truth_tree_table_name(self):
        return 'tree_true_' + self.region['name'] + '_' + self.get_date_name()

    def get_dp_tree_table_name(self):
        return 'tree_dp_' + self.region['name'] + '_' + self.get_date_name()

    def get_utility_table_name(self, detail_level):
        return 'utility_' + detail_level

    def get_dipalda_tree_table_name(self):
        if type(self.shared_parameter_approach) is float:
            approach = str(int(self.shared_parameter_approach * 100)).replace('-', '_')
        else:
            approach = str(self.shared_parameter_approach)

        return 'tree_dipalda_' + self.region['name'] + '_' + self.get_date_name() + '_' + \
            str(self.user_type['type']) + '_' + approach

    def get_tree_df(self, tree_table_name):
        return self.db.get_tree_df(tree_table_name)

    def calculate_shared_privacy_parameters(self, approach, user_parameters):
        h_min, h_max, k_min, k_max, p_min, p_max = user_parameters
        k_per_level = []
        p_per_level = []
        if approach == 'default':
            for h in range(0, h_max + 1):
                k_per_level.append(float((k_max + k_min) / 2.))
                p_per_level.append(float((p_max + p_min) / 2.))
        else:
            # estimate k per level as min[max(users' k), count in root / M^h) * (1 - deviation)]
            # estimate p per level as max[min(users' p), (count in root / M^h) / population count in node of
            # level h * (1 + deviation)
            sql = '''select count(*) as count from ''' + self.get_points_table_name() + ''' where covid = TRUE;'''
            df = pd.read_sql(sql, self.db.engine)
            number_sensitive_users_in_root = df['count'].values[0]
            sql = '''select count(*) as count from ''' + self.get_points_table_name() + ''';'''
            df = pd.read_sql(sql, self.db.engine)
            number_users_in_root = df['count'].values[0]
            degree = 4
            for h in range(0, h_max + 1):
                estimated_number_sensitive_users_in_node = float(number_sensitive_users_in_root) / np.power(degree, h)
                estimated_number_users_in_node = float(number_users_in_root) / np.power(degree, h)
                if estimated_number_sensitive_users_in_node >= k_max:
                    k = k_max
                else:
                    k = int(estimated_number_sensitive_users_in_node * (1 - approach))
                estimated_p = float(estimated_number_sensitive_users_in_node / estimated_number_users_in_node)
                if estimated_p <= p_min:
                    p = p_min
                else:
                    p = float(estimated_p * (1 + approach))
                k_per_level.append(k)
                p_per_level.append(p)

        return k_per_level, p_per_level

    def print_region_statistics(self):
        bounding_box = ds.Rectangle(self.db.get_total_bounds(self.get_list_region_alias()), srid=4326)
        max_extension = bounding_box.max_extension_meters()
        print('\n##### Region statistics #####')
        print('Region name:', self.region['name'])
        print('Nuts codes:', self.get_list_region_alias())
        print('Extension:', int(max_extension) / 1_000, 'km')
        pd.set_option('display.max_columns', None)
        print('Covid statistics:')
        df_covid_stats = self.db.get_covid_stats(self.get_list_region_alias())
        print(df_covid_stats)
        df_covid_stats_subregion = self.db.get_covid_stats_per_subregion(self.get_list_region_alias())
        print(df_covid_stats_subregion)
        print('Population statistics:')
        df_stats, df_sum = self.db.get_region_stats(self.get_list_region_alias())
        print(df_sum)
        print('Subregion statistics:')
        print(df_stats)

    def create_ground_truth_tree(self):
        true_tree = ds.QuadTree(self.bounding_box, self.get_ground_truth_tree_table_name(), self.db)
        true_tree.insert(self.get_points_table_name(), k_per_level=None,
                         p_per_level=None, max_level=self.max_level, save_points=True, always_divide=True)
        return true_tree

    def create_dipalda_tree(self, k_per_level, p_per_level):
        dipalda_tree = ds.QuadTree(self.bounding_box, self.get_dipalda_tree_table_name(), self.db)
        dipalda_tree.insert(self.get_points_table_name(), k_per_level, p_per_level,
                            max_level=self.max_level)
        self.create_empty_dipalda_tree_if_not_exists()
        return dipalda_tree

    def create_empty_dipalda_tree_if_not_exists(self):
        if not self.db.exists(self.get_dipalda_tree_table_name()):
            node_item = {'level': 'int', 'position': 'int', 'name': 'str',
                         'is_divided': 'bool', 'k': 'int', 'p': 'float',
                         'is_k_valid': 'bool',
                         'is_p_valid': 'bool', 'population_count': 'int',
                         'sensitive_users_count': 'int', 'valid_sensitive_users_count': 'int',
                         'sensitive_users_ratio': 'float',
                         'surface_area_m2': 'float', 'points': 'geometry',
                         'rectangle': 'geometry'}
            df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in node_item.items()})
            df.to_sql(name=self.get_dipalda_tree_table_name(), con=self.db.engine, index=False)

    def create_dp_tree(self):
        """Create DP-tree as in Liu et al. 2022."""
        epsilon = 1.
        privacy_ratio = np.power(2, 1. / 3)
        theta = 0.3
        t = 0.00004 * self.db.get_nr_points(self.region['nuts_id'])
        root_epsilon = ds.calculate_root_epsilon(epsilon, privacy_ratio, self.max_level)
        if (self.db.exists(self.get_ground_truth_tree_table_name())
                and not self.db.exists(self.get_dp_tree_table_name())):
            self.db.copy(self.get_ground_truth_tree_table_name(), self.get_dp_tree_table_name())
        dp_tree = ds.DPQuadTree(self.bounding_box, self.get_dp_tree_table_name(), self.db)
        dp_tree.insert(self.get_points_table_name(), self.max_level, root_epsilon,
                       privacy_ratio)
        dp_tree.adjust_count(t, theta)

        self.db.update_tree(self.get_dp_tree_table_name(), dp_tree)
        return dp_tree

    def evaluate_utility(self, tree_table_name, max_evaluation_level, threshold_ratio, threshold_density):
        """
        Calculate for each node:
            - infection count error (number of infections - true minus anonymized value),
            - infection density error (number of infections per square kilometer - true minus anonymized value),
            - infection ratio error (number of infections per 100.000 inhabitants - absolute true minus anonymized value),
            - disease hotspot detection (based on true/false positives/negatives of each node's classification as hotspot,
            where a node is a hotspot when
                    - its infection rate, or
                    - its infection density
            is above a certain threshold).
        Calculate for each level as well as the whole tree:
            - relative infection count error (infection count error divided by true infection count),
            - average infection density error
            - average infection ratio error
            - precision, recall, accuracy and f-score.
        """
        conn = self.db.get_new_connection()

        # infections_true = number of true infections
        # infections_priv = number of anonymized infections
        # surface_area_km2 = surface area in square kilometers
        # population_count = number of inhabitants
        # infection_density_true = number of true infections per square kilometer
        # infection_density_priv = number of anonymized infections per square kilometer
        # infection_ratio_true = number of true infections divided by population count
        # infection_ratio_priv = number of anonymized infections divided by population count
        # hot_dens_true = density based true hotspot classification
        # hot_dens_priv = density based anonymized hotspot classification
        # hot_ratio_true = ratio based true hotspot classification
        # hot_ratio_priv = ratio based anonymized hotspot classification
        tmp_table_name = 'tmp_metrics_' + tree_table_name
        tree_results_table_name_node = 'utility_node_' + tree_table_name
        tree_results_table_name_level = 'utility_level_' + tree_table_name
        tree_results_table_name_total = 'utility_total_' + tree_table_name

        sql = "SELECT level, position, name, is_divided, k, p, is_k_valid, is_p_valid, is_in_tree, " + \
              "infection_count_true, infection_count_priv, surface_area_km2, population_count, " + \
              "ABS(infection_count_true - infection_count_priv) as infection_count_error, " + \
              "case when infection_count_true = 0 then 0. else " + \
              "cast(ABS(infection_count_true - infection_count_priv) as float)/cast(infection_count_true as float) " + \
              "end as rel_infection_count_error, " + \
              "infection_count_true/surface_area_km2 as infection_density_true, " + \
              "infection_count_priv/surface_area_km2 as infection_density_priv, " + \
              "abs(infection_count_priv - infection_count_true)/surface_area_km2 as infection_density_error, " + \
              "case when population_count = 0 then 0. else infection_count_true/population_count end " + \
              "as infection_ratio_true, " + \
              "case when population_count = 0 then 0. else infection_count_priv/population_count end " + \
              "as infection_ratio_priv, " + \
              "case when population_count = 0 then 0. else " + \
              "abs(infection_count_true - infection_count_priv)/population_count end as infection_ratio_error, " + \
              "infection_count_true/surface_area_km2 >= " + str(threshold_density) + " as hot_dens_true, " + \
              "infection_count_priv/surface_area_km2 >= " + str(threshold_density) + " as hot_dens_priv, " + \
              "case when population_count = 0 then 0. else infection_count_true/population_count end >= " + str(threshold_ratio) + " as hot_ratio_true, " + \
              "case when population_count = 0 then 0. else infection_count_priv/population_count end >= " + str(threshold_ratio) + " as hot_ratio_priv " + \
              "INTO " + tmp_table_name + \
              " FROM (" + \
              "SELECT gt.level, gt.position, gt.name, t.is_divided, t.k, t.p, t.is_k_valid, t.is_p_valid, " + \
              "gt.sensitive_users_count as infection_count_true, " + \
              "case when t.valid_sensitive_users_count is null then 0. else t.valid_sensitive_users_count end as " + \
              "infection_count_priv, " + \
              "gt.surface_area_m2/(1000*1000.) as surface_area_km2, " + \
              "cast(gt.population_count as float) as population_count, " + \
              "t.level is not null as is_in_tree " + \
              "FROM " + self.get_ground_truth_tree_table_name() + " AS gt LEFT JOIN " + \
              tree_table_name + " AS t ON gt.level = t.level AND gt.level <= " + str(max_evaluation_level) + \
              " AND gt.position=t.position) as a;"
        self.db.execute_with_existing_connection(sql, conn)

        # true/false positives/negatives for hotspot detection (ratio and density based)
        sql = "SELECT *, "
        for t in ['dens', 'ratio']:
            sql += "cast(hot_" + t + "_true and hot_" + t + "_priv as integer) as tp_" + t + ", " + \
                   "cast(not hot_" + t + "_true and hot_" + t + "_priv as integer) as fp_" + t + ", " + \
                   "cast(hot_" + t + "_true and not hot_" + t + "_priv as integer) as fn_" + t + ", " + \
                   "cast(not hot_" + t + "_true and not hot_" + t + "_priv as integer) as tn_" + t + ", "
        sql = sql[:-2] + " INTO " + tree_results_table_name_node + " FROM " + tmp_table_name + ";"
        self.db.execute_with_existing_connection(sql, conn)

        sql = "DROP TABLE " + tmp_table_name + ";"
        self.db.execute_with_existing_connection(sql, conn)

        # rel_infection_count_error = relative infection count error
        # avg_infection_density_error = average infection density error
        # avg_infection_ratio_error = average infection ratio error
        # precision, recall, accuracy and f-score for hotspot detection (ratio and density based)
        sql = "SUM(infection_count_true) as infection_count_true, " + \
              "SUM(infection_count_priv) as infection_count_priv, " + \
              "SUM(infection_count_error) as infection_count_error, " + \
              "case when SUM(infection_count_true) = 0 then 0. else " + \
              "cast(SUM(infection_count_error)/SUM(infection_count_true) as float) end as " + \
              "rel_infection_count_error, " + \
              "SUM(surface_area_km2) as surface_area_km2, " + \
              "SUM(population_count) as population_count, " + \
              "SUM(infection_density_error) as infection_density_error_sum, " + \
              "AVG(infection_density_error) as infection_density_error_avg, " + \
              "SUM(infection_ratio_error) as infection_ratio_error_sum, " + \
              "AVG(infection_ratio_error) as infection_ratio_error_avg, "
        for t in ['dens', 'ratio']:
            sql += "SUM(tp_" + t + ") as tp_" + t + ", " + \
                   "SUM(fp_" + t + ") AS fp_" + t + ", " + \
                   "SUM(tn_" + t + ") as tn_" + t + ", " + \
                   "SUM(fn_" + t + ") AS fn_" + t + ", " + \
                   "case when SUM(tp_" + t + ") + SUM(fn_" + t + ") = 0 then 1. else " + \
                   "SUM(tp_" + t + ")/(SUM(tp_" + t + ") + SUM(fn_" + t + ") + 0.) end as precision_" + t + ", " + \
                   "case when SUM(tp_" + t + ") + SUM(fp_" + t + ") = 0 then 1. else " + \
                   "SUM(tp_" + t + ")/(SUM(tp_" + t + ") + SUM(fp_" + t + ") + 0.) end as recall_" + t + ", " + \
                   "case when (SUM(tp_" + t + ") + SUM(tn_" + t + ") + " + \
                   "SUM(fp_" + t + ") + SUM(fn_" + t + ") + 0.) = 0 then 1. else " + \
                   "(SUM(tp_" + t + ") + SUM(tn_" + t + "))/(SUM(tp_" + t + ") + SUM(tn_" + t + ") + " + \
                   "SUM(fp_" + t + ") + SUM(fn_" + t + ") + 0.) end as accuracy_" + t + ", " + \
                   "case when SUM(tp_" + t + ") + SUM(fp_" + t + ") + SUM(fn_" + t + ") = 0 then 1. else " + \
                   "2 * SUM(tp_" + t + ")/(2 * SUM(tp_" + t + ") + SUM(fp_" + t + ") + SUM(fn_" + t + ") + 0.) " + \
                   "end as fscore_" + t + ", "

        sql_level = "SELECT level, " + sql[:-2] + " INTO " + tree_results_table_name_level + " FROM " + \
                    tree_results_table_name_node + " GROUP BY level;"
        self.db.execute_with_existing_connection(sql_level, conn)

        sql_tree = "SELECT " + sql[:-2] + " INTO " + tree_results_table_name_total + " FROM " + \
                   tree_results_table_name_node + ";"
        self.db.execute_with_existing_connection(sql_tree, conn)

        # add results to a single table and drop temp tree tables
        for detail_level in ['node', 'level', 'total']:
            results_tree_table_name = self.get_utility_table_name(detail_level) + '_' + tree_table_name
            table_cols = "'" + self.region['name'] + "' as region, " + str(self.user_type['type']) +\
                         " as user_type, '" + str(self.shared_parameter_approach) + \
                         "' as shared_parameter_approach, " + self.get_date_name() + " as date, t.* "

            if not self.db.exists(self.get_utility_table_name(detail_level)):
                sql = "SELECT " + table_cols + "INTO " + self.get_utility_table_name(detail_level) + " FROM " + \
                      results_tree_table_name + " as t;"
            else:
                sql = "INSERT INTO " + self.get_utility_table_name(detail_level) + \
                      " SELECT " + table_cols + " FROM " + results_tree_table_name + " as t;"
            self.db.execute_with_existing_connection(sql, conn)

            sql = "DROP TABLE " + results_tree_table_name + ";"
            self.db.execute_with_existing_connection(sql, conn)

        self.db.close_connection(conn)
