import sys

from experiment import Experiment
import parameters
from dipalda.database import Database
from plot import Plot


if __name__ == '__main__':
    args = sys.argv[1:]

    task = args[0]

    host = args[1]
    port = args[2]
    database = args[3]
    user = args[4]
    password = args[5]
    db = Database(host, port, database, user, password)
    p = parameters.Parameters()

    plot = Plot(db, base_path='plots/', parameters=p)
    exp = None
    if len(args) >= 9:
        date_id = int(args[6])
        region_id = int(args[7])
        user_type_id = int(args[8])
        shared_parameter_approach_id = int(args[9])
        exp = Experiment(db=db,
                         date=p.dates[date_id],
                         region=p.regions[region_id],
                         user_type=p.user_types[user_type_id],
                         shared_parameter_approach=p.shared_parameter_approaches[shared_parameter_approach_id],
                         max_level=p.max_level)

    max_evaluation_level = int(args[10]) if len(args) >= 10 else p.max_level

    if task == 'import_data':
        db.read_data_sets_into_db()
    elif task == 'preprocess_data':
        db.data_pre_processing(p)
    elif task == 'create_parameters_table':
        db.create_parameters_table(p.user_types)
    elif task == 'create_points_table':
        if not db.exists(exp.get_points_table_name()):
            db.create_points_table(points_table_name=exp.get_points_table_name(),
                                   list_region_alias=exp.get_list_region_alias(),
                                   date_name=exp.get_date_name(),
                                   user_parameters=exp.get_user_parameters())
    elif task == 'create_ground_truth_tree':
        true_tree = exp.create_ground_truth_tree()
    elif task == 'create_dipalda_tree':
        k_per_level, p_per_level = exp.calculate_shared_privacy_parameters(exp.shared_parameter_approach,
                                                                           exp.get_user_parameters())
        dipalda_tree = exp.create_dipalda_tree(k_per_level, p_per_level)
    elif task == 'evaluate_utility':
        df, _ = exp.db.get_region_stats(exp.get_list_region_alias())
        average_population_density_per_km2 = df['pop_cnt'].sum() / df['surface_area_km2'].sum()
        hotspot_threshold_density = average_population_density_per_km2 * p.hotspot_threshold_ratio
        exp.evaluate_utility(exp.get_dipalda_tree_table_name(), max_evaluation_level, p.hotspot_threshold_ratio,
                             hotspot_threshold_density)
    elif task == 'create_dp_tree':
        dp_tree = exp.create_dp_tree()
    elif task == 'print_region_statistics':
        for region in p.regions:
            exp.region = region
            exp.print_region_statistics()
    else:
        pass

    db.close_connection_pool()
