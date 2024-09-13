class Parameters:
    def __init__(self):

        p_range = [(0.5, 1), (0.1, 0.5), (0.05, 0.1), (0.03, 0.05), (0.01, 0.03), (0, 0.01)]
        k_range = [(0, 5), (5, 20), (20, 40), (40, 60), (60, 100)]
        h_range = [(7, 7), (4, 7)]
        self.user_types = [
            {'type': i,
             'k': {'lower': k[0], 'upper': k[1]},
             'p': {'lower': p[0], 'upper': p[1]},
             'h': {'lower': h[0], 'upper': h[1]}
             } for i, (k, p, h) in enumerate([(k, p, h) for h in h_range for k in k_range for p in p_range])]

        self.max_level = 7

        # low_covid_rate_date = '2020-09-25'
        medium_covid_rate_date = '2021-05-26'
        high_covid_rate_date = '2022-01-28'
        self.dates = [medium_covid_rate_date, high_covid_rate_date]

        self.regions = [
            {'country': 'DE', 'name': 'berlin', 'nuts_id': ['DE300'], 'type': 'urban'},
            {'country': 'IT', 'name': 'firenzearea', 'nuts_id': ['ITI1%%'], 'type': 'mixed'},
            {'country': 'DK', 'name': 'jylland', 'nuts_id': ['DK04%%'], 'type': 'rural'},
            {'country': 'IT', 'name': 'milano', 'nuts_id': ['ITC4C'], 'type': 'urban'},
            {'country': 'DE', 'name': 'colognearea', 'nuts_id': ['DEA1%%', 'DEA2%%', 'DEA3%%', 'DEA4%%'], 'type': 'mixed'},
            {'country': 'DE', 'name': 'luneburgarea', 'nuts_id': ['DE93%%'], 'type': 'rural'},
        ]

        self.countries = list(set([e['country'] for e in self.regions]))

        # Calculation approach of shared privacy parameters
        # 'default': fixed k and p as average of user type value range
        # 'ratio': deviations = [0.1, 0.05, 0, -0,05] (from more relaxed to more strict)
        #          estimate k per level as (count in root / M^h) * (1 - deviation)
        #          estimate p per level as (count in root / M^h) / population count in node of level h * (1 + deviation)
        self.shared_parameter_approaches = ['default', 0.05, 0.1, 0.15]


        # https://www.bundesregierung.de/breg-de/themen/coronavirus/corona-regeln-und-einschrankungen-1734724
        # "Für die vereinbarten Öffnungsschritte wurde als Voraussetzung vereinbart, dass in dem Land oder der Region
        # eine stabile oder sinkende 7-Tage-Inzidenz von unter 100 Neuinfektionen pro 100.000 EinwohnerInnen erreicht
        # wird."
        self.infection_rate_threshold = 100. / 100_000

        # Average population density in Germany 214 inhabitants per square kilometer.
        # threshold of infection rate per square kilometer as of which a vertex will be classified as a hotspot
        self.average_population_density_per_km2 = 214
        self.hotspot_threshold_density = self.infection_rate_threshold * self.average_population_density_per_km2
        # hotspot_threshold_density = 0.214 -> equals one infection per 4.7 km2

        # threshold of infection rate per 100.000 inhabitants as of which a vertex will be classified as a hotspot
        self.hotspot_threshold_ratio = self.infection_rate_threshold
