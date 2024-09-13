# DIPALDA: Distributed Privacy-Aware Location Data Aggregation

### DIPALDA
Analyzing location data from many individuals can
provide valuable insights, especially when linked with private
attributes like personal health information. A recent application
includes identifying COVID-19 outbreaks by aggregating indi-
viduals’ health data across a geographical hierarchy. However,
analyzing such sensitive information can threaten the individuals’
privacy, especially when honest-but-curious third parties are
involved. To encourage people to share their data for such
analyses, strong privacy protection and building trust in the
privacy approach are crucial, requiring clear privacy parameters
that can be tailored to individual needs. To address these
requirements, we introduce DIPALDA, a new anonymization
technique for DIstributed, Privacy-Aware Location Data Ag-
gregation on hierarchically structured personal location data.
DIPALDA leverages three privacy parameters: k-anonymity,
minimum cloaking area size, and maximum re-identification
probability, effectively countering re-identification and location
privacy attacks. Our extensive experiments with COVID-19
propagation data demonstrate that DIPALDA achieves a suitable
trade-off between utility, privacy, and explainability.

### Build
#### Set up database
To set up a PostgreSQL database with PostGIS, first, update the Dockerfile with your credentials. Then, create and 
start the docker container by running
```bash
sudo docker build . -t postgis -f Dockerfile
sudo docker run --name dipalda -d -p 5438:5438 postgis
```
The database is running on localhost.

Test the database connection and list all available databases by running
```bash
sudo psql -l -p 5438 -h localhost -U user
```

If you want to explore the database, login to get a psql console by running
```bash
sudo psql -d dipalda -U user -p 5438 -h localhost
```

#### Get required resources:
1) [Download](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/administrative-units-statistical-units/nuts)
NUTS codes from EUROSTAT (File: NUTS_RG_20M_2021_3035.shp) and place it in resources folder.
2) [Download](https://ec.europa.eu/eurostat/web/gisco/geodata/reference-data/population-distribution-demography/geostat#geostat11)
census data from EUROSTAT (File: ESTAT_Census_2011_V1-0.gpkg) and place it in resources folder.


### License
*DIPALDA* is licensed under [Apache License, Version 2.0](LICENSE)



### Acknowledgement
*DIPALDA* is developed by University of Leipzig & ScaDS.AI Dresden/Leipzig, Germany, funded by the Federal Ministry of 
Education and Research of Germany and by Sächsische Staatsministerium für Wissenschaft, Kultur und Tourismus in the 
program Center of Excellence for AI-research ”Center for Scalable Data Analytics and Artificial Intelligence 
Dresden/Leipzig”, project identification number: ScaDS.AI.

### References
[[1]](resources/2024265013.pdf) Schneider et al. (2024), Distributed, Privacy-Aware Location Data Aggregation