# Electric Vehicles in Energy Communities: Investigating the Distribution Grid Hosting Capacity
BSc SSE Thesis of Daniil Aktanka


## Data source
The simulation data is not directly included in this git due to size limitations. In order to to run the simulation locally, follow these intructions:

1. Head over to the [Open Power Data source website](https://data.open-power-system-data.org/household_data/).
2. Using the following configuration, download the *household_data_1min_singleindex_filtered.csv* file:

![instructions](/datasource/data_source_settings.png)

5. Update relevant settings in relevant jupyter notebooks:

        da.folder_path = r"\\path\\"
        da.imp_procc("data_filename.csv", keep_cols)

## Bulk simulation results
Bulk simulation results used for the final thesis comparison of scenarios can be found in the */results/backup/* folder. They contain data from 3 scenarios, each consisting of 100 simulations.
