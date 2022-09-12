# Electric Vehicles in Energy Communities: Investigating the Distribution Grid Hosting Capacity
BSc SSE Thesis of Daniil Aktanka


## Data source
The simulation data is not directly included in this git due to size limitations. In order to to run the simulation locally, follow these intructions:

1. Head over to the source website: https://data.open-power-system-data.org/household_data/
2. Use the following configuration for download:

![instructions](/datasource/data_source_settings.png)

5. Update relevant settings in relevant jupyter notebooks:

        da.folder_path = r"\\path\\"
        da.imp_procc("data_filename.csv", keep_cols")
