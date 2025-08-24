# Files
This folder represents our "data", which we use to train and validate our models. Here, our main focus is on the `mlogit_Train_wide` dataset, which represents information on the population surveyed and their choice of transportation mode. We include a metadata file for all present datasets called `mlogit_choice_data_dictionary.pdf` and a specific metadata file for the `mlogit_Train_wide.csv` dataset, namely `train_metadata.pdf`.

```bash
.
├── mlogit_Beef_long.csv
├── mlogit_Car_wide.csv
├── mlogit_Catsup_wide.csv
├── mlogit_choice_data_dictionary.pdf   # Metadata PDF from MLogit R pkg
├── mlogit_choice_data.pickle           # Combined binary data
├── mlogit_Cracker_wide.csv
├── mlogit_Electricity_wide.csv
├── mlogit_Fishing_wide.csv
├── mlogit_HC_wide.csv
├── mlogit_Heating_wide.csv
├── mlogit_Ketchup_wide.csv
├── mlogit_MobilePhones_long.csv
├── mlogit_Mode_wide.csv
├── mlogit_ModeCanada_long.csv
├── mlogit_Telephone_long.csv
├── mlogit_TollRoad_wide.csv
├── mlogit_Train_wide.csv
├── mlogit_Tuna_wide.csv
├── README.md                           # This file!
└── train_metadata.pdf                  # Metadata for mlogit_Train_wide.csv
```

# Generation
While we provide you here with all the `CSV` files, we also want to make sure that you can recreate it yourself. For that, please make sure that you have the `R` language installed (you can find a how-to [here](https://www.r-project.org/)).

