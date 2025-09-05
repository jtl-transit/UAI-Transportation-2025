# Files
This folder represents our "data", which we use to train and validate our models. Here, our main focus is on the `mlogit_Train_wide` dataset, which represents information on the population surveyed and their choice of transportation mode. We include a metadata file for all present datasets called `mlogit_choice_data_dictionary.pdf` and a specific metadata file for the `mlogit_Train_wide.csv` dataset, namely `train_metadata.pdf`.

```bash
.
├── mlogit_choice_data_dictionary.pdf   # Metadata PDF from MLogit R pkg
├── mlogit_Train_wide.csv               # Wide-format Train choice data
├── README.md                           # This file!
└── train_metadata.pdf                  # Metadata for mlogit_Train_wide.csv
```

# Generation
While we provide you here with the `CSV` file for the train data, we also want to make sure that you can recreate it yourself. For that, please make sure that you have the `R` language installed (you can find a how-to [here](https://www.r-project.org/)) as well as `Python`. You can then run the files in the [`data_prep` folder](../code/data_prep/). To obtain the same CSV file as shared here, you will want to run the [`R` script](../code/data_prep/data-generation.r), which will download the wide-format data.
