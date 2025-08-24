# Created on Mon Aug 18 11:26:00 2025
# Export datasets from the R library `mlogit` v0.2-4
# Adapted from: https://github.com/cjsyzwsh/ASU-DNN/blob/master/code/0_export_data.R

# Function to install packages if not already installed
install_if_missing <- function(packages) {
    for (pkg in packages) {
        if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
            if (pkg == "mlogit") {
                mlogit_url <- "https://cran.r-project.org/src/contrib/Archive/mlogit/mlogit_0.2-4.tar.gz"
                install.packages(mlogit_url, repos = NULL, type = "source")
            } else {
                install.packages(pkg)
            }
        }
    }
}

# Install and load required packages
required_packages <- c("maxLik", "mlogit", "AER")
install_if_missing(required_packages)

# Load libraries
library(mlogit)
library(AER)

# Set working directory to the parent of the script's directory
this_file <- parent.frame(2)$ofile
if (is.null(this_file)) this_file <- sys.frames()[[1]]$ofile
if (is.null(this_file)) this_file <- commandArgs(trailingOnly = FALSE)[grep("--file=", commandArgs(trailingOnly = FALSE))]
if (length(this_file) > 0) {
  script_dir <- dirname(normalizePath(sub("--file=", "", this_file[1])))
  setwd(dirname(script_dir))
}

# Create output directory if it doesn't exist
output_dir <- "../data"
if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
}

# Define dataset configurations
datasets <- list(
    list(name = "Beef", format = "long"),
    list(name = "Car", format = "wide"),
    list(name = "Catsup", format = "wide"),
    list(name = "Cracker", format = "wide"),
    list(name = "Electricity", format = "wide"),
    list(name = "Fishing", format = "wide"),
    list(name = "HC", format = "wide"),
    list(name = "Heating", format = "wide"),
    list(name = "Ketchup", format = "wide"),
    list(name = "MobilePhones", format = "long"),
    list(name = "Mode", format = "wide"),
    list(name = "ModeCanada", format = "long"),
    list(name = "Telephone", format = "long"),
    list(name = "TollRoad", format = "wide"),
    list(name = "Train", format = "wide"),
    list(name = "Tuna", format = "wide")
)

# Load and export datasets
cat("Loading and exporting datasets...\n")
for (dataset in datasets) {
    tryCatch(
        {
            # Load dataset
            data(list = dataset$name, package = "mlogit")

            # Get the dataset
            data_obj <- get(dataset$name)

            # Create filename
            filename <- file.path(output_dir, paste0("mlogit_", dataset$name, "_", dataset$format, ".csv"))

            # Export to CSV
            write.csv(data_obj, filename, row.names = FALSE)
            cat("Exported:", basename(filename), "\n")
        },
        error = function(e) {
            cat("Error processing", dataset$name, ":", e$message, "\n")
        }
    )
}

cat("Data export completed!\n")
