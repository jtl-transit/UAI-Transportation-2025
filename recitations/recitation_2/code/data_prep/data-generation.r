# Created on Mon Aug 18 11:26:00 2025
# Export Train dataset from the R library `mlogit` v0.2-4

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

# Get the directory of this script
this_file <- sub("^--file=", "", commandArgs(trailingOnly = FALSE)[grep("^--file=", commandArgs(trailingOnly = FALSE))][1])
script_dir <- dirname(normalizePath(this_file))

# Create output directory (recitation_2/data) if it doesn't exist
output_dir <- file.path(dirname(script_dir), "data")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load and export Train dataset
cat("Loading and exporting Train dataset...\n")
data(Train, package = "mlogit")
filename <- file.path(output_dir, "mlogit_Train_wide.csv")
write.csv(Train, filename, row.names = FALSE)
cat("Exported:", filename, "\n")
cat("Data export completed!\n")
