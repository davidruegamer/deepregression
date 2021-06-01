#' Function to check python environment and install necessary packages
#'
#' Note: The package currently relies on tensorflow version 2.0.0 which is
#' not available for the latest python versions 3.9 and later.
#' If you encounter problems with installing the required python modules
#' please make sure, that a correct python version is configured using
#' `py_discover_config` and change the python version if required.
#' Internally uses keras::install_keras.
#'
#' @param force if TRUE, forces the installations
#' @return Function that checks if a Python environment is available
#' and contains TensorFlow. If not the recommended version is installed.
#'
#' @export
check_and_install <- function(force = FALSE) {
  if (!reticulate::py_module_available("tensorflow") || force) {
    keras::install_keras(version = "2.4.3", tensorflow = "2.1", 
                         extra_packages = c("tfprobability==0.9", "six")) # nocov
  } else {
    message("Tensorflow found, skipping tensorflow installation!")
    if (!reticulate::py_module_available("tensorflow_probability") || 
        !reticulate::py_module_available("six")) {
      message("Installing pytho modules 'tfprobability' and 'six'") # nocov
      reticulate::py_install(packages = c("tensorflow-probability==0.9", "six")) # nocov
    }
  }
}
