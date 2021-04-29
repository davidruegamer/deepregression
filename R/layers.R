# from keras
as_constraint <- getFromNamespace("as_constraint", "keras")
compose_layer <- getFromNamespace("compose_layer", "keras")

create_layer <- function (layer_class, object, args = list()) 
{
  args$input_shape <- args$input_shape
  args$batch_input_shape = args$batch_input_shape
  args$batch_size <- args$batch_size
  args$dtype <- args$dtype
  args$name <- args$name
  args$trainable <- args$trainable
  args$weights <- args$weights
  constraint_args <- grepl("^.*_constraint$", names(args))
  constraint_args <- names(args)[constraint_args]
  for (arg in constraint_args) args[[arg]] <- as_constraint(args[[arg]])
  if (inherits(layer_class, "R6ClassGenerator")) {
    common_arg_names <- c("input_shape", "batch_input_shape", 
                          "batch_size", "dtype", "name", "trainable", "weights")
    py_wrapper_args <- args[common_arg_names]
    py_wrapper_args[sapply(py_wrapper_args, is.null)] <- NULL
    for (arg in names(py_wrapper_args)) args[[arg]] <- NULL
    r6_layer <- do.call(layer_class$new, args)
    python_path <- system.file("python", package = "deepregression")
    layers <- reticulate::import_from_path("layers", path = python_path)
    py_wrapper_args$r_build <- r6_layer$build
    py_wrapper_args$r_call <- reticulate::py_func(r6_layer$call)
    py_wrapper_args$r_compute_output_shape <- r6_layer$compute_output_shape
    layer <- do.call(layers$RLayer, py_wrapper_args)
    r6_layer$.set_wrapper(layer)
  }
  else {
    layer <- do.call(layer_class, args)
  }
  if (missing(object) || is.null(object)) 
    layer
  else invisible(compose_layer(object, layer))
}

tib_layer = function(input_dim, use_bias, la) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("TibLinearLasso", path = python_path)
  layers$TibLinearLasso(input_dim = input_dim, use_bias = use_bias, la = la)
}