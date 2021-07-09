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

tib_layer = function(input_dim, units, use_bias, la, ...) {
  python_path <- system.file("python", package = "deepregression")
  layers <- reticulate::import_from_path("layers", path = python_path)
  if(units == 1) return(
    layers$TibLinearLasso(input_dim = input_dim, use_bias = use_bias, la = la, ...)
  ) else return(
    layers$TibLinearLassoMC(input_dim = input_dim, num_outputs = units, use_bias = use_bias, la = la, ...)
  )
}

tp_layer = function(a, b, pen=NULL, name=NULL) {
  x <- tf_row_tensor(a, b) %>% 
    layer_dense(units = 1, activation = "linear", 
                name = name, use_bias = FALSE, 
                kernel_regularizer = pen)
  return(x)
}

ttp_layer = function(a, b, c, pen=NULL, name = NULL) {
  x <- tf_row_tensor(tf_row_tensor(a, b), c) %>% 
    layer_dense(units = 1, activation = "linear", 
                name = name, use_bias = FALSE, 
                kernel_regularizer = pen)
  return(x)
}

tf_stride_cols <- function(A, start, end=NULL)
{
  
  if(is.null(end)) end <- start
  return(
    #tf$strided_slice(A, c(0L,as.integer(start-1)), c(tf$shape(A)[1], as.integer(end)))
    tf$keras$layers$Lambda(function(x) x[,as.integer(start):as.integer(end)])(A)
    )
  

}
    
vc_block <- function(ncolNum, levFac, penalty = NULL, name = NULL){
  ret_fun <- function(x){ 
    
    a = tf_stride_cols(x, 1, ncolNum)
    b = tf$one_hot(tf$cast(
      (tf_stride_cols(x, ncolNum+1)[,1]), 
      dtype="int32"), 
      depth=levFac)
    return(tp_layer(a, b, pen=penalty, name=name))
    
  }
  return(ret_fun)
}

vvc_block <- function(ncolNum, levFac1, levFac2, penalty = NULL, name = NULL){
  ret_fun <- function(x) ttp_layer(x[,as.integer(1:ncolNum)],
                                  tf$one_hot(tf$cast(x[,as.integer((ncolNum+1))], dtype="int32"), 
                                             depth=levFac1),
                                  tf$one_hot(tf$cast(x[,as.integer((ncolNum+2))], dtype="int32"), 
                                             depth=levFac2), pen=penalty, name=name)
  return(ret_fun)
}

layer_factor <- function(nlev, units = 1, activation = "linear", use_bias = FALSE, name = NULL,
                         kernel_regularizer = NULL)
{
  
  ret_fun <- function(x) layer_dense(tf$squeeze(
    tf$one_hot(tf$cast(x, dtype="int32"), 
               depth = as.integer(nlev)),
    axis=1L
  ),
  units = units,
  activation = activation,
  use_bias = use_bias,
  name = name,
  kernel_regularizer = kernel_regularizer)
  return(ret_fun)
  
}

layer_random_effect <- function(freq, df)
{
  
  df_fun <- function(lam) sum((freq^2 + 2*freq*lam)/(freq+lam)^2)
  lambda = uniroot(function(x){df_fun(x)-df}, interval = c(0,1e15))$root
  nlev = length(freq)
  return(
    layer_factor(nlev = nlev, units = 1, activation = "linear", use_bias = FALSE, name = NULL,
                 kernel_regularizer = regularizer_l2(l = lambda/sum(freq)))
  )
  
}