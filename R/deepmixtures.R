#' @param mixture_dist integer either 0 or >= 2. If 0 (default),
#' no mixture distribution is fitted. If >= 2, a network is constructed that outputs
#' a multivariate response for each of the mixture components.
deepmixtures <- function(
  mixture_dist = 0
)
{
  
  if(mixture_dist){
    list_pred <- layer_lambda(preds,
                              f = function(x)
                              {
                                tf$split(x, num_or_size_splits =
                                           c(1L, as.integer(nr_params-1)),
                                         axis = 1L)
                              })
    list_pred[[1]] <- list_pred[[1]] %>%
      dense_layer(units = as.integer(mixture_dist),
                  activation = "softmax",
                  use_bias = FALSE)
    preds <- layer_concatenate(list_pred)
  }
  
  
}