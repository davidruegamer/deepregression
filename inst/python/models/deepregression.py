#' @title Initializing Deep Distributional Regression Models
#'
#'
#' @param n_obs number of observations
#' @param ncol_structured a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a structured part
#' the corresponding entry must be zero.
#' @param ncol_deep a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a deep part
#' the corresponding entry must be zero. If all parameters
#' are estimated by the same deep model, the first entry must be
#' non-zero while the others must be zero.
#' @param list_structured list of (non-linear) structured layers
#' (list length between 0 and number of parameters)
#' @param list_deep list of deep models to be used
#' (list length between 0 and number of parameters)
#' @param use_bias_in_structured logical, whether or not to use a bias in
#' structured layer
#' @param nr_params number of distribution parameters
#' @param train_together see \code{?deepregression}
#' @param lambda_lasso penalty parameter for l1 penalty of structured layers
#' @param lambda_ridge penalty parameter for l2 penalty of structured layers
#' @param family family specifying the distribution that is modelled
#' @param dist_fun a custom distribution applied to the last layer,
#' see \code{\link{make_tfd_dist}} for more details on how to construct
#' a custom distribution function.
#' @param variational logical value specifying whether or not to use
#' variational inference. If \code{TRUE}, details must be passed to
#' the via the ellipsis to the initialization function
#' @param weights observation weights used in the likelihood
#' @param learning_rate learning rate for optimizer
#' @param optimizer optimizer used (defaults to adam)
#' @param fsbatch_optimizer logical; see \code{?deepregression}
#' @param monitor_metric see \code{?deepregression}
#' @param posterior function defining the posterior
#' @param prior function defining the prior
#' @param orthog_fun function defining the orthogonalization
#' @param orthogX vector of columns defining the orthgonalization layer
#' @param kl_weight KL weights for variational networks
#' @param output_dim dimension of the output (> 1 for multivariate outcomes)
#' @param mixture_dist see \code{?deepregression}
#' @param split_fun see \code{?deepregression}
#' @param ind_fun see \code{?deepregression}
#' @param extend_output_dim see \code{?deepregression}
#' @param offset list of logicals corresponding to the paramters;
#' defines per parameter if an offset should be added to the predictor
#' @param additional_penalty to specify any additional penalty, provide a function
#' that takes the \code{model$trainable_weights} as input and applies the
#' additional penalty. In order to get the correct index for the trainable
#' weights, you can run the model once and check its structure.
#' @param constraint_fun function; a constraint for the linear layers
#' @param compile_model logical; whether to compile the model (default is TRUE)
#'
#' @export
'''
deepregression_init <- function(
  n_obs,
  ncol_structured,
  ncol_deep,
  list_structured,
  list_deep,
  use_bias_in_structured = FALSE,
  nr_params = 2,
  train_together = NULL,
  lambda_lasso=NULL,
  lambda_ridge=NULL,
  family,
  dist_fun = NULL,
  variational = TRUE,
  weights = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  # fsbatch_optimizer = FALSE,
  monitor_metric = list(),
  posterior = posterior_mean_field,
  prior = prior_trainable,
  orthog_fun = orthog,
  orthogX = NULL,
  kl_weight = 1 / n_obs,
  output_dim = 1,
  mixture_dist = FALSE,
  split_fun = split_model,
  ind_fun = function(x) x,
  extend_output_dim = 0,
  offset = NULL,
  additional_penalty = NULL,
  constraint_fun = NULL,
  compile_model = TRUE
)
{

  # check injection
  # if(length(inject_after_layer) > nr_params)
  #   stop("Can't have more injections than parameters.")
  # if(any(sapply(inject_after_layer, function(x) x%%1!=0)))
  #   stop("inject_after_layer must be a positive / negative integer")

  if(variational){
    dense_layer <- function(x, ...)
      layer_dense_variational(x,
                              make_posterior_fn = posterior,
                              make_prior_fn = prior,
                              kl_weight = kl_weight,
                              ...
      )
  }else{
    dense_layer <- function(x, ...)
      layer_dense(x, ...)
  }


  # define the input layers
  inputs_deep <- lapply(ncol_deep, function(param_list){
    if(is.list(param_list) & length(param_list)==0) return(NULL)
    lapply(param_list, function(nc){
      if(sum(unlist(nc))==0) return(NULL) else{
        if(is.list(nc) & length(nc)>1){
          layer_input(shape = list(as.integer(sum(unlist(nc)))))
        }else if(is.list(nc) & length(nc)==1){
          layer_input(shape = as.list(as.integer(nc[[1]])))
        }else stop("Not implemented yet.")
      }
    })
  })
  inputs_struct <- lapply(1:length(ncol_structured), function(i){
    nc = ncol_structured[i]
    if(nc==0) return(NULL) else
      # if(!is.null(list_structured[[i]]) & nc > 1)
      # nc>1 will cause problems when implementing ridge/lasso
      layer_input(shape = list(as.integer(nc)))
  })

  if(!is.null(orthogX)){
    ox <- lapply(1:length(orthogX), function(i){

      x = orthogX[[i]]
      if(is.null(x) | is.null(inputs_deep[[i]])) return(NULL) else{
        lapply(x, function(xx){
          if(is.null(xx) || xx==0) return(NULL) else
            return(layer_input(shape = list(as.integer(xx))))})
      }
    })
  }else{
    ox <- NULL
  }

  if(!is.null(offset)){

    offset_inputs <- lapply(offset, function(odim){
      if(is.null(odim) | odim==0) return(NULL) else{
        return(
          layer_input(shape = list(odim))
        )
      }
    })

    ones_initializer = tf$keras.initializers$Ones()

    offset_layers <- lapply(offset_inputs, function(x){
      if(is.null(x)) return(NULL) else
        return(
          x %>%
            layer_dense(units = 1,
                        activation = "linear",
                        use_bias = FALSE ,
                        trainable = FALSE,
                        kernel_initializer = ones_initializer))
    })


  }

  # extend one or more layers' output dimension
  if(length(extend_output_dim) > 1 || extend_output_dim!=0){
    output_dim <- output_dim + extend_output_dim
  }else{
    output_dim <- rep(output_dim, length(inputs_struct))
  }

  if(!is.null(lambda_ridge) && !is.list(lambda_ridge))
    lambda_ridge <- as.list(rep(lambda_ridge, length(inputs_struct)))
  if(!is.null(lambda_lasso) && !is.list(lambda_lasso))
    lambda_lasso <- as.list(rep(lambda_lasso, length(inputs_struct)))
  if(!is.list(constraint_fun))
    constraint_fun <- list(constraint_fun)[rep(1, length(inputs_struct))]

  # define structured predictor
  structured_parts <- lapply(1:length(inputs_struct),
                             function(i){
                               if(is.null(inputs_struct[[i]]))
                               {
                                 return(NULL)
                               }else{
                                 if(is.null(list_structured[[i]]))
                                 {
                                   if(!is.null(lambda_lasso[[i]]) &
                                      is.null(lambda_ridge[[i]])){
                                     # l1 = tf$keras$regularizers$l1(l=lambda_lasso[[i]])
                                     lasso_layer <- tib_layer(
                                       input_dim = ncol_structured[i],
                                       use_bias = use_bias_in_structured,
                                       la = lambda_lasso[[i]],
                                       name = paste0("tib_lasso_", i)
                                     )
                                     return(inputs_struct[[i]] %>% lasso_layer)
                                   }else if(!is.null(lambda_ridge[[i]]) &
                                            is.null(lambda_lasso[[i]])){
                                     l2 = tf$keras$regularizers$l2(l=lambda_ridge[[i]])
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = as.integer(output_dim[i]),
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                kernel_regularizer = l2,
                                                kernel_constraint = constraint_fun[[i]],
                                                name = paste0("structured_ridge_",
                                                              i))
                                     )
                                   }else if(!is.null(lambda_ridge[[i]]) &
                                            !is.null(lambda_lasso[[i]])){
                                     l12 = tf$keras$regularizers$l1_l2(l1=lambda_lasso[[i]],
                                                                       l2=lambda_ridge[[i]])
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = as.integer(output_dim[i]),
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                kernel_regularizer = l12,
                                                kernel_constraint = constraint_fun[[i]],
                                                name = paste0("structured_elastnet_",
                                                              i))
                                     )
                                   }else{
                                     return(inputs_struct[[i]] %>%
                                              dense_layer(
                                                units = as.integer(output_dim[i]),
                                                activation = "linear",
                                                use_bias = use_bias_in_structured,
                                                kernel_constraint = constraint_fun[[i]],
                                                name = paste0("structured_linear_",
                                                              i))
                                     )
                                   }
                                 }else{
                                   this_layer <- list_structured[[i]]
                                   return(inputs_struct[[i]] %>% this_layer)
                                 }
                               }
                             })


  # split deep parts in two parts, where
  # the first part is used in the orthogonalization
  # and the second is put back on top of the first
  # after orthogonalization

  # if(!train_together &
  #    (length(inputs_deep[!sapply(inputs_deep,is.null)]) !=
  #     length(list_deep[!sapply(list_deep,is.null)])) &
  #    any(!sapply(inputs_deep, is.null)) & length(ncol_deep)>1)
  #   stop(paste0("If paramters of distribution are not trained together, ",
  #        "a deep model must be provided for each parameter."))
  deep_split <- lapply(ncol_deep[1:nr_params], function(param_list){
    lapply(names(param_list), function(nn){
      if(is.null(nn)) return(NULL) else if(grepl("^vc_.*",nn) | grepl("^fctly_.*",nn)) return(
        list(list_deep[[nn]],
             function(x) x)) else
               split_fun(list_deep[[nn]])
    })
  })

  if(!is.null(train_together) && !is.null(list_deep) &
     !(length(list_deep)==1 & is.null(list_deep[[1]])))
    list_deep_shared <- list_deep[sapply(names(list_deep),function(nnn)
      !nnn%in%names(ncol_deep[1:nr_params]))] else
        list_deep_shared <- NULL

  list_deep <- lapply(deep_split, function(param_list)
    lapply(param_list, "[[", 1))
  list_deep_ontop <- lapply(deep_split, function(param_list)
    lapply(param_list, "[[", 2))

  # define deep predictor
  deep_parts <- lapply(1:length(list_deep), function(i)
    if(is.null(inputs_deep[[i]]) | length(inputs_deep[[i]])==0)
      return(NULL) else
        lapply(1:length(list_deep[[i]]), function(j) return(
          list_deep[[i]][[j]](inputs_deep[[i]][[j]]))
          ))

  ############################################################
  ################# Apply Orthogonalization ##################

  # create final linear predictor per distribution parameter
  # -> depending on the presence of a deep or structured part
  # the corresponding part is returned. If both are present
  # the deep part is projected into the orthogonal space of the
  # structured part

  if(!is.null(train_together) && !is.null(list_deep_shared) &
     any(!sapply(inputs_deep, is.null))){

    shared_parts <- lapply(unique(unlist(train_together)), function(i)
      list_deep_shared[[i]](
        inputs_deep[[nr_params + i]][[1]]
      ))

    colind_shared <-
      apply(sapply(1:length(shared_parts),function(j)
        sapply(train_together, function(tt) if(length(tt)==0) 0 else tt == j)),
        2, cumsum)

  }else{

    shared_parts <- NULL

  }

  list_pred_param <- lapply(1:nr_params, function(i){

    if(!is.null(shared_parts)){

      shared_i <- if(length(train_together[[i]])==0) NULL else
        shared_parts[[train_together[[i]]]][
          ,
          colind_shared[,train_together[[i]]][i],
          drop=FALSE]
    }else{
      shared_i <- NULL
    }

    combine_model_parts(deep = deep_parts[[i]],
                        deep_top = list_deep_ontop[[i]],
                        struct = structured_parts[[i]],
                        ox = ox[[i]],
                        orthog_fun = orthog_fun,
                        shared = shared_i)
  }
  )

  if(!is.null(offset)){

    for(i in 1:length(list_pred_param)){

      if(!is.null(offset[[i]]) & offset[[i]]!=0)
        list_pred_param[[i]] <- layer_add(list(list_pred_param[[i]],
                                               offset_layers[[i]]))

    }

  }

  # concatenate predictors
  # -> just to split them later again?
  if(length(list_pred_param) > 1)
    preds <- layer_concatenate(list_pred_param) else
      preds <- list_pred_param[[1]]

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

  ############################################################
  ### Define Distribution Layer and Variational Inference ####



  # define the distribution function applied in the last layer
  
  # special families needing transformations
  
  if(family %in% c("betar", "gammar", "pareto_ls", "inverse_gamma_ls")){
    
    # trafo_list <- family_trafo_funs(family)
    # predsTrafo <- layer_lambda(object = preds, f = trafo_fun)
    # preds <- layer_concatenate(predsTrafo)
    
    dist_fun <- family_trafo_funs_special(family)
    
  }
  
  # apply the transformation for each parameter
  # and put in the right place of the distribution
  if(is.null(dist_fun))
    dist_fun <- make_tfd_dist(family)
  
  # make model variational and output distribution
  # if(variational){
  #
  #   out <- preds %>%
  #     layer_dense_variational(
  #       units = length(nr_params),
  #       make_posterior_fn = posterior,
  #       make_prior_fn = prior,
  #       kl_weight = kl_weight
  #     ) %>%
  #     layer_distribution_lambda(dist_fun)
  #
  # }else{
  
  out <- preds %>%
    tfprobability::layer_distribution_lambda(dist_fun)
  
  # }
  
  ############################################################
  ################# Define and Compile Model #################
  
  # define all inputs
  inputList <- unname(c(
    unlist(inputs_deep[!sapply(inputs_deep, is.null)],
           recursive = F),
    inputs_struct[!sapply(inputs_struct, is.null)],
    unlist(ox[!sapply(ox, is.null)]))
  )
  
  if(!is.null(offset)){
    
    inputList <- c(inputList,
                   unlist(offset_inputs[!sapply(offset_inputs, is.null)]))

  }

  # the final model is defined by its inputs
  # and outputs

  model <- keras_model(inputs = inputList,
                       outputs = out)

  # define weights to be equal to 1 if not given
  if(is.null(weights)) weights <- 1

  # the negative log-likelihood is given by the negative weighted
  # log probability of the model
  if(family!="pareto_ls"){  
    negloglik <- function(y, model)
      - weights * (model %>% ind_fun() %>% tfd_log_prob(y)) 
  }else{
    negloglik <- function(y, model)
      - weights * (model %>% ind_fun() %>% tfd_log_prob(y + model$scale))
  }
        
  
  if(!is.null(additional_penalty)){

    add_loss <- function(x) additional_penalty(
      model$trainable_weights
    )
    model$add_loss(add_loss)

  }

  # compile the model using the defined optimizer,
  # the negative log-likelihood as loss funciton
  # and the defined monitoring metrics as metrics
  if(compile_model){
    model %>% compile(optimizer = optimizer,
                      loss = negloglik,
                      metrics = monitor_metric)
  
  return(model)
    
  }else{
    
    return(list(model = model,
                loss = negloglik,
                monitor_metric = monitor_metric))
    
  }

}
'''
