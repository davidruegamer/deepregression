#' @title Fitting Semi-Structured Deep Distributional Regression
#'
#' @param y response variable
#' @param list_of_formulae a named list of right hand side formulae,
#' one for each parameter of the distribution specified in \code{family};
#' set to \code{~ 1} if the parameter should be treated as constant.
#' Use the \code{s()}-notation from \code{mgcv} for specification of
#' non-linear structured effects and \code{d(...)} for
#' deep learning predictors (predictors in brackets are separated by commas),
#' where \code{d} can be replaced by an name name of the names in
#' \code{list_of_deep_models}, e.g., \code{~ 1 + s(x) + my_deep_mod(a,b,c)},
#' where my_deep_mod is the name of the neural net specified in
#' \code{list_of_deep_models} and \code{a,b,c} are features modeled via
#' this network.
#' @param list_of_deep_models a named list of functions
#' specifying a keras model.
#' See the examples for more details.
#' @param family a character specifying the distribution. For information on
#' possible distribution and parameters, see \code{\link{make_tfd_dist}}
#' @param train_together a list of formulae of the same length as \code{list_of_formulae}
#' specifying the deep predictors that should be trained together and then the results are;
#' fed into different distribution parameters; use the same name for the deep predictor to
#' indicate for which distribution parameter they should be used. For example, if the second
#' and fourth list entry are \code{~ lstm_mod(text)} then the jointly learned \code{lstm_mod}
#' network is added to the linear predictor of the second and fourth distribution parameter.
#' Those network names should then be excluded from the \code{list_of_formulae}
#' @param data data.frame or named list with input features
#' @param image_var named list; names correspond to image variables, values for each list item
#' corresponds to the input size of the respective image, e.g., \code{list(image = list(c(200,200,3)))}.
#' @param dim_deep list for each distribution parameter with NULL or integer vector entries; 
#' this is an optional argument to manually specify the input dimensions for the unstructured 
#' model part(s) and required if placeholders are used for unstructured data sources. E.g.,
#' if the formula contains \code{nn(images)} where \code{images} are ids of images that
#' are loaded using a generator, the size of the image is part of the respective
#' @param df degrees of freedom for all non-linear structural terms;
#' either one common value or a list of the same length as number of parameters and
#' each list item a list of the same length as number of smooth terms in the
#' respective formula where elements can be vectors for two- or multidimensional smooth terms
#' @param lambda_lasso scalar value or list of \code{length(list_of_formulae)};
#' smoothing parameter for lasso regression; can be combined with ridge
#' @param lambda_ridge scalar value or list of \code{length(list_of_formulae)};
#' smoothing parameter for ridge regression; can be combined with lasso
#' @param defaultSmoothing function applied to all s-terms, per default (NULL)
#' the minimum df of all possible terms is used.
#' @param cv_folds a list of lists, each list element has two elements, one for
#' training indices and one for testing indices;
#' if a single integer number is given,
#' a simple k-fold cross-validation is defined, where k is the supplied number.
#' @param validation_data data for validation during training.
#' @param validation_split percentage of training data used for validation.
#' Per default 0.2.
#' @param dist_fun a custom distribution applied to the last layer,
#' see \code{\link{make_tfd_dist}} for more details on how to construct
#' a custom distribution function.
#' @param learning_rate learning rate for optimizer
#' @param optimizer optimzer used. Per default ADAM.
#' @param fsbatch_optimizer logical; use a special optimizer conducting a mini-batch variant of the 
#' Fellner-Schall algorithm.
#' @param fsbatch_options call to \code{fsbatch_control} with list of options for fsbatch_optimizer.
#' See \code{?fsbatch_control} for details.
#' @param variational logical value specifying whether or not to use
#' variational inference. If \code{TRUE}, details must be passed to
#' the via the ellipsis to the initialization function
#' (see \code{\link{deepregression_init}})
#' @param monitor_metric Further metrics to monitor
#' @param posterior_fun function defining the posterior function for the variational
#' verison of the network
#' @param prior_fun function defining the prior function for the variational
#' verison of the network
#' @param seed integer value used as a seed in data splitting
#' @param tf_seed a seed for tensorflow (only works with R version >= 2.2.0)
#' @param mixture_dist integer either 0 or >= 2. If 0 (default),
#' no mixture distribution is fitted. If >= 2, a network is constructed that outputs
#' a multivariate response for each of the mixture components.
#' @param split_fun a function separating the deep neural network in two parts
#' so that the orthogonalization can be applied to the first part before
#' applying the second network part; per default, the function \code{split_model} is
#' used which assumes a dense layer as penultimate layer and separates the network
#' into a first part without this last layer and a second part only consisting of a
#' single dense layer that is fed into the output layer
#' @param null_space_penalty logical value;
#' if TRUE, the null space will also be penalized for smooth effects.
#' Per default, this is equal to the value give in \code{variational}.
#' @param ind_fun function applied to the model output before calculating the
#' log-likelihood. Per default independence is assumed by applying \code{tfd_independent}.
#' @param extend_output_dim integer value >= 0 for extending the output dimension by an
#' additive constant. If set to a value > 0, a multivariate response with dimension
#' \code{1 + extend_output_dim} is defined.
#' @param offset a list of column vectors (i.e. matrix with ncol = 1) or NULLs for each
#' parameter, in case an offset should be added to the additive predictor;
#' if NULL, no offset is used
#' @param offset_val a list analogous to offset for the validation data
#' @param absorb_cons logical; adds identifiability constraint to the basisi.
#' See \code{?mgcv::smoothCon} for more details.
#' @param anisotropic whether or not use anisotropic smoothing (default is TRUE)
#' @param zero_constraint_for_smooths logical; the same as absorb_cons,
#' but done explicitly. If true a constraint is put on each smooth to have zero mean. Can
#' be a vector of \code{length(list_of_formulae)} for each distribution parameter.
#' @param orthog_type one of two options; If \code{"manual"}, 
#' the QR decomposition is calculated before model fitting, 
#' otherwise (\code{"tf"}) a QR is calculated in each batch iteration via TF.
#' The first only works well for larger batch sizes or ideally batch_size == NROW(y).
#' @param orthogonalize logical; if set to \code{FALSE}, orthogonalization is deactivated
#' @param hat1 logical; if TRUE, the smoothing parameter is defined by the trace of the hat
#' matrix sum(diag(H)), else sum(diag(2*H-HH))
#' @param sp_scale positive constant; for scaling the DRO calculated penalty (1 per default)
#' @param order_bsp NULL or integer; order of Bernstein polynomials; if not NULL,
#' a conditional transformation model (CTM) is fitted.
#' @param y_basis_fun,y_basis_fun_prime basis functions for y transformation for CTM case
#' @param split_between_shift_and_theta if \code{family == 'transformation_model'} and
#' \code{!is.null(train_together)}, \code{split_between_shift_and_theta} is supposed
#' to define how many of the last layer's hidden units are used for the shift
#' term and how many for the theta term (an integer vector of length 2).
#' @param addconst_interaction positive constant;
#' a constant added to the additive predictor of the interaction term.
#' If \code{NULL}, terms are left unchanged. If 0 and predictors have negative values in their
#' design matrix, the minimum value of all predictors is added to ensure positivity.
#' If > 0, the minimum value plus the \code{addconst_interaction} is added to each predictor
#' in the interaction term.
#' @param additional_penalty a penalty that is added to the negative log-likelihood; must be
#' a function of the keras model's \code{trainable_weights} (including necessary subsetting) which
#' is always called \code{model}
#' @param penalty_summary keras function; summary function for the penalty in the spline layer;
#' default is \code{k_sum}. Another option could be \code{k_mean}.
#' @param convertfun function to convert objects into a matrix or tensor format.
#' @param compile_model logical; whether to compile the model or not
#' @param base_distribution tfd_distribution or string; the base distribution for 
#' transformation models, see \code{?deeptransformation\_init}.
#' @param ... further arguments passed to the \code{deepregression\_init} function
#'
#' @import tensorflow tfprobability keras mgcv dplyr R6 reticulate Matrix
#'
#' @importFrom keras fit compile
#' @importFrom Metrics auc
#' @importFrom tfruns is_run_active view_run_metrics update_run_metrics write_run_metadata
#' @importFrom graphics abline filled.contour matplot par points
#' @importFrom stats as.formula model.matrix terms terms.formula uniroot var dbeta coef predict
#' @importFrom methods slotNames is as
#'
#' @export
#'
#' @examples
#' library(deepregression)
#'
#' n <- 1000
#' data = data.frame(matrix(rnorm(4*n), c(n,4)))
#' colnames(data) <- c("x1","x2","x3","xa")
#' formula <- ~ 1 + deep_model(x1,x2,x3) + s(xa) + x1
#'
#' deep_model <- function(x) x %>%
#' layer_dense(units = 32, activation = "relu", use_bias = FALSE) %>%
#' layer_dropout(rate = 0.2) %>%
#' layer_dense(units = 8, activation = "relu") %>%
#' layer_dense(units = 1, activation = "linear")
#'
#' y <- rnorm(n) + data$xa^2 + data$x1
#'
#' mod <- deepregression(
#'   list_of_formulae = list(loc = formula, scale = ~ 1),
#'   data = data, validation_data = list(data, y), y = y,
#'   list_of_deep_models = list(deep_model = deep_model), 
#'   df = 6, 
#'   tf_seed = 1
#' )
#'
#' # train for more than 10 epochs to get a better model
#' mod %>% fit(epochs = 10, early_stopping = TRUE)
#' mod %>% plot()
#' mod %>% coef()
#'
deepregression <- function(
  y,
  list_of_formulae,
  list_of_deep_models,
  family = c(
    "normal", "bernoulli", "bernoulli_prob", "beta", "betar",
    "cauchy", "chi2", "chi","exponential", "gamma_gamma",
    "gamma", "gammar", "gumbel", "half_cauchy", "half_normal", "horseshoe",
    "inverse_gamma", "inverse_gamma_ls", "inverse_gaussian", 
    "laplace", "log_normal", "logistic",
    "multinomial", "multinoulli", "negbinom", "negbinom_ls", 
    "pareto", "pareto_ls", "poisson",
    "poisson_lograte", "student_t",
    "student_t_ls", "truncated_normal", "uniform", "zinb", "zip",
    "transformation_model"
  ),
  train_together = list(),
  data,
  image_var = list(),
  dim_deep = NULL,
  # batch_size = NULL,
  # epochs = 10L,
  df = NULL,
  lambda_lasso = NULL,
  lambda_ridge = NULL,
  convert_factors = FALSE,
  # defaultSp = 1,
  # defaultSmoothing = function(smoothTerm){
  #   smoothTerm$sp = defaultSp
  #   return(smoothTerm)
  # },
  defaultSmoothing = NULL,
  cv_folds = NULL,
  validation_data = NULL,
  validation_split = ifelse(is.null(validation_data) & is.null(cv_folds), 0.2, 0),
  dist_fun = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  fsbatch_optimizer = FALSE,
  fsbatch_options = fsbatch_control(),
  variational = FALSE,
  monitor_metric = list(),
  seed = 1991-5-4,
  tf_seed = NULL,
  mixture_dist = 0,
  split_fun = split_model,
  posterior_fun = posterior_mean_field,
  prior_fun = prior_trainable,
  null_space_penalty = variational,
  ind_fun = function(x) tfd_independent(x),
  extend_output_dim = 0,
  offset = NULL,
  offset_val = NULL,
  absorb_cons = FALSE,
  anisotropic = TRUE,
  zero_constraint_for_smooths = TRUE,
  orthog_type = c("tf", "manual"),
  orthogonalize = TRUE,
  hat1 = FALSE,
  sp_scale = NROW(y),
  order_bsp = NULL,
  y_basis_fun = function(y) eval_bsp(y, order = order_bsp, supp = range(y)),
  y_basis_fun_prime = function(y) eval_bsp_prime(y, order = order_bsp,
                                                 supp = range(y)) / diff(range(y)),
  split_between_shift_and_theta = NULL,
  addconst_interaction = NULL,
  additional_penalty = NULL,
  penalty_summary = k_sum,
  convertfun = as.matrix,
  compile_model = TRUE,
  base_distribution = "normal",
  # compress = TRUE,
  ...
)
{

  if(!is.null(tf_seed)) 
    try(tensorflow::set_random_seed(tf_seed), silent = TRUE)

  # first check if an env is available
  if(!reticulate::py_available())
  {
    message("No Python Environemt available. Use check_and_install() ",
            "to install recommended environment.")
    invisible(return(NULL))
  }

  if(!py_module_available("tensorflow"))
  {
    message("Tensorflow not available. Use install_tensorflow() ",
            "or check_and_install() to update you system.")
    invisible(return(NULL))
  } # nocov end

  # check family
  family <- match.arg(family)
  # convert data.frame to list
  if(is.data.frame(data)){
    # if(compress){
    # data_repr <- data
    data <- as.list(
      # compress_data(
      data
      # )
    )
  }
  # }else{
  # data_repr <- data
  # }else{
  # warning("Data compression currently not available for list inputs.")
  # data_repr <- data
  # }
  # if(any(sapply(data, is.data.frame)))
  #   stop("Data.frames within the input list are now allowed.")
  # get column names of data
  varnames <- names(data)
  if(is.null(varnames) | any(varnames==""))
    stop("If data is a list, names must be given.")
  # for convenience transform NULL to list(NULL) for list_of_deep_models
  if(missing(list_of_deep_models) | is.null(list_of_deep_models)){
    list_of_deep_models <- list(NULL)
    if(length(train_together)==0) warning("No deep model specified")
  }else if(!is.list(list_of_deep_models)) stop("list_of_deep_models must be a list.")

  if(length(train_together)>0 & (!is.list(train_together) |
                                 length(train_together) != length(list_of_formulae)))
    stop("If specified, train_together must be of same length as list_of_formulae.")
  # get names of networks
  netnames <- names(list_of_deep_models)
  if("vc" %in% netnames | "vvc" %in% netnames)
    stop("Please rename networks in the list of deep models with name 'vc' or 'vvc'.")
  if(is.null(netnames) & length(list_of_deep_models) > 0)
    netnames <- "d"
  if(!is.null(list_of_deep_models) && is.null(names(list_of_deep_models)))
    names(list_of_deep_models) <- rep("d", length(list_of_deep_models))
  # number of observations
  n_obs <- NROW(y)
  # number of output dim
  output_dim <- NCOL(y)
  # check consistency of #parameters
  nr_params <- length(list_of_formulae)
  # add fake parameter for train_together models
  # if(length(train_together)>0)
  #   nr_params <- nr_params + length(unique(list_of_formulae))
  if(is.null(dist_fun) & family != "transformation_model")
    nrparams_dist <- make_tfd_dist(family, return_nrparams = TRUE) else
      nrparams_dist <- nr_params # - length(unique(list_of_formulae))
  if(nrparams_dist < nr_params) # - length(unique(list_of_formulae)))
  {
    warning("More formulae specified than parameters available.",
            " Will only use ", nrparams_dist, " formula(e).")
    nr_params <- nrparams_dist
    list_of_formulae <- list_of_formulae[1:nrparams_dist]
  }
  # check for lists in list
  if(is.list(data)){
    if(any(sapply(data, class)=="list"))
      stop("Cannot deal with lists in list. Please remove list items in your data input.")
  }
  # check list of formulae is always one-sided
  if(any(sapply(list_of_formulae, function(x)
    attr( terms(x, data = data[!sapply(data, is.null)]) , "response" ) != 0 ))){
    stop("Only one-sided formulas are allowed in list_of_formulae.")
  }
  # check orthog type
  orthog_type <- match.arg(orthog_type)
  # check monitor metric for auc
  if(is.list(monitor_metric) && "auc" %in% monitor_metric)
    if(length(monitor_metric)==1)
      monitor_metric <- auc_metric else
        warning("If auc is chosen as metric, it must be the only specified metric.")
  if(!is.null(offset) & !is.list(offset))
    stop("Argument offset must be a list of offsets for each parameter.")
  # add train together networks in case there are any
  if(length(train_together)>0){

    warning("train_together not well tested in combination with orthogonalization.")
    nulls <- sapply(train_together,is.null)
    train_together[!nulls] <-
      lapply(train_together[!nulls], remove_intercept)
    list_of_formulae <- c(list_of_formulae, unique(train_together[!nulls]))

  }
  
  if(is.null(names(list_of_formulae))){
   names(list_of_formulae) <- names_families(family)
  }
  if(!is.null(df) && !is.list(df)) df <- list(df)[rep(1,length(list_of_formulae))]

  if(length(image_var)>0 & (
    !is.list(image_var) | is.null(names(image_var))))
    stop("image_var must be a named list with lists as elements.")
  
  # if(convert_factors)
  #   data <- fac_to_int_representation(data)
  
  cat("Preparing additive formula(e)...")
  # parse formulae
  parsed_formulae_contents <- lapply(1:length(list_of_formulae),
                                     function(i)
                                       get_contents(
                                         lf = list_of_formulae[[i]],
                                         data = data,
                                         df = df[[i]],
                                         anisotropic = anisotropic,
                                         variable_names = varnames,
                                         network_names = netnames,
                                         defaultSmoothing = defaultSmoothing,
                                         absorb_cons = absorb_cons,
                                         null_space_penalty = null_space_penalty,
                                         hat1 = hat1,
                                         sp_scale = sp_scale,
                                         image_var = image_var,
                                         nr_param = i
                                       )
  )
  cat(" Done.\n")
  
  # extract vc layers
  for(i in 1:length(parsed_formulae_contents)){
    has_layer <- sapply(parsed_formulae_contents[[i]]$deepterms, function(x) !is.null(attr(x, "layer")))
    if(any(has_layer)){
      list_of_deep_models <- c(list_of_deep_models, 
                               lapply(parsed_formulae_contents[[i]]$deepterms[has_layer], function(x)attr(x,"layer")))
    }
  }

  # check for zero ncol linterms
  for(i in 1:length(parsed_formulae_contents)){
    if(NCOL(parsed_formulae_contents[[i]]$linterms)==0)
      parsed_formulae_contents[[i]]["linterms"] <- list(NULL)
  }

  if(length(zero_constraint_for_smooths)==1)
    zero_constraint_for_smooths <- rep(zero_constraint_for_smooths, 
                                       length(parsed_formulae_contents))
  parsed_formulae_contents <- lapply(1:length(parsed_formulae_contents), 
                                     function(i) orthog_smooth(parsed_formulae_contents[[i]],
                                                               zero_cons = zero_constraint_for_smooths[i])
                                     )

  for(i in 1:length(parsed_formulae_contents))
    attr(parsed_formulae_contents[[i]],"zero_cons") <- zero_constraint_for_smooths[i]


  if(family=="transformation_model" & !is.null(addconst_interaction)){
    # ensure positivity of interaction
    parsed_formulae_contents[[2]] <- correct_min_val(parsed_formulae_contents[[2]],
                                                     addconst_interaction)
    if(is.null(attr(parsed_formulae_contents[[2]], "minval")))
      addconst_interaction <- NULL else
        addconst_interaction <- attr(parsed_formulae_contents[[2]], "minval")
    # if(minval<0) y <- y - minval
  }

  # get columns per term
  ncol_deep <- lapply(lapply(
    parsed_formulae_contents, "[[", "deepterms"), function(x){
      ret <- #if(is.data.frame(x[[1]]) & length(x)==1) list(NCOL(x[[1]])) else
        lapply(x, nCOL)
      names(ret) <- names(x)
      return(ret)
    })

  ncol_structured <- sapply(
    parsed_formulae_contents[!sapply(parsed_formulae_contents,is.null)],
    function(x){
      ncolsmooth <- 0
      if(!is.null(x[['smoothterms']]))
        ncolsmooth <- sum_cols_smooth(x[['smoothterms']])
      return(ncol_lint(x[['linterms']]) + ncolsmooth)

    })
  # create structured layers
  list_structured <- lapply(1:length(parsed_formulae_contents), function(i)
    get_layers_from_s(parsed_formulae_contents[[i]], i,
                      variational = variational,
                      posterior_fun = posterior_fun,
                      output_dim = output_dim,
                      trafo =
                        (family == "transformation_model" &
                           i == 2),
                      k_summary = penalty_summary,
                      fsbatch_optimizer = fsbatch_optimizer,
                      nobs = n_obs
                      # prior_fun = prior_fun
    ))


  cat("Translating data into tensors...")
  input_cov <- make_cov(parsed_formulae_contents, 
                        convertfun = convertfun,
                        olddata = data,
                        orthogonalize = orthogonalize,
                        retcol = FALSE,
                        returnX = (orthog_type!="manual"))
  ox <- attr(input_cov, "ox")
  if(is.null(ox)) ox <- list(NULL)[rep(1, length(parsed_formulae_contents))]
  attr(input_cov, "ox") <- NULL
  cat(" Done.\n")
  # if(orthogonalize)
  #   ox <- lapply(parsed_formulae_contents, make_orthog,
  #                retcol = FALSE,
  #                returnX = (orthog_type=="tf")) else
  #                  ox <- list(NULL)[rep(1,length(parsed_formulae_contents))]
  # input_cov <- unname(c(input_cov,
  #                       unlist(lapply(ox[!sapply(ox,is.null)],
  #                                     function(x_per_param)
  #                                       lapply(x_per_param[!sapply(x_per_param,is.null)],
  #                                                     function(x)
  #                                                       convertfun(x))),
  #                              recursive = F)
  # ))

  if(!is.null(offset)){

    cat("Using an offset.")
    input_cov <- c(input_cov, lapply(offset[!sapply(offset, is.null)],
                                     function(x) convertfun(matrix(x, ncol = 1)))
                   # recursive = FALSE)
                   )

  }

  param_names <- names(list_of_formulae)
  l_names_effects <- lapply(parsed_formulae_contents, get_names)
  ind_structterms <- lapply(parsed_formulae_contents, get_indices)

  if(!is.null(validation_data)){
    if(!is.list(validation_data) && length(validation_data)!=2 |
       is.data.frame(validation_data))
      stop("Validation data must be a list of length two ",
           "with first entry for the data (features) and second entry for response.")
    if(is.data.frame(validation_data[[1]]))
      validation_data[[1]] <- as.list(validation_data[[1]])
    validation_data[[1]] <- prepare_newdata(parsed_formulae_contents,
                                            validation_data[[1]])

    if(!is.null(offset_val)){
      # print("Using an offset.")
      validation_data[[1]] <- c(validation_data[[1]],
                                lapply(offset_val[!sapply(offset_val, is.null)],
                                              function(x) convertfun(matrix(x, ncol = 1)))#,
                                       # recursive = FALSE)
                                )
    }
  }

  if(!is.null(cv_folds))
  {

    validation_split <- NULL
    validation_data <- NULL
    if(!is.list(cv_folds) & is.numeric(cv_folds) & is.null(dim(cv_folds))){

      if(cv_folds <= 0) stop("cv_folds must be a positive integer, but is ",
                             cv_folds, ".")
      cv_folds <- make_cv_list_simple(data_size=NROW(data[[1]]), round(cv_folds),
                                      seed)

    }
  }

  # define orthogonalization function
  orthog_fun <- switch(orthog_type,
                       col = orthog_ncol,
                       tf = orthog_tf,
                       manual = orthog)

  orthogX <- NULL
  if(orthogonalize) orthogX <- nestNCOL(ox)
  
  # check if transformation models

  if(family=="transformation_model"){

    # atm_toplayer <- NULL
    
    if(nr_params==3){
      
      pos_lags <- sum(sapply(parsed_formulae_contents, function(x)
        length(x$deepterms) + !is.null(x$linterms)))
      intercept_form <- "(Intercept)" %in% names(parsed_formulae_contents[[3]]$linterms)
      if(intercept_form)
        input_cov[[pos_lags]] <- input_cov[[pos_lags]][,-1]
      atm_lags <- NCOL(input_cov[[pos_lags]])
      # 
      # # input_cov <- unlist_order_preserving(input_cov)
      if(!is.null(validation_data) & intercept_form){
        validation_data[[1]][[pos_lags]] <-
          validation_data[[1]][[pos_lags]][,-1]
      }
      
      ncol_structured <- ncol_structured[1:2]

      # atm_toplayer <- list_structured[[3]]
      # list_structured <- list_structured[1:2]

    }
    
    input_cov <- c(input_cov,
                   list(y_basis_fun(y),
                        y_basis_fun_prime(y)))
    
    if(!is.null(validation_data))
      validation_data <- list(unname(
        c(validation_data[[1]], 
          list(y_basis_fun(validation_data[[2]]),
               y_basis_fun_prime(validation_data[[2]])))
      ), validation_data[[2]])

    model <- deeptransformation_init(
      n_obs = n_obs,
      ncol_structured = ncol_structured,
      ncol_deep = ncol_deep,
      support = range(y),
      list_structured = list_structured,
      list_deep = list_of_deep_models,
      orthogX = orthogX,
      lambda_lasso = lambda_lasso,
      lambda_ridge = lambda_ridge,
      monitor_metric = monitor_metric,
      optimizer = optimizer,
      split_fun = split_fun,
      orthog_fun = orthog_fun,
      order_bsp = order_bsp,
      train_together = train_together_ind(train_together),
      split_between_shift_and_theta = split_between_shift_and_theta,
      addconst_interaction = addconst_interaction,
      base_distribution = base_distribution,
      atm_lags = ifelse(nr_params==2,0,atm_lags),
      ...
    )

  }else{
    
    if(fsbatch_optimizer){
      
      model <- dr_init(
        n_obs = n_obs,
        ncol_structured = ncol_structured,
        list_structured = list_structured,
        nr_params = nr_params,
        family = family,
        dist_fun = dist_fun,
        monitor_metric = monitor_metric,
        output_dim = output_dim,
        mixture_dist = mixture_dist,
        ind_fun = ind_fun,
        extend_output_dim = extend_output_dim,
        offset = if(is.null(offset)) NULL else lapply(offset, NCOL0),
        additional_penalty = additional_penalty,
        fsbatch_options = fsbatch_options,
        optimizer = optimizer
      )
      
      
    }else{
     
      
      #############################################################
      # initialize the model
      model <- deepregression_init(
        n_obs = n_obs,
        ncol_structured = ncol_structured,
        ncol_deep = ncol_deep,
        list_structured = list_structured,
        list_deep = list_of_deep_models,
        nr_params = nr_params,
        train_together = train_together_ind(train_together),
        family = family,
        variational = variational,
        dist_fun = dist_fun,
        kl_weight = 1 / n_obs,
        orthogX = orthogX,
        lambda_lasso = lambda_lasso,
        lambda_ridge = lambda_ridge,
        monitor_metric = monitor_metric,
        optimizer = optimizer,
        # fsbatch_optimizer = fsbatch_optimizer,
        # fsbatch_factor = fsbatch_factor,
        output_dim = output_dim,
        mixture_dist = mixture_dist,
        split_fun = split_fun,
        posterior = posterior_fun,
        prior = prior_fun,
        ind_fun = ind_fun,
        extend_output_dim = extend_output_dim,
        offset = if(is.null(offset)) NULL else lapply(offset, NCOL0),
        orthog_fun = orthog_fun,
        additional_penalty = additional_penalty,
        compile_model = compile_model,
        ...
      )
      #############################################################
      
      
    }
  }


  ret <- list(model = model,
              init_params =
                list(
                  input_cov = input_cov,
                  n_obs = n_obs,
                  y = y,
                  offset = offset,
                  validation_split = validation_split,
                  validation_data = validation_data,
                  cv_folds = cv_folds,
                  l_names_effects = l_names_effects,
                  parsed_formulae_contents = parsed_formulae_contents,
                  data = data,
                  ind_structterms = ind_structterms,
                  param_names = param_names,
                  ellipsis = list(...),
                  family = family,
                  orthogonalize = orthogonalize,
                  orthog_type = orthog_type,
                  list_of_formulae = list_of_formulae,
                  image_var = image_var
                ))

  
  this_class <- ifelse(compile_model, 
                       "deepregression",
                       "deepregression_custom"
  )
                  
  
  class(ret) <-  this_class
  if(family=="transformation_model"){
    class(ret) <- c("deeptrafo", this_class)
    ret$init_params <- c(ret$init_params,
                         order_bsp = order_bsp,
                         y_basis_fun = y_basis_fun,
                         y_basis_fun_prime = y_basis_fun_prime,
                         base_distribution = base_distribution
                         )
  }

  return(ret)

}

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
                                       units = as.integer(output_dim[i]),
                                       use_bias = use_bias_in_structured,
                                       la = lambda_lasso[[i]],
                                       name = paste0("tib_lasso_", i)
                                     )
                                     return(tf$add(
                                       lasso_layer(tf_stride_cols(inputs_struct[[i]],2,inputs_struct[[i]]$shape[[2]])),
                                       layer_dense(units=as.integer(output_dim[i]), name = paste0("intercept_",i),
                                                   use_bias=FALSE)(
                                         tf_stride_cols(inputs_struct[[i]],1))
                                       ))
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



#' @title Initializing a Distributional Regression Model
#'
#'
#' @param n_obs number of observations
#' @param ncol_structured a vector of length #parameters
#' defining the number of variables used for each of the parameters.
#' If any of the parameters is not modelled using a structured part
#' the corresponding entry must be zero.
#' @param list_structured list of (non-linear) structured layers
#' (list length between 0 and number of parameters)
#' @param nr_params number of distribution parameters
#' @param family family specifying the distribution that is modelled
#' @param dist_fun a custom distribution applied to the last layer,
#' see \code{\link{make_tfd_dist}} for more details on how to construct
#' a custom distribution function.
#' @param weights observation weights used in the likelihood
#' @param monitor_metric see \code{?deepregression}
#' @param output_dim dimension of the output (> 1 for multivariate outcomes)
#' @param mixture_dist see \code{?deepregression}
#' @param ind_fun see \code{?deepregression}
#' @param extend_output_dim see \code{?deepregression}
#' @param offset list of logicals corresponding to the paramters;
#' defines per parameter if an offset should be added to the predictor
#' @param additional_penalty to specify any additional penalty, provide a function
#' that takes the \code{model$trainable_weights} as input and applies the
#' additional penalty. In order to get the correct index for the trainable
#' weights, you can run the model once and check its structure.
#' @param fsbatch_options options for Fellner-Schall algorithm, see 
#' \code{?deepregression}
#' @param optimizer see \code{?deepregression} 
#'
#' @export
dr_init <- function(
  n_obs,
  ncol_structured,
  list_structured,
  nr_params = 2,
  family,
  dist_fun = NULL,
  weights = NULL,
  monitor_metric = list(),
  output_dim = 1,
  mixture_dist = FALSE,
  ind_fun = function(x) x,
  extend_output_dim = 0,
  offset = NULL,
  additional_penalty = NULL,
  fsbatch_options = fsbatch_control(),
  optimizer = tf$keras$optimizers$SGD()
)
{
  
  inputs_struct <- lapply(1:length(ncol_structured), function(i){
    nc = ncol_structured[i]
    if(nc==0) return(NULL) else
      # if(!is.null(list_structured[[i]]) & nc > 1)
      # nc>1 will cause problems when implementing ridge/lasso
      layer_input(shape = list(as.integer(nc)))
  })
  
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
  
  
  # define structured predictor
  structured_parts <- lapply(1:length(inputs_struct),
                             function(i){
                               if(is.null(inputs_struct[[i]]))
                               {
                                 return(NULL)
                               }else{
                                 if(is.null(list_structured[[i]]))
                                 {
                                   return(inputs_struct[[i]] %>%
                                            layer_dense(
                                              units = as.integer(output_dim[i]),
                                              activation = "linear",
                                              use_bias = FALSE,
                                              name = paste0("structured_linear_",
                                                            i))
                                   )
                                 }else{
                                   this_layer <- list_structured[[i]]
                                   return(inputs_struct[[i]] %>% this_layer)
                                 }
                               }
                             })
  
  
  
  if(!is.null(offset)){
    
    for(i in 1:length(structured_parts)){
      
      if(!is.null(offset[[i]]) & offset[[i]]!=0)
        structured_parts[[i]] <- layer_add(list(structured_parts[[i]],
                                               offset_layers[[i]]))
      
    }
    
  }
  
  # concatenate predictors
  # -> just to split them later again?
  if(length(structured_parts) > 1)
    preds <- layer_concatenate(structured_parts) else
      preds <- structured_parts[[1]]
  
  if(mixture_dist){
    list_pred <- layer_lambda(preds,
                              f = function(x)
                              {
                                tf$split(x, num_or_size_splits =
                                           c(1L, as.integer(nr_params-1)),
                                         axis = 1L)
                              })
    list_pred[[1]] <- list_pred[[1]] %>%
      layer_dense(units = as.integer(mixture_dist),
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
  inputList <- unname(inputs_struct[!sapply(inputs_struct, is.null)])
  
  if(!is.null(offset)){
    
    inputList <- c(inputList,
                   unlist(offset_inputs[!sapply(offset_inputs, is.null)]))
    
  }
  
  # Plist <- lapply(list_structured, function(x) x$P)
  # Plist <- Plist[!sapply(Plist, is.null)]
  # Plist <- lapply(Plist, function(x) lapply(1:length(x), function(i) x[[i-1]]))
  
  kerasGAM <- do.call("build_kerasGAM", #c(list(Plist = Plist[[1]]), 
                      fsbatch_options)#)
  
  # the final model is defined by its inputs
  # and outputs
  model <- kerasGAM(inputs = inputList,
                    outputs = out)
  
  l <- 1  
  for(i in 1:length(list_structured)){

    if(!is.null(list_structured[[i]])){
      reg <- list_structured[[i]]$get_penalty
      
      # lambdas <- list_structured[[i]]$mask * tf$exp(
      #     list_structured[[i]]$lambdas)/
      #   tf$cast(list_structured[[i]]$n, "float32")
      # P <- list_structured[[i]]$P
      # bigP = as.matrix(
      #   bdiag(lapply(1:length(lambdas),
      #                function(j) as.matrix(lambdas[j]*tf$cast(
      #                  P[j-1]$to_dense(), dtype = "float32")))))
      # layer_nr <- grep("pen_linear", 
      #                   sapply(1:length(model$trainable_weights), function(k) 
      #                     model$trainable_weights[[k]]$name))[l]
      # reg <- function(x){ 
      #   return(
      #     tf$reduce_sum(tf_incross(
      #       model$trainable_weights[[layer_nr]],
      #       tf$cast(bigP, "float32")))
      #   )
      # }
      model$add_loss(reg)
      l <- l + 1
    }
  }
  
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
  model %>% compile(optimizer = optimizer,
                    loss = negloglik,
                    metrics = monitor_metric)
  
  return(model)

}


#' @title Initializing Deep Transformation Models
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
#' @param lambda_lasso penalty parameter for l1 penalty of structured layers
#' @param lambda_ridge penalty parameter for l2 penalty of structured layers
#' @param weights observation weights used in the likelihood
#' @param learning_rate learning rate for optimizer
#' @param optimizer optimizer used (defaults to adam)
#' @param monitor_metric see \code{?deepregression}
#' @param orthog_fun function defining the orthogonalization
#' @param orthogX vector of columns defining the orthgonalization layer
#' @param split_fun see \code{?deepregression}
#' @param order_bsp NULL or integer; order of Bernstein polynomials; if not NULL,
#' a conditional transformation model (CTM) is fitted.
#' @param use_bias_in_structured whether or not to use a bias in structured
#' layers
#' @param train_together see \code{?deepregression}
#' @param split_between_shift_and_theta see \code{?deepregression}
#' @param interact_pred_trafo specifies a transformation function applied
#' to the interaction predictor using a layer lambda (e.g. to ensure positivity)
#' @param addconst_interaction additive constant added to predictor matrix to
#' ensure positivity
#' @param penalize_bsp scalar value > 0; amount of penalization of Bernstein polynomials
#' @param order_bsp_penalty integer; order of Bernstein polynomial penalty. 0 results in a
#' penalty based on integrated squared second order derivatives, values >= 1 in difference
#' penalties
#' @param base_distribution a string ("normal", "logistic") or TFP distribution; 
#' the base distribution for the transformation model. 
#' Per default \code{tfd_normal(loc = 0, scale = 1)} but any other distribution is possible (e.g.,
#' \code{tfd_logistic(loc = 0, scale = 1)}).
#'
#' @export
#'
deeptransformation_init <- function(
  n_obs,
  ncol_structured,
  ncol_deep,
  list_structured,
  list_deep,
  support,
  lambda_lasso=NULL,
  lambda_ridge=NULL,
  weights = NULL,
  learning_rate = 0.01,
  optimizer = optimizer_adam(lr = learning_rate),
  monitor_metric = list(),
  orthog_fun = orthog,
  orthogX = NULL,
  split_fun = split_model,
  order_bsp,
  use_bias_in_structured = FALSE,
  train_together = NULL,
  split_between_shift_and_theta = NULL,
  interact_pred_trafo = NULL,
  addconst_interaction = NULL,
  penalize_bsp = 0,
  order_bsp_penalty = 2,
  base_distribution = "normal",
  batch_shape = NULL,
  atm_lags = 0,
  atm_toplayer = NULL
)
{

  nr_params = 2 # shift & interaction term
  output_dim = rep(1, nr_params) # only univariate responses
  # if(length(list_deep)==1 & is.null(list_deep[[1]]))
  #   list_deep <- list_deep[rep(1,2)]
  
  
  # define the input layers
  inputs_deep <- lapply(ncol_deep, function(param_list){
    if(is.list(param_list) & length(param_list)==0) return(NULL)
    lapply(param_list, function(nc){
      
      if(!is.null(batch_shape)) shape <- NULL else{
        
        if(sum(unlist(nc))!=0){
          if(is.list(nc) & length(nc)>1)
            shape <- list(as.integer(sum(unlist(nc)))) else if(is.list(nc) & length(nc)==1)
              shape <- as.list(as.integer(nc[[1]]))
        }else{
          return(NULL)
        }     
      }
      return(
        layer_input(shape = shape,
                    batch_shape = batch_shape)
        )
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
  }

  # inputs for BSP trafo of Y, both n x tilde{M}
  input_theta_y <- layer_input(shape = list(order_bsp+1L))
  input_theta_y_prime <- layer_input(shape = list(order_bsp+1L))
  if(atm_lags){ 
    
    arlags <- ar_lags_layer(order_bsp, support)
    input_theta_atm <- layer_input(shape = list(as.integer(atm_lags)))
    input_atm_bsp <- 
      tf$split(input_theta_atm,
               num_or_size_splits = as.integer(atm_lags),
               axis = 1L) %>% 
      arlags
    
  }else{
      input_theta_atm <- input_atm_bsp <- NULL
  }
  
  structured_parts <- vector("list", 2)

  # define structured predictor
  if(is.null(inputs_struct[[1]]))
  {
    structured_parts[[1]] <-  NULL

  }else{

    if(is.null(list_structured[[1]]))
    {
      if(!is.null(lambda_lasso) & is.null(lambda_ridge)){

        l1 = tf$keras$regularizers$l1(l=lambda_lasso)

        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            kernel_regularizer = l1,
            name = paste0("structured_lasso_",
                          1))

      }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){

        l2 = tf$keras$regularizers$l2(l=lambda_ridge)

        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            kernel_regularizer = l2,
            name = paste0("structured_ridge_",
                          1))


      }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){

        l12 = tf$keras$regularizers$l1_l2(l1=lambda_lasso,
                                          l2=lambda_ridge)

        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            kernel_regularizer = l12,
            name = paste0("structured_elastnet_",
                          1))

      }else{

        structured_parts[[1]] <- inputs_struct[[1]] %>%
          layer_dense(
            units = as.integer(output_dim[1]),
            activation = "linear",
            use_bias = use_bias_in_structured,
            name = paste0("structured_linear_",
                          1))

      }

    }else{

      this_layer <- list_structured[[1]]
      structured_parts[[1]] <- inputs_struct[[1]] %>% this_layer

    }
  }


  if(!is.null(train_together) && !is.null(list_deep) &
     !(length(list_deep)==1 & is.null(list_deep[[1]])))
    list_deep_shared <- list_deep[sapply(names(list_deep),function(nnn)
      !nnn%in%names(ncol_deep[1:nr_params]))] else
        list_deep_shared <- NULL

  list_deep <- lapply(ncol_deep[1:nr_params], function(param_list){
    lapply(names(param_list), function(nn){
      if(is.null(nn)) return(NULL) else
        list_deep[[nn]]
    })
  })

  # define deep predictor
  deep_parts <- lapply(1:length(list_deep), function(i)
    if(is.null(inputs_deep[[i]]) | length(inputs_deep[[i]])==0)
      return(NULL) else
        lapply(1:length(list_deep[[i]]), function(j)
          list_deep[[i]][[j]](inputs_deep[[i]][[j]])))

  if(!is.null(train_together) && !is.null(list_deep_shared) &
     any(!sapply(inputs_deep, is.null))){

    shared_parts <- lapply(unique(unlist(train_together)), function(i)
      list_deep_shared[[i]](
        inputs_deep[[nr_params + i]][[1]]
      ))

    deep_parts[[1]] <- lapply(shared_parts, function(spa) spa[
      ,1:as.integer(split_between_shift_and_theta[1]),drop=F])
    deep_parts[[2]] <- lapply(shared_parts, function(spa) spa[
      ,(as.integer(split_between_shift_and_theta[1])+1L):
        (as.integer(sum(split_between_shift_and_theta))),drop=F])

  }

  ############################################################
  ################# Apply Orthogonalization ##################

  # create final linear predictor per distribution parameter
  # -> depending on the presence of a deep or structured part
  # the corresponding part is returned. If both are present
  # the deep part is projected into the orthogonal space of the
  # structured part

  deep_top_shift <- NULL
  if(!is.null(deep_parts[[1]]))
    deep_top_shift <- list(function(x) layer_dense(x, units = 1,
                                                   activation = "linear"))[
                                                     rep(1,length(deep_parts[[1]]))]

  ## shift term
  final_eta_pred <- combine_model_parts(deep = deep_parts[[1]],
                                        deep_top = deep_top_shift,
                                        struct = structured_parts[[1]],
                                        ox = ox[[1]],
                                        orthog_fun = orthog_fun,
                                        shared = NULL)

  ## interaction term
  if(is.null(deep_parts[[2]])){

    deep_part_ia <- NULL

  }else if(is.null(ox[[2]]) | (is.null(ox[[2]][[1]]) & length(ox[[2]])==1)){

    if(length(deep_parts[[2]])==1)
      deep_part_ia <- deep_parts[[2]][[1]] else
        deep_part_ia <- layer_add(deep_parts[[2]])

  }else{

    deep_part_ia <- orthog_fun(deep_parts[[2]], ox[[2]])

  }

  if(is.null(deep_parts[[2]])){

    interact_pred <- inputs_struct[[2]]

  }else if(is.null(inputs_struct[[2]])){

    interact_pred <- deep_part_ia

  }else{

    interact_pred <- layer_concatenate(list(inputs_struct[[2]],deep_part_ia))

  }

  if(!is.null(interact_pred_trafo)){

    # define Gamma weights
    thetas_layer <- layer_mono_multi_trafo(input_shape =
                                             list(NULL, (order_bsp+1L)*
                                                    (ncol(interact_pred)[[1]])),
                                           dim_bsp = c(order_bsp+1L))

    rho_part <- tf_row_tensor_right_part(input_theta_y, interact_pred) %>%
      thetas_layer() %>%
      layer_lambda(f = interact_pred_trafo)

    # rho_part <- tf$add(
    #   tf$constant(matrix(
    #     c(rep(neg_shift_bsp, (ncol(interact_pred)[[1]])),
    #       rep(0, (ncol(interact_pred)[[1]])*order_bsp)
    #     ), nrow=1), dtype="float32"),
    #   rho_part)

    aTtheta <- tf$matmul(
      tf$multiply(tf_row_tensor_left_part(input_theta_y,
                                          interact_pred),
                  rho_part),
      tf$ones(shape = c((order_bsp+1L)*(ncol(interact_pred)[[1]]),1))
    )
    aPrimeTtheta <- tf$matmul(
      tf$multiply(tf_row_tensor_left_part(input_theta_y_prime,
                                          interact_pred),
                  rho_part),
      tf$ones(shape = c((order_bsp+1L)*(ncol(interact_pred)[[1]]),1))
    )

  }else{

    # define Gamma weights
    thetas_layer <- layer_mono_multi(input_shape =
                                       list(NULL, (order_bsp+1L)*
                                              (ncol(interact_pred)[[1]])),
                                     dim_bsp = c(order_bsp+1L))

    ## thetas
    AoB <- tf_row_tensor(input_theta_y, interact_pred)
    AprimeoB <- tf_row_tensor(input_theta_y_prime, interact_pred)

    aTtheta <- AoB %>% thetas_layer()
    aPrimeTtheta <- AprimeoB %>% thetas_layer()

    # if(!is.null(addconst_interaction))
    # {
    #
    #   correction <- tf$multiply(tf$constant(matrix(addconst_interaction), dtype="float32"),
    #                             tf_row_tensor_left_part(input_theta_y, interact_pred)) %>%
    #     thetas_layer()
    #   correction_prime <- tf$multiply(tf$constant(matrix(addconst_interaction), dtype="float32"),
    #                                   tf_row_tensor_left_part(input_theta_y_prime, interact_pred)) %>%
    #     thetas_layer()
    #
    #   aTtheta <- tf$add(aTtheta, correction)
    #   aPrimeTtheta <- tf$add(aPrimeTtheta, correction_prime)
    #
    # }

  }
  
  # Check if model is SATs
  if(atm_lags){
    
    # if h1 is given, transform F_{t-1} using h1
    if(!is.null(interact_pred_trafo)){
      
     rho_parts <- lapply(input_atm_bsp, function(inp) 
       tf_row_tensor_right_part(inp, interact_pred) %>%
        thetas_layer() %>%
        layer_lambda(f = interact_pred_trafo))

      
      aTtheta_lags <- lapply(rho_parts, function(rp) tf$matmul(
        tf$multiply(tf_row_tensor_left_part(input_theta_y,
                                            interact_pred),
                    rp),
        tf$ones(shape = c((order_bsp+1L)*(ncol(interact_pred)[[1]]),1))
      ))
      
      
      # otherwise just combine lags with interact_pred via RWT
      # and add thetas layer on top
    }else{ 

      ## thetas
      AoBs <- lapply(input_atm_bsp, function(inp) 
        tf_row_tensor(inp, interact_pred))
      
      aTtheta_lags <- lapply(AoBs, function(aob) aob %>% thetas_layer())

      
    }
    
    # SAT with no additional h2 
    # just learn linear combination of transformed lags
    if(is.null(atm_toplayer))
      atm_toplayer <- function(x){ x %>% layer_dense(1) }
    
    aTtheta_lags <- if(length(aTtheta_lags)==1) aTtheta_lags[[1]] else layer_concatenate(aTtheta_lags)
    final_eta_pred <- layer_add(list(final_eta_pred, (aTtheta_lags %>% atm_toplayer)))
    
  } # SATs end

  # if(!is.null(addconst_interaction))
  # {
  #
  #   modeled_terms <- layer_concatenate(list(
  #     final_eta_pred,
  #     aTtheta,
  #     aPrimeTtheta,
  #     correction,
  #     correction_prime
  #     # tf$add(tf$multiply(tf$constant(matrix(0),dtype="float32"), aTtheta), correction)
  #   ))
  #
  # }else{
  
  modeled_terms <- layer_concatenate(list(
    final_eta_pred,
    aTtheta,
    aPrimeTtheta
  ))
  
  # }
  
  # evaluate base_distribution once
  # otherwise it will be a symbolic tensor
  if(is.null(base_distribution) || (is.character(base_distribution) & 
                                    base_distribution=="normal")){
    bd <- tfd_normal(loc = 0, scale = 1)
  }else if((is.character(base_distribution) & 
            base_distribution=="logistic")){ 
    bd <- tfd_logistic(loc = 0, scale = 1)
  }else{
    bd <- base_distribution
  }
  
  neg_ll <- function(y, model) {
    
    # shift term/lin pred
    w_eta <- model[, 1, drop = FALSE]
    
    # first part of the loglikelihood, n x (order + 1)
    aTtheta <- model[, 2, drop = FALSE]
    aTtheta_shift <- aTtheta + w_eta
    first_term <- bd %>% tfd_log_prob(aTtheta_shift)
    
    # second part of the loglikelihood
    aPrimeTtheta <- model[, 3, drop =  FALSE]
    sec_term <- tf$math$log(tf$clip_by_value(aPrimeTtheta, 1e-8, Inf))
    
    neglogLik <- -1 * (first_term + sec_term)
    
    return(neglogLik)
  }
  
  
  inputList <- unname(c(
    unlist(inputs_deep[!sapply(inputs_deep, is.null)],
           recursive = F),
    inputs_struct[!sapply(inputs_struct, is.null)]))
  
  if(!is.null(input_theta_atm))
    inputList <- unname(c(inputList, list(input_theta_atm)))
  
  inputList <- unname(c(inputList, 
                        unlist(ox[!sapply(ox, is.null)]),
                        input_theta_y,
                        input_theta_y_prime
  )
  )

  model <- keras_model(inputs = inputList,
                       outputs = modeled_terms)

  mono_layer_ind <- grep(
    "constraint_mono_layer",
    sapply(model$trainable_weights, function(x) x$name)
  )

  # add penalty for interaction term
  if(is.null(list_structured[[2]]))
  {
    if(!is.null(lambda_lasso) & is.null(lambda_ridge)){

      reg = function(x) tf$keras$regularizers$l1(l=lambda_lasso)(
        model$trainable_weights[[mono_layer_ind]]
        )

    }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){

      reg = function(x) tf$keras$regularizers$l2(l=lambda_ridge)(
        model$trainable_weights[[mono_layer_ind]]
        )


    }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){

      reg = function(x) tf$keras$regularizers$l1_l2(
        l1=lambda_lasso,
        l2=lambda_ridge)(model$trainable_weights[[mono_layer_ind]])

    }else{

      reg = NULL # no penalty

    }

    reg2 = NULL

  }else{

    if(penalize_bsp)
      bspP <- secondOrderPenBSP(order_bsp, order_diff = order_bsp_penalty)
    bigP <- list_structured[[2]]
    if(!is.null(deep_part_ia))
    {

      bigP <- bdiag(list(bigP, diag(rep(1, ncol(deep_part_ia)[[1]]))))

    }
    if(length(bigP@x)==0 & penalize_bsp==0){
      reg = NULL
    }else if(penalize_bsp==0){

      reg = function(x) k_mean(k_batch_dot(model$trainable_weights[[mono_layer_ind]], k_dot(
        # tf$constant(
        sparse_mat_to_tensor(as(kronecker(diag(rep(1, ncol(input_theta_y)[[1]])),bigP),
                                "CsparseMatrix")),
        # dtype = "float32"),
        model$trainable_weights[[mono_layer_ind]]),
        axes=2) # 1-based
      )

    }else if(length(bigP@x)==0)
    {

      reg = function(x) k_mean(k_batch_dot(model$trainable_weights[[mono_layer_ind]],
                                           k_dot(
        # tf$constant(
        sparse_mat_to_tensor(as(kronecker(penalize_bsp*bspP,
                                          diag(rep(1, ncol(interact_pred)[[1]]))),
                                "CsparseMatrix")),
        # dtype = "float32"),
        model$trainable_weights[[mono_layer_ind]]),
        axes=2) # 1-based
      )

    }else{

      reg = function(x) k_mean(k_batch_dot(model$trainable_weights[[mono_layer_ind]],
                                           k_dot(
        # tf$constant(
        sparse_mat_to_tensor(
          as(kronecker(penalize_bsp*bspP, diag(rep(1, ncol(interact_pred)[[1]])))
             + kronecker(diag(rep(1, ncol(input_theta_y)[[1]])),bigP), "CsparseMatrix")),
        # dtype = "float32"),
        model$trainable_weights[[mono_layer_ind]]),
        axes=2) # 1-based
      )

    }

    if(!is.null(lambda_lasso) & is.null(lambda_ridge)){

      reg2 = function(x) tf$keras$regularizers$l1(l=lambda_lasso)(
        model$trainable_weights[[mono_layer_ind]])

    }else if(!is.null(lambda_ridge) & is.null(lambda_lasso)){

      reg2 = function(x) tf$keras$regularizers$l2(l=lambda_ridge)(
        model$trainable_weights[[mono_layer_ind]])


    }else if(!is.null(lambda_ridge) & !is.null(lambda_lasso)){

      reg2 = function(x) tf$keras$regularizers$l1_l2(
        l1=lambda_lasso,
        l2=lambda_ridge)(model$trainable_weights[[mono_layer_ind]])

    }else{

      reg2 = NULL

    }

  }

  # add penalization
  if(!is.null(reg)) model$add_loss(reg)
  # add additional l1 or l2
  if(!is.null(reg2)) model$add_loss(reg2)
  
  model %>% compile(
    optimizer = optimizer,
    loss      = neg_ll,
    metrics   = monitor_metric
  )

  return(model)


}
