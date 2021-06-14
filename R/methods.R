#' @title Generic functions for deepregression models
#'
#' @param x deepregression object
#' @param which which effect to plot, default selects all.
#' @param which_param integer of length 1.
#' Corresponds to the distribution parameter for
#' which the effects should be plotted.
#' @param plot logical, if FALSE, only the data for plotting is returned
#' @param use_posterior logical; if \code{TRUE} it is assumed that
#' the strucuted_nonlinear layer has stored a list of length two
#' as weights, where the first entry is a vector of mean and sd
#' for each network weight. The sd is transformed using the \code{exp} function.
#' The plot then shows the mean curve +- 2 times sd.
#' @param grid_length the length of an equidistant grid at which a two-dimensional function
#' is evaluated for plotting.
#' @param ... further arguments, passed to fit, plot or predict function
#'
#' @method plot deepregression
#' @export
#' @rdname methodDR
#'
plot.deepregression <- function(
  x,
  which = NULL,
  # which of the nonlinear structured effects
  which_param = 1, # for which parameter
  plot = TRUE,
  use_posterior = FALSE,
  grid_length = 40,
  ... # passed to plot function
)
{
  this_ind <- x$init_params$ind_structterms[[which_param]]
  if(all(this_ind$type!="smooth")) return("No smooth effects. Nothing to plot.")
  if(is.null(which)) which <- 1:length(which(this_ind$type=="smooth"))
  plus_number_lin_eff <- sum(this_ind$type=="lin")

  plotData <- vector("list", length(which))
  org_feature_names <-
    names(x$init_params$l_names_effects[[which_param]][["smoothterms"]])
  phi <- tryCatch(x$model$get_layer(paste0("structured_nonlinear_",
                                  which_param))$get_weights(), 
                  error = function(e){
                    layer_nr <- grep("pen_linear", 
                                     sapply(1:length(x$model$trainable_weights), 
                                            function(k) x$model$trainable_weights[[k]]$name))
                    # FixMe: multiple penalized layers
                    list(as.matrix(x$model$trainable_weights[[layer_nr[which_param]]]))
                    })
  
  if(length(phi)>1){
    if(use_posterior){
      phi <- matrix(phi[[1]], ncol=2, byrow=F)
    }else{
      phi <- as.matrix(phi[[2]], ncol=1)
    }
  }else{
    phi <- phi[[1]]
  }

  for(w in which){

    nam <- org_feature_names[w]
    this_ind_this_w <- do.call("Map",
                               c(":", as.list(this_ind[w+plus_number_lin_eff,
                                                       c("start","end")])))[[1]]
    BX <-
      x$init_params$parsed_formulae_contents[[
        which_param]]$smoothterms[[nam]][[1]]$X
    if(use_posterior){

      # get the correct index as each coefficient has now mean and sd
      phi_mean <- phi[this_ind_this_w,1]
      phi_sd <- log(exp(log(expm1(1)) + phi[this_ind_this_w,2])+1)
      plotData[[w]] <-
        list(org_feature_names = nam,
             value = unlist(x$init_params$data[strsplit(nam,",")[[1]]]),
             design_mat = BX,
             coef = phi[this_ind_this_w,],
             mean_partial_effect = BX%*%phi_mean,
             sd_partial_effect = sqrt(diag(BX%*%diag(phi_sd^2)%*%t(BX))))
    }else{
      plotData[[w]] <-
        list(org_feature_name = nam,
             value = sapply(strsplit(nam,",")[[1]], function(xx)
               x$init_params$data[[xx]]),
             design_mat = BX,
             coef = phi[this_ind_this_w,],
             partial_effect = BX%*%phi[this_ind_this_w,])
    }
    if(plot){
      nrcols <- pmax(NCOL(plotData[[w]]$value), length(unlist(strsplit(nam,","))))
      if(nrcols==1)
      {
        if(use_posterior){
          plot(plotData[[w]]$mean_partial_effect[order(plotData[[w]]$value)] ~
                 sort(plotData[[w]]$value),
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ylim = c(min(plotData[[w]]$mean_partial_effect -
                              2*plotData[[w]]$sd_partial_effect),
                        max(plotData[[w]]$mean_partial_effect +
                              2*plotData[[w]]$sd_partial_effect)),
               ...)
          with(plotData[[w]], {
            points((mean_partial_effect + 2 * sd_partial_effect)[order(plotData[[w]]$value)] ~
                     sort(plotData[[w]]$value), type="l", lty=2)
            points((mean_partial_effect - 2 * sd_partial_effect)[order(plotData[[w]]$value)] ~
                     sort(plotData[[w]]$value), type="l", lty=2)
          })
        }else{
          plot(partial_effect[order(value),1] ~ sort(value[,1]),
               data = plotData[[w]][c("value", "partial_effect")],
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ...)
        }
      }else if(nrcols==2){
        sTerm <- x$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[w]][[1]]
        this_x <- do.call(seq, c(as.list(range(plotData[[w]]$value[,1])),
                                 list(l=grid_length)))
        this_y <- do.call(seq, c(as.list(range(plotData[[w]]$value[,2])),
                                 list(l=grid_length)))
        df <- as.data.frame(expand.grid(this_x,
                                        this_y))
        colnames(df) <- sTerm$term
        pmat <- PredictMat(sTerm, data = df)
        if(attr(x$init_params$parsed_formulae_contents[[which_param]],"zero_cons"))
          pmat <- orthog_structured_smooths(pmat,P=NULL,L=matrix(rep(1,nrow(pmat)),ncol=1))
        pred <- pmat%*%phi[this_ind_this_w,]
        #this_z <- plotData[[w]]$partial_effect
        suppressWarnings(
          filled.contour(
            this_x,
            this_y,
            matrix(pred, ncol=length(this_y)),
            ...,
            xlab = colnames(df)[1],
            ylab = colnames(df)[2],
            # zlab = "partial effect",
            main = sTerm$label
          )
        )
        # warning("Plotting of effects with ", nrcols, "
        #         covariate inputs not supported yet.")
      }else{
        warning("Plotting of effects with ", nrcols,
                " covariate inputs not supported.")
      }
    }
  }

  invisible(plotData)
}

#' @title Generic functions for deeptrafo models
#'
#' @param x deepregression object
#' @param which which effect to plot, default selects all.
#' @param which_param integer, either 1 or 2.
#' 1 corresponds to the shift term, 2 to the interaction term.
#' @param plot logical, if FALSE, only the data for plotting is returned
#' @param grid_length the length of an equidistant grid at which a two-dimensional function
#' is evaluated for plotting.
#' @param eval_grid logical; should plot be evaluated on a grid
#' @param ... further arguments, passed to fit, plot or predict function
#'
#' @method plot deeptrafo
#' @export
#' @rdname methodTrafo
#'
plot.deeptrafo <- function(
  x,
  which = NULL,
  # which of the nonlinear structured effects
  which_param = 1, # for which parameter
  plot = TRUE,
  grid_length = 40,
  eval_grid = FALSE,
  ... # passed to plot function
)
{
  this_ind <- x$init_params$ind_structterms[[which_param]]
  if(all(this_ind$type!="smooth")) return("No smooth effects. Nothing to plot.")
  if(is.null(which)) which <- 1:length(which(this_ind$type=="smooth"))
  plus_number_lin_eff <- sum(this_ind$type=="lin")

  plotData <- vector("list", length(which))
  org_feature_names <-
    names(x$init_params$l_names_effects[[which_param]][["smoothterms"]])
  if(which_param==1){
    phi <- matrix(get_shift(x), ncol=1)
  }else{
    phi <- get_theta(x)
    # FIXME: Is the following necessary?
    if(ncol(phi)>1) phi <- t(phi)
  }

  for(w in which){

    nam <- org_feature_names[w]
    this_ind_this_w <- do.call("Map",
                               c(":", as.list(this_ind[w+plus_number_lin_eff,
                                                       c("start","end")])))[[1]]
    BX <-
      x$init_params$parsed_formulae_contents[[
        which_param]]$smoothterms[[nam]][[1]]$X

    plotData[[w]] <-
      list(org_feature_name = nam,
           value = sapply(gsub("by = ","",strsplit(nam,",")[[1]]), function(xx)
             x$init_params$data[[xx]]),
           design_mat = BX,
           coef = phi[this_ind_this_w,],
           partial_effects = BX%*%phi[this_ind_this_w,])

    nrcols <- pmax(NCOL(plotData[[w]]$value), length(unlist(strsplit(nam,","))))

    if(plot | nrcols > 2 | eval_grid){
      if(which_param==1){

        if(nrcols==1)
        {

          plot(partial_effects[order(value),1] ~ sort(value[,1]),
               data = plotData[[w]][c("value", "partial_effects")],
               main = paste0("s(", nam, ")"),
               xlab = nam,
               ylab = "partial effect",
               ...)

        }else if(nrcols==2){
          sTerm <- x$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[w]][[1]]
          this_x <- do.call(seq, c(as.list(range(plotData[[w]]$value[,1])),
                                   list(l=grid_length)))
          this_y <- do.call(seq, c(as.list(range(plotData[[w]]$value[,2])),
                                   list(l=grid_length)))
          df <- as.data.frame(expand.grid(this_x,
                                          this_y))
          colnames(df) <- gsub("by = ","",strsplit(nam,",")[[1]])
          pmat <- PredictMat(sTerm, data = df)
          if(attr(x$init_params$parsed_formulae_contents[[which_param]],"zero_cons") & 
             !is.null(x$init_params$parsed_formulae_contents[[which_param]]$linterms) & 
             !any(grepl("by =", strsplit(nam,",")[[1]])))
            pmat <- orthog_structured_smooths(pmat,P=NULL,L=matrix(rep(1,nrow(pmat)),ncol=1))
          pred <- pmat%*%phi[this_ind_this_w,]
          #this_z <- plotData[[w]]$partial_effect

          plotData[[w]] <- list(df = df,
                                design_mat = pmat,
                                coef = phi[this_ind_this_w,],
                                pred = pred)

          if(plot)
            suppressWarnings(
              filled.contour(
                this_x,
                this_y,
                matrix(pred, ncol=length(this_y)),
                ...,
                xlab = colnames(df)[1],
                ylab = colnames(df)[2],
                # zlab = "partial effect",
                main = sTerm$label
              )
            )
          # warning("Plotting of effects with ", nrcols, "
          #         covariate inputs not supported yet.")
        }else if(nrcols==3){

          if(plot) warning("Will only return the plot data for 3d effects.")

          sTerm <- x$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[w]][[1]]
          this_x <- do.call(seq, c(as.list(range(plotData[[w]]$value[,1])),
                                   list(l=grid_length)))
          this_y <- do.call(seq, c(as.list(range(plotData[[w]]$value[,2])),
                                   list(l=grid_length)))
          this_z <- do.call(seq, c(as.list(range(plotData[[w]]$value[,3])),
                                   list(l=grid_length)))
          df <- as.data.frame(expand.grid(this_x,
                                          this_y,
                                          this_z))
          colnames(df) <- sTerm$term
          if(sTerm$by!="NA") colnames(df)[ncol(df)] <- sTerm$by
          pmat <- PredictMat(sTerm, data = df)
          if(attr(x$init_params$parsed_formulae_contents[[which_param]],"zero_cons") & 
             !is.null(x$init_params$parsed_formulae_contents[[which_param]]$linterms) & 
             sTerm$by=="NA")
            pmat <- orthog_structured_smooths(pmat,P=NULL,L=matrix(rep(1,nrow(pmat)),ncol=1))
          pred <- pmat%*%phi[this_ind_this_w,]
          #this_z <- plotData[[w]]$partial_effect

          plotData[[w]] <- list(df = df,
                                design_mat = pmat,
                                coef = phi[this_ind_this_w,],
                                pred = pred)

        }else{
          warning("Plotting of effects with ", nrcols,
                  " covariate inputs not supported.")
        }
      }else{ # plot effects in theta
        
        suppressWarnings(
          matplot(
            #sort(plotData[[w]]$value[,1]),
            #1:ncol(plotData[[w]]$partial_effects),
            x=sort(plotData[[w]]$value[,1]),
            y=(plotData[[w]]$partial_effects[order(plotData[[w]]$value[,1]),]),
            ...,
            xlab = plotData[[w]]$org_feature_name,
            ylab = paste0("partial effects ", plotData[[w]]$org_feature_name),
            # zlab = "partial effect",
            type = "l"
          )
        )

      }
    }
  }

  invisible(plotData)
}


#' Function to prepare data for deepregression use
#'
#' @export
#'
#' @param x a deepregression object
#' @param data a data.frame or list
#' @param pred logical, where the data corresponds to a prediction task
#'
#' @rdname methodDR
#'
prepare_data <- function(
  x,
  data,
  pred=TRUE
)
{
  if(length(data)>1 & is.list(data) & !is.null(x$init_params$offset))
  {

    message("Using the second list item of data as offset.")
    newdata_processed <- prepare_newdata(
      x$init_params$parsed_formulae_contents,
      data[[1]], pred=pred,
      orthogonalize = x$init_params$orthogonalize,
      returnX = (x$init_params$orthog_type=="tf"))
    if(is.list(data[[2]]))
      data[[2]] <- unlist(data[[2]], recursive = FALSE)
    newdata_processed <- c(newdata_processed,
                           tf$constant(matrix(data[[2]], ncol = 1),
                                       dtype="float32"))

  }else{

    newdata_processed <- prepare_newdata(
      x$init_params$parsed_formulae_contents,
      data, pred=pred,
      orthogonalize = x$init_params$orthogonalize,
      returnX = (x$init_params$orthog_type=="tf"))

    # for trafo models
    if(length(x$init_params$parsed_formulae_contents)>1 &&
       !is.null(attr(x$init_params$parsed_formulae_contents[[2]], "minval")))
      return(list(data=newdata_processed,
                  minval=attr(x$init_params$parsed_formulae_contents[[2]], "minval")))

  }
  return(newdata_processed)
}

#' Predict based on a deepregression object
#'
#' @param object a deepregression model
#' @param newdata optional new data, either data.frame or list
#' @param batch_size batch_size for generator (image use cases)
#' @param apply_fun which function to apply to the predicted distribution,
#' per default \code{tfd_mean}, i.e., predict the mean of the distribution
#' @param convert_fun how should the resulting tensor be converted,
#' per default \code{as.matrix}
#' @param dtype string for conversion 
#'
#' @export
#' @rdname methodDR
#'
predict.deepregression <- function(
  object,
  newdata = NULL,
  batch_size = NULL,
  apply_fun = tfd_mean,
  convert_fun = as.matrix,
  dtype = "float32",
  ...
)
{

  if(length(object$init_params$image_var)>0){
   
    if(!is.null(newdata)){
      if(is.data.frame(newdata)) newdata <- as.list(newdata)
      newdata_processed <- prepare_data(object, newdata, pred=FALSE)
      
      data_tab <- list(newdata_processed)
      data_image <- as.data.frame(newdata[names(object$init_params$image_var)], 
                                  stringsAsFactors = FALSE)
      
    }else{

      data_tab <- list(unname(object$init_params$input_cov))
      data_image <- as.data.frame(object$init_params$data[names(object$init_params$image_var)], 
                                  stringsAsFactors = FALSE)
    }
    # prepare generator
    max_data <- NROW(data_image)
    if(is.null(batch_size)) batch_size <- 20
    steps_per_epoch <- ceiling(max_data/batch_size)
    
    generator <- make_generator(data_image, data_tab, batch_size, 
                                # FIXME: multiple images
                                target_size = unname(unlist(object$init_params$image_var)[1:2]),
                                color_mode = unname(ifelse(
                                  unlist(object$init_params$image_var)[3]==3, 
                                  "rgb", "grayscale")),
                                x_col = names(object$init_params$image_var),
                                is_trafo = object$init_params$family=="transformation_model", 
                                shuffle = FALSE)
    
    if(is.null(apply_fun)) apply_fun <- function(x){x}
    return(sapply(1:steps_per_epoch, function(i) 
      convert_fun(apply_fun(object$model(generator$`__getitem__`(as.integer(i-1))))))
    )
  
  }else{
    
    if(is.null(newdata)){
      yhat <- object$model(lapply(unname(object$init_params$input_cov), function(x) 
        tf$cast(x,dtype)))
    }else{
      # preprocess data
      if(is.data.frame(newdata)) newdata <- as.list(newdata)
      newdata_processed <- prepare_data(object, newdata, pred=FALSE)
      newdata_processed <- lapply(unname(newdata_processed), function(x) tf$cast(x,dtype))
      yhat <- object$model(newdata_processed)
    }
   
    if(!is.null(apply_fun))
      return(convert_fun(apply_fun(yhat))) else
        return(convert_fun(yhat))
     
  }

}

#' Predict based on a deeptrafo object
#'
#' @param object a deeptrafo model
#' @param newdata optional new data, either data.frame or list
#' @param ... not used atm
#' @return returns a function with two parameters: the actual response
#' and \code{type} in \code{c('trafo', 'pdf', 'cdf', 'interaction')}
#' determining the returned value
#'
#' @export
#' @rdname methodDR
#'
predict.deeptrafo <- function(
  object,
  newdata = NULL,
  which = NULL,
  cast_float = FALSE,
  ...
)
{

  if(is.null(newdata)){
    inpCov <- unname(object$init_params$input_cov)
  }else{
    # preprocess data
    if(is.data.frame(newdata)) newdata <- as.list(newdata)
    inpCov <- prepare_data(object, newdata, pred=TRUE)
    if(cast_float) inpCov <- lapply(inpCov, function(x) tf$constant(x, dtype = "float32"))
    if(length(inpCov)==2 && !is.null(names(inpCov)) && names(inpCov)[2]=="minval")
    {
      minval <- inpCov[[2]]
      inpCov <- inpCov[[1]]
    }else{
      minval <- NULL
    }
    inpCov <- c(inpCov, list(NULL), list(NULL))
  }
  
  image_data <- NULL
  if(length(object$init_params$image_var)>0){
    if(!is.null(newdata)){
      image_data <- as.data.frame(newdata[names(object$init_params$image_var)], 
                                  stringsAsFactors = FALSE)
    }else{
      image_data <- as.data.frame(object$init_params$data[names(object$init_params$image_var)], 
                                  stringsAsFactors = FALSE) 
    }
  }

  # TODO: make prediction possible for one observation only; fix type mismatch pdf grid
  trafo_fun <- function(y, type = c("trafo", "pdf", "cdf", "interaction", "shift", "output", "sample"),
                        which = NULL, grid = FALSE, batch_size = NULL)
  {
    type <- match.arg(type)

    # if(!is.null(minval)) y <- y - sum(minval*get_theta(object))

    ay <- tf$cast(object$init_params$y_basis_fun(y), tf$float32)
    aPrimey <- tf$cast(object$init_params$y_basis_fun_prime(y), tf$float32)
    inpCov[length(inpCov)-c(1,0)] <- list(ay, aPrimey)
    mod_output <- evaluate.deeptrafo(object, inpCov, y, image_data, batch_size = batch_size)
    if(type=="output") return(mod_output)
    w_eta <- mod_output[, 1, drop = FALSE]
    aTtheta <- mod_output[, 2, drop = FALSE]
    # if(!is.null(minval))
    #   aTtheta <- aTtheta - sum(minval*get_theta(object))
    if(type=="interaction"){

      if(is.null(newdata))
        newdata <- object$init_params$data

      ret <- cbind(interaction = as.matrix(aTtheta),
                   as.data.frame(newdata)
      )

      if(ncol(mod_output)==5)
        ret <- cbind(ret, correction = as.matrix(mod_output[,4,drop=FALSE]))

      return(ret)

    }

    if(type=="shift"){

      if(is.null(newdata))
        newdata <- object$init_params$data

      return(cbind(shift = as.matrix(w_eta),
                   as.data.frame(newdata)))

    }
    ytransf <- aTtheta + w_eta
    yprimeTrans <- mod_output[, 3, drop = FALSE]
    # if(!is.null(minval))
    #   yprimeTrans + sum(minval*get_theta(object))
    theta <- get_theta(object)
    if(grid)
    {

      grid_eval <- t(as.matrix(
        tf$matmul(inpCov[[2]],
                  tf$transpose(
                    tf$matmul(ay,
                              tf$cast(theta, tf$float32)
                    )))))
      grid_eval <- grid_eval +
        t(as.matrix(w_eta)[,rep(1,nrow(grid_eval))])

      if(type=="pdf")
        grid_prime_eval <- t(as.matrix(
          tf$matmul(inpCov[[2]],
                    tf$transpose(
                      tf$matmul(aPrimey,
                                tf$cast(theta, tf$float32)
                      )))))


    }

    if(grid) type <- paste0("grid_",type)

    ret <- switch (type,
                   trafo = (ytransf %>% as.matrix),
                   pdf = ((tfd_normal(0,1) %>% tfd_prob(ytransf) %>%
                             as.matrix)*as.matrix(yprimeTrans)),
                   cdf = (tfd_normal(0,1) %>% tfd_cdf(ytransf) %>%
                            as.matrix),
                   grid_trafo = grid_eval,
                   grid_pdf = ((tfd_normal(0,1) %>% tfd_prob(grid_eval) %>%
                                  as.matrix)*as.matrix(grid_prime_eval)),
                   grid_cdf = (tfd_normal(0,1) %>% tfd_cdf(grid_eval) %>%
                                 as.matrix),
                   sample
    )

    return(ret)

  }

  return(trafo_fun)

}

evaluate.deeptrafo <- function(object, newdata, y, data_image, batch_size = NULL)
{
  
  
  if(length(object$init_params$image_var)>0){
    
    data_tab <- list(newdata, tf$cast(matrix(y, ncol=1), tf$float32))
    
    # prepare generator
    max_data <- NROW(data_image)
    if(is.null(batch_size)) batch_size <- 32
    steps_per_epoch <- ceiling(max_data/batch_size)
    
    generator <- make_generator(data_image, data_tab, batch_size, 
                                # FIXME: multiple images
                                target_size = unname(unlist(object$init_params$image_var)[1:2]),
                                color_mode = unname(ifelse(
                                  unlist(object$init_params$image_var)[3]==3, 
                                  "rgb", "grayscale")),
                                x_col = names(object$init_params$image_var),
                                is_trafo = object$init_params$family=="transformation_model", 
                                shuffle = FALSE)
    
    mod_output <- object$model$predict(generator)
    
  }else{
    
    if(is.null(batch_size)){
      
      mod_output <- object$model(list(newdata, tf$cast(matrix(y, ncol=1), tf$float32)))
    
    }else{
      
        max_data <- NROW(newdata[[1]])
        steps_per_epoch <- ceiling(max_data/batch_size)
        
        mod_output <- lapply(1:steps_per_epoch, 
                             function(i){
                               index <- (i-1)*batch_size + 1:batch_size
                               object$model(list(lapply(newdata, function(x) subset_array(x, index)), 
                                            tf$cast(matrix(y[index], ncol=1), tf$float32)))
                             })
        mod_output <- do.call("rbind", lapply(mod_output, as.matrix))
    
    }
    
  }
  
  return(mod_output)
  
}


#' Function to extract fitted distribution
#'
#' @param object a deepregression object
#' @param apply_fun function applied to fitted distribution,
#' per default \code{tfd_mean}
#' @param ... further arguments passed to the predict function
#'
#' @export
#' @rdname methodDR
#'
fitted.deepregression <- function(
  object, apply_fun = tfd_mean, ...
)
{
  return(
    predict.deepregression(object, apply_fun=apply_fun, ...)
  )
}

#' Generic fit function
#'
#' @param x object
#' @param ... further arguments passed to the class-specific function
#'
#' @export
fit <- function (x, ...) {
  UseMethod("fit", x)
}

#' Fit a deepregression model
#'
#' @param x a deepregresison object.
#' @param early_stopping logical, whether early stopping should be user.
#' @param verbose logical, whether to print losses during training.
#' @param view_metrics logical, whether to trigger the Viewer in RStudio / Browser.
#' @param patience integer, number of rounds after which early stopping is done.
#' @param save_weights logical, whether to save weights in each epoch.
#' @param auc_callback logical, whether to use a callback for AUC
#' @param validation_data optional specified validation data
#' @param callbacks a list of callbacks for fitting
#' @param convertfun function to convert R into Tensor object
#' @param ... further arguments passed to
#' \code{keras:::fit.keras.engine.training.Model}
#'
#'
#' @export fit deepregression
#' @export
#' 
#' @rdname methodDR
#'
fit.deepregression <- function(
  x,
  early_stopping = FALSE,
  verbose = TRUE,
  view_metrics = FALSE,
  patience = 20,
  save_weights = FALSE,
  auc_callback = FALSE,
  validation_data = NULL,
  callbacks = list(),
  convertfun = function(x) tf$constant(x, dtype="float32"),
  ...
)
{

  # make callbacks
  if(save_weights){
    weighthistory <- WeightHistory$new()
    callbacks <- append(callbacks, weighthistory)
  }
  if(early_stopping & length(callbacks)==0)
    callbacks <- append(callbacks,
                        callback_early_stopping(patience = patience))
  
  is_trafo = x$init_params$family=="transformation_model"

  if(auc_callback){

    if(is.null(validation_data)) stop("Must provide validation data via argument val_data.")
    if(is.data.frame(validation_data[[1]])) validation_data[[1]] <- as.list(validation_data[[1]])
    val_data_processed <- prepare_data(x, validation_data[[1]], pred=TRUE)

    auc_cb <- auc_roc$new(training = list(unname(x$init_params$input_cov), x$init_params$y),
                          validation = list(unname(val_data_processed), val_data[[2]]))
    callbacks <- append(callbacks,
                        auc_cb)
    verbose <- FALSE
  }
  
  # if(monitor_weights){
  #   # object$history <- WeightHistory$new()
  #   weight_callback <- callback_lambda(
  #     on_epoch_begin = function(epoch, logs)
  #       coef_prior_to_epoch <<- unlist(coef(object)),
  #     on_epoch_end = function(epoch, logs)
  #       print(sum(abs(coef_prior_to_epoch-unlist(coef(object)))))
  #     )
  #   this_callbacks <- append(this_callbacks, weight_callback)
  # }

  if(is.null(x$init_params$input_cov)){

    input_x <- prepare_newdata(x$init_params$parsed_formulae_contents,
                               x$init_params$data,
                               pred = FALSE,
                               orthogonalize = x$init_params$orthogonalize,
                               returnX = (x$init_params$orthog_type=="tf"))
    if(!is.null(x$init_params$offset))
      input_x <- c(input_x, unlist(lapply(x$init_params$offset, function(yy)
        tf$constant(matrix(yy, ncol = 1), dtype="float32")), recursive = FALSE))

  }else{

    input_x <- x$init_params$input_cov

  }

  # if(any(sapply(input_x, function(x)class(x)[1])=="placeholder")){
  if(length(x$init_params$image_var)>0){
      
    batch_size <- list(...)$batch_size
    if(is.null(batch_size)) batch_size <- 32
    input_y <- matrix(x$init_params$y, ncol=1)

    if(is.null(x$init_params$validation_split) & 
       is.null(list(...)$validation_split))
    {
      
      data_tab <- list(input_x, input_y)
      data_image <- as.data.frame(x$init_params$data[names(x$init_params$image_var)], 
                                  stringsAsFactors = FALSE)
      
      # only fit generator
      max_data <- NROW(input_x[[1]])
      steps_per_epoch <- ceiling(max_data/batch_size)

      generator <- make_generator(data_image, data_tab, batch_size, 
                                  # FIXME: multiple images
                                  target_size = unname(unlist(x$init_params$image_var)[1:2]),
                                  color_mode = unname(ifelse(
                                    unlist(x$init_params$image_var)[3]==3, 
                                                      "rgb", "grayscale")),
                                  x_col = names(x$init_params$image_var),
                                  is_trafo = is_trafo)
      
      if(!is.null(list(...)$validation_data) | 
         !is.null(x$init_params$validation_data)){
        validation_data <- if(!is.null(list(...)$validation_data))
          list(...)$validation_data else
            x$init_params$validation_data
        max_data <- NROW(validation_data[[1]][[1]])
        data_tab_val <- list(validation_data[[1]], 
                             validation_data[[2]])
        data_image_val <- as.data.frame(x$init_params$validation_data[
          names(x$init_params$image_var)], 
                                    stringsAsFactors = FALSE)
        validation_data <- make_generator(data_image_val, data_tab_val, batch_size, 
                                          # FIXME: multiple images
                                          target_size = unname(
                                            unlist(x$init_params$image_var)[1:2]),
                                          color_mode = unname(ifelse(
                                            unlist(x$init_params$image_var)[3]==3, 
                                            "rgb", "grayscale")),
                                          x_col = names(x$init_params$image_var),
                                          is_trafo = is_trafo)
        validation_steps <- ceiling(max_data/batch_size)
      }else{
        validation_data <- NULL
        validation_steps <- NULL
      }
    
      
    }else{
      
      if(!is.null(list(...)$validation_split)){
        val_split <- list(...)$validation_split 
      }else{
        val_split <- x$init_params$validation_split
      }

      input_x <- lapply(input_x, as.matrix)
      
      ind_val <- sample(1:NROW(input_y), round(NROW(input_y)*val_split))
      ind_train <- setdiff(1:NROW(input_y), ind_val)
      input_x_train <- subset_input_cov(input_x, ind_train)
      input_x_val <- subset_input_cov(input_x, ind_val)
      input_y_train <- matrix(subset_array(input_y, ind_train), ncol=1)
      input_y_val <- matrix(subset_array(input_y, ind_val), ncol=1)
                
      max_data_train <- NROW(input_x_train[[1]])
      steps_per_epoch <- ceiling(max_data_train/batch_size)

      data_tab <- list(input_x_train, input_y_train)
      data_image <- as.data.frame(x$init_params$data[names(x$init_params$image_var)], 
                                  stringsAsFactors = FALSE)[ind_train,,drop=FALSE]
      
      generator <- make_generator(data_image = data_image, 
                                  data_tab = data_tab, 
                                  batch_size = batch_size, 
                                  # FIXME: multiple images
                                  target_size = unname(unlist(x$init_params$image_var)[1:2]),
                                  color_mode = unname(ifelse(
                                    unlist(x$init_params$image_var)[3]==3, 
                                                             "rgb", "grayscale")),
                                  x_col = names(x$init_params$image_var),
                                  is_trafo = is_trafo)
      
      max_data_val <- NROW(input_x_val[[1]])
      validation_steps <- ceiling(max_data_val/batch_size)

      data_tab_val <- list(input_x_val, input_y_val)
      data_image_val <- as.data.frame(x$init_params$data[names(x$init_params$image_var)], 
                                  stringsAsFactors = FALSE)[ind_val,,drop=FALSE]
      
      validation_data <- make_generator(data_image = data_image_val, 
                                        data_tab = data_tab_val, 
                                        batch_size = batch_size, 
                                        # FIXME: multiple images
                                        target_size = unname(
                                          unlist(x$init_params$image_var)[1:2]),
                                        color_mode = unname(ifelse(unlist(
                                          x$init_params$image_var)[3]==3, 
                                          "rgb", "grayscale")),
                                        x_col = names(x$init_params$image_var),
                                        is_trafo = is_trafo)
      
    }
    
  }else{
    
    generator <- NULL
    
  }

  args <- list(...)
  input_list_model <-
    list(object = x$model,
         validation_split = x$init_params$validation_split,
         validation_data = x$init_params$validation_data,
         callbacks = callbacks,
         verbose = verbose,
         view_metrics = view_metrics
    )
  if(is.null(generator)){
    input_list_model <- c(input_list_model,
                          list(x = input_x,
                               y = x$init_params$y
                          ))
  }else{
    input_list_model$validation_split <- 
      input_list_model$validation_data <- NULL
    input_list_model <- c(input_list_model, list(
      generator = generator,
      steps_per_epoch = as.integer(steps_per_epoch),
      validation_data = validation_data,
      validation_steps = as.integer(validation_steps)
    ))
  }
  args <- append(args,
                 input_list_model[!names(input_list_model) %in%
                                    names(args)])
  # if(length(x$init_params$ellipsis)>0)
  #   args <- append(args,
  #                  x$init_params$ellipsis[
  #                    !names(x$init_params$ellipsis) %in% names(args)])
  
  if(!is.null(generator)){ 
    args$validation_split <- NULL
    args$batch_size <- NULL
  }

  if(is.null(generator))
    ret <- do.call(fit_fun, args) else
      ret <- do.call(fit_generator, args)
  if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer
  invisible(ret)
}

##' Fit a custom deepregression models 
##' 
##' @param x a deepregression_custom model (created by a call to 
##' \code{deepregression} with argument \code{compile_model = FALSE})
##' @param train function taking the keras model, inputs and outputs as an
##' updates the model
##' @param epochs integer; the number of epochs to train
##' @param verbose logical; whether or not to print progress
##' @param print_fun function to print metrics
##' @param ... further arguments passed to \code{train} function
##' 
## fit.deepregression_custom <- function(
##   x,
##   train,
#   epochs,
#   verbose = TRUE,
#   print_fun = function(x) paste(x, collapse = ", "),
#   ...
# )
# {
#   
#   data_image <- NULL
#   if(length(x$init_params$image_var)>0)
#     data_image <- as.data.frame(x$init_params$data[names(x$init_params$image_var)], 
#                                 stringsAsFactors = FALSE)
#   
#   for (epoch in epochs)) {
#     
#     current_loss <- train(model = x$model$model, 
#                           loss = x$model$loss
#                           inputs = x$init_params$input_cov, 
#                           outputs = x$init_params$y,
#                           images = data_image
#                           )
#     if(verbose) cat("Epoch:", epoch,", Loss:", as.numeric(current_loss), "\n")
#     # Track progress
#     # if(length(model$model$monitor_metric)>0){
#     #   if(is.list(model$model$monitor_metric))
#     #     sapply(model$model$monitor_metric(current_loss) else
#     #       model$model$monitor_metric(current_loss)
#     #   
#     # }
#   } 
#   
# }

#' Extract layer weights / coefficients from model
#'
#' @param object a deepregression model
#' @param variational logical, if TRUE, the function takes into account
#' that coefficients have both a mean and a variance
#' @param params integer, indicating for which distribution parameter
#' coefficients should be returned (default is all parameters)
#' @param type either NULL (all types of coefficients are returned),
#' "linear" for linear coefficients or "smooth" for coefficients of 
#' smooth terms
#'
#' @method coef deepregression
#' @export
#' @rdname methodDR
#'
coef.deepregression <- function(
  object,
  variational = FALSE,
  params = NULL,
  type = NULL,
  ...
)
{
  nrparams <- length(object$init_params$parsed_formulae_contents)
  if(is.null(params)) params <- 1:nrparams
  layer_names <- sapply(object$model$layers, "[[", "name")
  lret <- vector("list", length(params))
  names(lret) <- object$init_params$param_names[params]
  if(is.null(type))
    type <- c("linear", "smooth")
  for(j in 1:length(params)){
    i = params[j]
    sl <- paste0("structured_linear_",i)
    slas <- paste0("structured_lasso_",i)
    snl <- paste0("structured_nonlinear_",i)
    tl <- paste0("tib_lasso_", i)
    lret[[j]] <- list(structured_linear = NULL,
                      structured_lasso = NULL,
                      structured_nonlinear = NULL
                      )

    lret[[j]]$structured_linear <-
      if(sl %in% layer_names)
        object$model$get_layer(sl)$get_weights()[[1]] else
          NULL
    if(slas %in% layer_names | tl %in% layer_names){
        if(slas %in% layer_names)
          lret[[j]]$structured_lasso <- object$model$get_layer(slas)$get_weights()[[1]] else
            lret[[j]]$structured_lasso <- 
              rbind(object$model$get_layer(tl)$get_weights()[[2]],
                    object$model$get_layer(tl)$get_weights()[[1]] *
                      matrix(object$model$get_layer(tl)$get_weights()[[3]], ncol=1))
    }else{
      lret[[j]]$structured_lasso <- NULL
    }
    if(snl %in% layer_names){
      cf <- object$model$get_layer(snl)$get_weights()
      if(length(cf)==2 & variational){
        lret[[j]]$structured_nonlinear <-  cf[[1]]
      }else{
        lret[[j]]$structured_nonlinear <- cf[[length(cf)]]
      }
    }else{
      lret[[j]]$structured_nonlinear <- NULL
    }

    sel <- which(c("linear", "smooth") %in% type)
    if(is.character(object$init_params$ind_structterms[[i]]$type)){
      object$init_params$ind_structterms[[i]]$type <- factor(
        object$init_params$ind_structterms[[i]]$type,
        levels = c("lin", "smooth")
      )
    }
    struct_terms_fitting_type <- 
      sapply(as.numeric(object$init_params$ind_structterms[[i]]$type),
             function(x) x%in%sel)
    length_names <- 
      (
        object$init_params$ind_structterms[[i]]$end - 
          object$init_params$ind_structterms[[i]]$start + 1
      )
    sel_ind <- rep(struct_terms_fitting_type, length_names)
    if(any(sapply(lret[[j]], NCOL)>1)){
      lret[[j]] <- lapply(lret[[j]], function(x) x[sel_ind,])
      lret[[j]]$linterms <- do.call("rbind", lret[[j]][
        c("structured_linear", "structured_lasso")])
      lret[[j]]$smoothterms <- lret[[j]]["structured_nonlinear"]
      lret[[j]] <- lret[[j]][c("linterms","smoothterms")[sel]]
      lret[[j]] <- lret[[j]][!sapply(lret[[j]],function(x) is.null(x) | is.null(x[[1]]))]
      lret[[j]] <- do.call("rbind", lret[[j]])
      rownames(lret[[j]]) <- rep(unlist(object$init_params$l_names_effects[[i]][
        c("linterms","smoothterms")]),
        length_names[struct_terms_fitting_type])
    }else if(length(lret[[j]])>0){
      lret[[j]] <- unlist(lret[[j]])
      lret[[j]] <- lret[[j]][sel_ind]
      names(lret[[j]]) <- rep(unlist(object$init_params$l_names_effects[[i]][
        c("linterms","smoothterms")[sel]]),
        length_names[struct_terms_fitting_type])
    }
    
  }
  return(lret)

}

#' Print function for deepregression model
#'
#' @export
#' @rdname methodDR
#' @param x a \code{deepregression} model
#' @param ... unused
#'
#' @method print deepregression
#'
print.deepregression <- function(
  x,
  ...
)
{
  print(x$model)
  fae <- x$init_params$list_of_formulae
  cat("Model formulae:\n---------------\n")
  invisible(sapply(1:length(fae), function(i){ cat(names(fae)[i],":\n"); print(fae[[i]])}))
}

#' @title Cross-validation for deepgression objects
#' @param ... further arguments passed to
#' \code{keras:::fit.keras.engine.training.Model}
#' @param x deepregression object
#' @param verbose whether to print training in each fold
#' @param patience number of patience for early stopping
#' @param plot whether to plot the resulting losses in each fold
#' @param print_folds whether to print the current fold
#' @param mylapply lapply function to be used; defaults to \code{lapply}
#' @param save_weights logical, whether to save weights in each epoch.
#' @param cv_folds see \code{deepregression}
#' @param stop_if_nan logical; whether to stop CV if NaN values occur
#' @param callbacks a list of callbacks used for fitting
#' @export
#' @rdname methodDR
#'
#' @return Returns an object \code{drCV}, a list, one list element for each fold
#' containing the model fit and the \code{weighthistory}.
#'
#'
#'
cv <- function(
  x,
  verbose = FALSE,
  patience = 20,
  plot = TRUE,
  print_folds = TRUE,
  cv_folds = NULL,
  stop_if_nan = TRUE,
  mylapply = lapply,
  save_weights = FALSE,
  callbacks = list(),
  ...
)
{

  if(is.null(cv_folds)){
    cv_folds <- x$init_params$cv_folds
  }else if(!is.list(cv_folds) & is.numeric(cv_folds)){
    cv_folds <- make_cv_list_simple(
      data_size = NROW(x$init_params$data[[1]]),
      cv_folds)
  }else{
    stop("Wrong format for cv_folds.")
  }
  if(is.null(cv_folds)){
    warning("No folds for CV given, using k = 10.\n")
    cv_folds <- make_cv_list_simple(
      data_size = NROW(x$init_params$data[[1]]), 10)
  }
  nrfolds <- length(cv_folds)
  old_weights <- x$model$get_weights()

  if(print_folds) folds_iter <- 1

  # subset fun
  if(NCOL(x$init_params$y)==1)
    subset_fun <- function(y,ind) y[ind] else
      subset_fun <- function(y,ind) subset_array(y,ind)

  res <- mylapply(cv_folds, function(this_fold){

    if(print_folds) cat("Fitting Fold ", folds_iter, " ... ")
    st1 <- Sys.time()

    # does not work?
    # this_mod <- clone_model(x$model)
    this_mod <- x$model

    train_ind <- this_fold[[1]]
    test_ind <- this_fold[[2]]

    # data
    if(is.data.frame(x$init_params$data)){
      train_data <- x$init_params$data[train_ind,, drop=FALSE]
      test_data <- x$init_params$data[test_ind,,drop=FALSE]
    }else if(class(x$init_params$data)=="list"){
      train_data <- lapply(x$init_params$data, function(x)
        subset_array(x, train_ind))
      test_data <- lapply(x$init_params$data, function(x)
        subset_array(x, test_ind))
    }else{
      stop("Invalid input format for CV.")
    }

    # make callbacks
    this_callbacks <- callbacks
    if(save_weights){
      weighthistory <- WeightHistory$new()
      this_callbacks <- append(this_callbacks, weighthistory)
    }

    args <- list(...)
    args <- append(args,
                   list(object = this_mod,
                        x = prepare_newdata(
                          x$init_params$parsed_formulae_contents,
                          train_data,
                          pred = FALSE,
                          orthogonalize = x$init_params$orthogonalize,
                          returnX = (x$init_params$orthog_type=="tf")
                          #index = train_ind,
                          #cv = TRUE
                          ),
                        y = subset_fun(x$init_params$y,train_ind),
                        validation_split = NULL,
                        validation_data = list(
                          prepare_newdata(
                            x$init_params$parsed_formulae_contents,
                            test_data,
                            pred = FALSE,
                            orthogonalize = x$init_params$orthogonalize,
                            returnX = (x$init_params$orthog_type=="tf")
                            # index = test_ind,
                            # cv = TRUE
                            ),
                          subset_fun(x$init_params$y,test_ind)
                        ),
                        callbacks = this_callbacks,
                        verbose = verbose,
                        view_metrics = FALSE
                   )
    )
    
    if(x$init_params$family=="transformation_model"){
      
      this_y <- subset_fun(x$init_params$y,train_ind)
      args$x <- c(args$x, 
                  list(x$init_params$y_basis_fun(this_y)),
                  list(x$init_params$y_basis_fun_prime(this_y)))
      if(!is.null(args$validation_data))
      {
        
        this_y <- subset_fun(x$init_params$y,test_ind)
        args$validation_data[[1]] <- 
          c(args$validation_data[[1]],
            list(x$init_params$y_basis_fun(this_y)),
            list(x$init_params$y_basis_fun_prime(this_y)))
        
      }
      
    }
    
    args <- append(args, x$init_params$ellipsis)

    ret <- do.call(fit_fun, args)
    if(save_weights) ret$weighthistory <- weighthistory$weights_last_layer

    if(stop_if_nan && any(is.nan(ret$metrics$validloss)))
      stop("Fold ", folds_iter, " with NaN's in ")

    if(print_folds) folds_iter <<- folds_iter + 1

    this_mod$set_weights(old_weights)
    td <- Sys.time()-st1
    if(print_folds) cat("\nDone in", as.numeric(td), "", attr(td,"units"), "\n")

    return(ret)

  })

  class(res) <- c("drCV","list")

  if(plot) try(plot_cv(res), silent = TRUE)

  x$model$set_weights(old_weights)

  invisible(return(res))

}

#' mean of model fit
#'
#' @export
#' @rdname methodDR
#'
#' @param x a deepregression model
#' @param data optional data, a data.frame or list
#' @param ... arguments passed to the predict function
#'
#' @method mean deepregression
#'
#'
mean.deepregression <- function(
  x,
  data = NULL,
  ...
)
{
  predict.deepregression(x, newdata = data, apply_fun = tfd_mean, ...)
}


#' Generic sd function
#'
#' @param x object
#' @param ... further arguments passed to the class-specific function
#'
#' @export
sd <- function (x, ...) {
  UseMethod("sd", x)
}

#' Standard deviation of fit distribution
#'
#' @param x a deepregression object
#' @param data either NULL or a new data set
#' @param ... arguments passed to the \code{predict} function
#'
#' @export
#' @rdname methodDR
#'
sd.deepregression <- function(
  x,
  data = NULL,
  ...
)
{
  predict.deepregression(x, newdata = data, apply_fun = tfd_stddev, ...)
}

#' Generic quantile function
#'
#' @param x object
#' @param ... further arguments passed to the class-specific function
#'
#' @export
quantile <- function (x, ...) {
  UseMethod("quantile", x)
}

#' Calculate the distribution quantiles
#'
#' @param x a deepregression object
#' @param data either \code{NULL} or a new data set
#' @param probs the quantile value(s)
#' @param ... arguments passed to the \code{predict} function
#'
#' @export
#' @rdname methodDR
#'
quantile.deepregression <- function(
  x,
  data = NULL,
  probs,
  ...
)
{
  predict.deepregression(x,
                         newdata = data,
                         apply_fun = function(x) tfd_quantile(x, value=probs),
                         ...)
}

#' Function to return the fitted distribution
#'
#' @param x the fitted deepregression object
#' @param data an optional data set
#' @param force_float forces conversion into float tensors
#'
#' @export
#'
get_distribution <- function(
  x,
  data = NULL,
  force_float = FALSE
)
{
  if(is.null(data)){
    disthat <- x$model(unname(x$init_params$input_cov))
  }else{
    # preprocess data
    if(is.data.frame(data)) data <- as.list(data)
    newdata_processed <- prepare_data(x, data, pred=TRUE)
    if(!is.null(attr(x$init_params$parsed_formulae_contents[[2]], "minval")))
      newdata_processed <- newdata_processed[[1]]
    if(force_float | !is.null(attr(newdata_processed, "ox"))) 
      newdata_processed <- lapply(newdata_processed, function(x) tf$cast(x, "float32"))
    disthat <- x$model(newdata_processed)
  }
  return(disthat)
}

#' Function to return the log_score
#'
#' @param x the fitted deepregression object
#' @param data an optional data set
#' @param this_y new y for optional data
#' @param ind_fun function indicating the dependency; per default (iid assumption)
#' \code{tfd_independent} is used.
#' @param convert_fun function that converts Tensor; per default \code{as.matrix}
#' @param summary_fun function summarizing the output; per default the identity
#'
#' @export
log_score <- function(
  x,
  data=NULL,
  this_y=NULL,
  ind_fun = function(x) tfd_independent(x,1),
  convert_fun = as.matrix,
  summary_fun = function(x) x
)
{
  is_trafo <- x$init_params$family=="transformation_model"

  if(is.null(data)){
    this_data <- unname(x$init_params$input_cov)
    this_data <- lapply(this_data, function(x) tf$cast(x, tf$float32))
    if(is_trafo)
      this_data <- list(this_data, tf$constant(matrix(x$init_params$y,
                                                      ncol=1),
                                               dtype = "float32"))
    if(length(x$init_params$image_var)>0)
    {
      if(is_trafo){
        preds <- x %>% predict(newdata = this_data)
        disthat <- preds(this_y, type="output")
      }else{
        disthat <- unlist(x$model %>% predict(newdata = this_data))
      }
    }else{
      disthat <- x$model(this_data)
    }
  }else{
    # preprocess data
    if(is.data.frame(data)) data <- as.list(data)
    newdata_processed <- prepare_data(x, data, pred=TRUE)
    if(is_trafo & length(x$init_params$image_var)==0){
      if(missing(this_y)) stop("Must provide this_y for transformation models and new data.")
      if(!is.null(attr(x$init_params$parsed_formulae_contents[[2]], "minval")))
        newdata_processed <- newdata_processed[[1]]
      newdata_processed <- list(unname(
        lapply(c(newdata_processed,
                 list(x$init_params$y_basis_fun(this_y)),
                 list(x$init_params$y_basis_fun_prime(this_y))),
               function(y) tf$cast(y, tf$float32))),
        tf$constant(matrix(this_y,
                           ncol=1),
                    dtype = "float32")
      )
    }else if(is_trafo & length(x$init_params$image_var)>0){
      newdata_processed <- data
    }
    if(length(x$init_params$image_var)>0)
    {
      if(is_trafo){
        preds <- x %>% predict(newdata = newdata_processed)
        disthat <- preds(this_y, type="output")
      }else{
        disthat <- unlist(x$model %>% predict(newdata = newdata_processed))
      }
    }else{
      disthat <- x$model(newdata_processed)
    }
  }

  if(is_trafo)
    return(summary_fun(
      convert_fun(
        calculate_log_score(x, disthat)
      ))
    )

  if(is.null(this_y)){
    this_y <- x$init_params$y
  }
  return(summary_fun(convert_fun(
    disthat %>% ind_fun() %>% tfd_log_prob(this_y)
  )))
}

#' Function to return the shift term
#'
#' @param x the fitted deeptrafo object
#'
#' @export
get_shift <- function(x)
{

  stopifnot("deeptrafo" %in% class(x))
  names_weights <- sapply(x$model$trainable_weights, function(x) x$name)
  lin_names <- grep("structured_linear_1", names_weights)
  nonlin_names <- grep("structured_nonlinear_1", names_weights)
  if(length(c(lin_names, nonlin_names))==0)
    stop("Not sure which layer to access for shift. ", 
         "Have you specified a structured shift predictor?")
  -1 * as.matrix(x$model$trainable_weights[[c(lin_names, nonlin_names)]] + 0)

}

#' Function to return the theta term
#'
#' @param x the fitted deeptrafo object
#'
#' @export
get_theta <- function(x)
{

  stopifnot("deeptrafo" %in% class(x))
  names_weights <- sapply(x$model$trainable_weights, function(x) x$name)
  reshape_softplus_cumsum(
    as.matrix(x$model$weights[[grep("constraint_mono_layer", names_weights)]] + 0),
    order_bsp_p1 = x$init_params$order_bsp + 1
  )

}

#' Function to return the minval term
#'
#' @param x the fitted deeptrafo object
#'
#' @details This value is only available if \code{addconst_interaction}
#' was specified in the model call.
#'
#' @export
get_minval <- function(x)
{
  stopifnot("deeptrafo" %in% class(x))
  attr(x$init_params$parsed_formulae_contents[[2]], "minval")

}

#' Function to set the weights of a deepregression object
#'
#' @param x deepregression object
#' @param weights a matrix with weights
#' @param param integer; for which parameter to set the weights
#' @param type character; for which type of layer to set the weights;
#' 
#' @export
#'
set_weights <- function(x, 
                        weights, 
                        param = NULL, 
                        type = c("linear", "nonlinear", "lasso", "ridge", "elasticnet"))
{
  
  type <- match.arg(type)
  name <- switch(type,
                 lasso = paste0("structured_lasso_", param),
                 ridge = paste0("structured_ridge_", param),
                 elasticnet = paste0("structured_elastnet_", param),
                 linear = paste0("structured_linear_", param),
                 nonlinear = paste0("structured_nonlinear_", param)
  )
                           

  x$model$get_layer(name)$set_weights(weights)
  
}

#' Return partial effect of one smooth term
#' 
#' @param object deepregression object
#' @param name string; for partial match with smooth term
#' @param return_matrix logical; whether to return the design matrix or
#' @param which_param integer; which distribution parameter
#' the partial effect (\code{FALSE}, default)
#' @param newdata data.frame; new data (optional)
#' 
#' @export
#' 
get_partial_effect <- function(object, name, return_matrix = FALSE, 
                               which_param = 1, newdata = NULL)
{
  
  if(is.null(newdata)) newdata <- object$init_params$data
  
  nms <- names(object$init_params$parsed_formulae_contents[[which_param]]$smoothterms)
  match <- grep(name, nms)
  if(all(!match)) stop("No matching name found.")
  
  sTerm <- object$init_params$parsed_formulae_contents[[which_param]]$smoothterms[[match]]
  if(is.list(sTerm)) sTerm <- sTerm[[1]]
  
  pmat <- PredictMat(sTerm, data = newdata)
  if(attr(object$init_params$parsed_formulae_contents[[which_param]],"zero_cons"))
    pmat <- orthog_structured_smooths(pmat,P=NULL,L=matrix(rep(1,nrow(pmat)),ncol=1))
  if(return_matrix) return(pmat)
  
  coefs <- coef(object, params = which_param, type = "smooth")[[1]]
  coefs <- coefs[grepl(name, names(coefs))]

  return(pmat%*%coefs)
  
}

