# repo contains one standalone script for the nanotabpfn model, priors generator, and model eval

# ===========================================================================
# nanoTabPFN - Pure R Implementation using torch for R
# ===========================================================================
# Translated from Python: https://github.com/automl/nanoTabPFN
#
# Key R vs Python differences handled:
#   1. Tensor indexing: R torch uses 1-based indexing (Python: 0-based)
#      - Python x[:, :k]           -> R x[, 1:k, ]
#      - Python x[:, k:]           -> R x[, (k+1):n, ]
#      - Python x[:, :, -1]        -> R x[, , ncols]
#   2. Dimension arguments: R torch uses 1-based dims
#      - Python dim=0 (batch)      -> R dim=1
#      - Python dim=1 (rows)       -> R dim=2
#      - Python dim=2 (cols)       -> R dim=3
#   3. CrossEntropyLoss: R torch expects 1-based class indices (1..C)
#      - Python targets: 0, 1, 2   -> R targets: 1, 2, 3
#   4. R reserved keywords: 'repeat' must be accessed via $`repeat`()
#   5. Generators: R uses 'coro' package with explicit generator functions
#      since 'yield' cannot be called inside R6 methods directly
# ===========================================================================

library(torch)
library(R6)
library(coro)

# ---------------------------------------------------------------------------
# FeatureEncoder
#   Normalizes features using train split mean/std, clips, embeds via Linear
# ---------------------------------------------------------------------------
FeatureEncoder <- nn_module(
  "FeatureEncoder",
  initialize = function(embedding_size) {
    self$linear_layer <- nn_linear(1, embedding_size)
  },
  forward = function(x, train_test_split_index) {
    # x: (batch_size, num_rows, num_features)
    x <- x$unsqueeze(-1)  # (batch, rows, features, 1)

    # R dim=2 corresponds to Python dim=1 (row dimension)
    mean_val <- torch_mean(x[, 1:train_test_split_index, , drop = FALSE],
                           dim = 2, keepdim = TRUE)
    std_val <- torch_std(x[, 1:train_test_split_index, , drop = FALSE],
                         dim = 2, keepdim = TRUE) + 1e-20
    x <- (x - mean_val) / std_val
    x <- torch_clamp(x, min = -100, max = 100)
    self$linear_layer(x)  # (batch, rows, features, embedding_size)
  }
)

# ---------------------------------------------------------------------------
# TargetEncoder
#   Pads y_test with train mean, embeds via Linear
# ---------------------------------------------------------------------------
TargetEncoder <- nn_module(
  "TargetEncoder",
  initialize = function(embedding_size) {
    self$linear_layer <- nn_linear(1, embedding_size)
  },
  forward = function(y_train, num_rows) {
    # y_train: (batch_size, num_train, 1)
    mean_val <- torch_mean(y_train, dim = 2, keepdim = TRUE)
    pad_len <- num_rows - y_train$shape[2]
    if (pad_len > 0) {
      padding <- mean_val$`repeat`(c(1, pad_len, 1))
      y <- torch_cat(list(y_train, padding), dim = 2)
    } else {
      y <- y_train
    }
    y <- y$unsqueeze(-1)
    self$linear_layer(y)
  }
)

# ---------------------------------------------------------------------------
# TransformerEncoderLayer
#   Bi-attention transformer: feature attention -> datapoint attention -> MLP
# ---------------------------------------------------------------------------
TransformerEncoderLayer <- nn_module(
  "TransformerEncoderLayer",
  initialize = function(embedding_size, nhead, mlp_hidden_size,
                        layer_norm_eps = 1e-5, batch_first = TRUE) {
    self$self_attention_between_datapoints <- nn_multihead_attention(
      embedding_size, nhead, batch_first = batch_first
    )
    self$self_attention_between_features <- nn_multihead_attention(
      embedding_size, nhead, batch_first = batch_first
    )
    self$linear1 <- nn_linear(embedding_size, mlp_hidden_size)
    self$linear2 <- nn_linear(mlp_hidden_size, embedding_size)
    self$norm1 <- nn_layer_norm(embedding_size, eps = layer_norm_eps)
    self$norm2 <- nn_layer_norm(embedding_size, eps = layer_norm_eps)
    self$norm3 <- nn_layer_norm(embedding_size, eps = layer_norm_eps)
  },
  forward = function(src, train_test_split_index) {
    # src: (batch, rows, cols, embedding)
    batch_size <- src$shape[1]
    rows_size  <- src$shape[2]
    col_size   <- src$shape[3]
    embedding_size <- src$shape[4]

    # ---- attention between features ----
    src <- src$reshape(c(batch_size * rows_size, col_size, embedding_size))
    attn_out <- self$self_attention_between_features(src, src, src)[[1]]
    src <- attn_out + src
    src <- src$reshape(c(batch_size, rows_size, col_size, embedding_size))
    src <- self$norm1(src)

    # ---- attention between datapoints ----
    src <- src$transpose(2, 3)  # (batch, cols, rows, embedding)
    src <- src$reshape(c(batch_size * col_size, rows_size, embedding_size))

    # Training rows attend to themselves
    src_train <- src[, 1:train_test_split_index, , drop = FALSE]
    src_left <- self$self_attention_between_datapoints(
      src_train, src_train, src_train
    )[[1]]

    # Test rows attend to training rows
    if (train_test_split_index < rows_size) {
      src_test <- src[, (train_test_split_index + 1):rows_size, , drop = FALSE]
      src_right <- self$self_attention_between_datapoints(
        src_test, src_train, src_train
      )[[1]]
      src <- torch_cat(list(src_left, src_right), dim = 2) + src
    } else {
      src <- src_left + src
    }

    src <- src$reshape(c(batch_size, col_size, rows_size, embedding_size))
    src <- src$transpose(2, 3)  # back to (batch, rows, cols, embedding)
    src <- self$norm2(src)

    # ---- MLP ----
    src <- self$linear2(nnf_gelu(self$linear1(src))) + src
    src <- self$norm3(src)
    src
  }
)

# ---------------------------------------------------------------------------
# Decoder: 2-layer MLP to logits
# ---------------------------------------------------------------------------
Decoder <- nn_module(
  "Decoder",
  initialize = function(embedding_size, mlp_hidden_size, num_outputs) {
    self$linear1 <- nn_linear(embedding_size, mlp_hidden_size)
    self$linear2 <- nn_linear(mlp_hidden_size, num_outputs)
  },
  forward = function(x) {
    self$linear2(nnf_gelu(self$linear1(x)))
  }
)

# ---------------------------------------------------------------------------
# NanoTabPFNModel
#   Full model: FeatureEncoder + TargetEncoder + NxTransformer + Decoder
# ---------------------------------------------------------------------------
NanoTabPFNModel <- nn_module(
  "NanoTabPFNModel",
  initialize = function(embedding_size, num_attention_heads,
                        mlp_hidden_size, num_layers, num_outputs) {
    self$feature_encoder <- FeatureEncoder(embedding_size)
    self$target_encoder  <- TargetEncoder(embedding_size)
    self$transformer_blocks <- list()
    for (i in seq_len(num_layers)) {
      block <- TransformerEncoderLayer(
        embedding_size, num_attention_heads, mlp_hidden_size
      )
      self$transformer_blocks[[i]] <- block
      # Register as named submodule so parameters are tracked
      self[[paste0("transformer_", i)]] <- block
    }
    self$decoder <- Decoder(embedding_size, mlp_hidden_size, num_outputs)
  },
  forward = function(src, train_test_split_index) {
    x_src <- src[[1]]
    y_src <- src[[2]]
    if (length(y_src$shape) < length(x_src$shape)) {
      y_src <- y_src$unsqueeze(-1)
    }
    x_src <- self$feature_encoder(x_src, train_test_split_index)
    num_rows <- x_src$shape[2]
    y_src <- self$target_encoder(y_src, num_rows)
    src <- torch_cat(list(x_src, y_src), dim = 3)
    for (block in self$transformer_blocks) {
      src <- block(src, train_test_split_index)
    }
    num_cols <- src$shape[3]
    if (train_test_split_index < num_rows) {
      # R: (train_test_split_index+1):num_rows  <=>  Python: train_test_split_index:
      output <- src[, (train_test_split_index + 1):num_rows, num_cols, , drop = FALSE]
    } else {
      output <- src[, integer(0), num_cols, , drop = FALSE]
    }
    output <- output$squeeze(3)
    self$decoder(output)
  }
)

# ---------------------------------------------------------------------------
# NanoTabPFNClassifier
#   sklearn-like interface with fit/predict_proba/predict
# ---------------------------------------------------------------------------
NanoTabPFNClassifier <- R6Class(
  "NanoTabPFNClassifier",
  public = list(
    model = NULL,
    device = NULL,
    X_train = NULL,
    y_train = NULL,
    num_classes = NULL,

    initialize = function(model, device = "cpu") {
      self$model <- model$to(device = device)
      self$device <- device
    },

    fit = function(X_train, y_train) {
      self$X_train <- X_train
      self$y_train <- y_train
      self$num_classes <- max(y_train) + 1
    },

    predict_proba = function(X_test) {
      x <- rbind(self$X_train, X_test)
      y <- self$y_train
      with_no_grad({
        x_t <- torch_tensor(x, dtype = torch_float())$unsqueeze(1)$to(device = self$device)
        y_t <- torch_tensor(y, dtype = torch_float())$unsqueeze(1)$to(device = self$device)
        out <- self$model(list(x_t, y_t), train_test_split_index = nrow(self$X_train))
        out <- out$squeeze(1)
        if (self$num_classes < out$shape[2]) {
          out <- out[, 1:self$num_classes, drop = FALSE]
        }
        probs <- nnf_softmax(out, dim = 2)
        as.array(probs$cpu())
      })
    },

    predict = function(X_test) {
      probs <- self$predict_proba(X_test)
      apply(probs, 1, which.max) - 1  # Convert 1-based back to 0-based
    }
  )
)

# ---------------------------------------------------------------------------
# Synthetic Prior Data Loader
#   Generates random classification datasets on-the-fly (no HDF5 required)
# ---------------------------------------------------------------------------
SyntheticPriorDataLoader <- R6Class(
  "SyntheticPriorDataLoader",
  public = list(
    num_steps = NULL,
    batch_size = NULL,
    device = NULL,
    max_num_classes = NULL,
    min_features = NULL,
    max_features = NULL,
    max_seq_len = NULL,

    initialize = function(num_steps = 2500, batch_size = 32,
                          device = "cpu", max_num_classes = 2,
                          min_features = 3, max_features = 5,
                          max_seq_len = 50) {
      self$num_steps <- num_steps
      self$batch_size <- batch_size
      self$device <- device
      self$max_num_classes <- max_num_classes
      self$min_features <- min_features
      self$max_features <- max_features
      self$max_seq_len <- max_seq_len
    },

    get_batch = function() {
      num_features <- sample(self$min_features:self$max_features, 1)
      num_classes  <- sample(2:self$max_num_classes, 1)
      num_total    <- sample(20:self$max_seq_len, 1)
      train_test_split_index <- max(1, as.integer(floor(num_total * 0.6)))

      X_array <- array(0, dim = c(self$batch_size, num_total, num_features))
      y_mat <- matrix(0L, nrow = self$batch_size, ncol = num_total)

      for (b in seq_len(self$batch_size)) {
        means <- matrix(rnorm(num_classes * num_features, sd = 2),
                        nrow = num_classes, ncol = num_features)
        for (i in seq_len(num_total)) {
          cls <- sample(1:num_classes, 1)
          X_array[b, i, ] <- rnorm(num_features, mean = means[cls, ], sd = 1)
          y_mat[b, i] <- as.integer(cls - 1)  # 0-based class labels
        }
      }

      list(
        x = torch_tensor(X_array, dtype = torch_float())$to(device = self$device),
        y = torch_tensor(y_mat, dtype = torch_long())$to(device = self$device),
        train_test_split_index = train_test_split_index
      )
    }
  )
)

# Generator wrapper: coro::yield cannot be called inside R6 methods directly
make_synthetic_iterator <- function(loader) {
  coro::generator(function() {
    for (step in seq_len(loader$num_steps)) {
      yield(loader$get_batch())
    }
  })()
}

# ---------------------------------------------------------------------------
# HDF5 Prior Dump Data Loader
#   Loads pre-generated prior datasets from an HDF5 file
#   Requires: install.packages('hdf5r')
# ---------------------------------------------------------------------------
PriorDumpDataLoader <- R6Class(
  "PriorDumpDataLoader",
  public = list(
    filename = NULL,
    num_steps = NULL,
    batch_size = NULL,
    device = NULL,
    pointer = 0,
    max_num_classes = NULL,
    n_total = NULL,

    initialize = function(filename, num_steps, batch_size, device = "cpu") {
      self$filename <- filename
      self$num_steps <- num_steps
      self$batch_size <- batch_size
      self$device <- device
      self$pointer <- 0
      if (!requireNamespace("hdf5r", quietly = TRUE)) {
        stop("Package 'hdf5r' is required. Install: install.packages('hdf5r')")
      }
      f <- hdf5r::H5File$new(filename, mode = "r")
      self$max_num_classes <- as.integer(f[["max_num_classes"]][])
      self$n_total <- f[["X"]]$dims[1]
      f$close_all()
    },

    get_batch_at = function(pos, batch_size) {
      if (!file.exists(self$filename)) {
        stop(paste0("HDF5 file not found: ", self$filename))
      }
      f <- hdf5r::H5File$new(self$filename, mode = "r")
      on.exit(f$close_all(), add = TRUE)

      X_ds   <- f[["X"]]
      y_ds   <- f[["y"]]
      nf_ds  <- f[["num_features"]]
      nd_ds  <- f[["num_datapoints"]]
      pos_ds <- f[["single_eval_pos"]]

      end <- min(pos + batch_size, self$n_total)
      actual_batch <- end - pos
      if (actual_batch <= 0) return(NULL)

      indices <- (pos + 1):end  # R uses 1-based indexing

      num_features <- as.integer(max(nf_ds[indices]))
      num_datapoints_batch <- as.integer(nd_ds[indices])
      max_seq_in_batch <- as.integer(max(num_datapoints_batch))

      x <- torch_tensor(
        X_ds[indices, 1:max_seq_in_batch, 1:num_features, drop = FALSE],
        dtype = torch_float()
      )$to(device = self$device)
      y <- torch_tensor(
        y_ds[indices, 1:max_seq_in_batch, drop = FALSE],
        dtype = torch_long()
      )$to(device = self$device)
      train_test_split_index <- as.integer(pos_ds[1])

      list(x = x, y = y, train_test_split_index = train_test_split_index)
    }
  )
)

make_hdf5_iterator <- function(loader) {
  coro::generator(function() {
    for (step in seq_len(loader$num_steps)) {
      end <- min(loader$pointer + loader$batch_size, loader$n_total)
      actual_batch <- end - loader$pointer
      if (actual_batch <= 0) {
        message("Finished iteration! Resetting pointer.")
        loader$pointer <- 0
        end <- min(loader$pointer + loader$batch_size, loader$n_total)
        actual_batch <- end - loader$pointer
      }
      batch <- loader$get_batch_at(loader$pointer, loader$batch_size)
      if (is.null(batch)) {
        loader$pointer <- 0
        batch <- loader$get_batch_at(0, loader$batch_size)
      }
      loader$pointer <- loader$pointer + loader$batch_size
      if (loader$pointer >= loader$n_total) {
        message("Finished iteration! Resetting pointer.")
        loader$pointer <- 0
      }
      yield(batch)
    }
  })()
}

# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

train_nanotabpfn <- function(model, prior, lr = 1e-4, device = "cpu",
                             num_steps = NULL, steps_per_eval = 25,
                             eval_func = NULL) {
  model$to(device = device)
  optimizer <- optim_adamw(model$parameters, lr = lr, weight_decay = 0.0)
  criterion <- nn_cross_entropy_loss()
  model$train()

  train_time <- 0
  eval_history <- list()
  losses <- c()

  step <- 0
  tryCatch({
    coro::loop(for (full_data in prior) {
      step_start_time <- Sys.time()
      train_test_split_index <- full_data$train_test_split_index
      if (train_test_split_index < 1) train_test_split_index <- 1

      # Model expects float targets; ensure correct dtype
      y_train <- full_data$y[, 1:train_test_split_index, drop = FALSE]$to(
        device = device, dtype = torch_float()
      )
      data <- list(full_data$x$to(device = device), y_train)
      targets <- full_data$y$to(device = device)

      output <- model(data, train_test_split_index = train_test_split_index)

      n_rows <- targets$shape[2]
      if (train_test_split_index >= n_rows) {
        step <- step + 1
        if (!is.null(num_steps) && step >= num_steps) break
        next
      }

      targets <- targets[, (train_test_split_index + 1):n_rows]
      targets <- targets$reshape(list(-1))
      output  <- output$view(c(-1, output$shape[3]))

      # CRITICAL R vs PYTHON DIFFERENCE:
      # R torch's nn_cross_entropy_loss expects 1-based Long targets (1..C)
      # whereas Python PyTorch uses 0-based (0..C-1).
      targets <- targets$to(dtype = torch_long()) + 1L
      num_outputs <- output$shape[2]
      targets <- torch_clamp(targets, min = 1L, max = as.integer(num_outputs))

      loss <- criterion(output, targets)
      total_loss <- as.numeric(loss$cpu()$detach())

      optimizer$zero_grad()
      loss$backward()
      nn_utils_clip_grad_norm_(model$parameters, 1.0)
      optimizer$step()

      step_train_duration <- as.numeric(
        difftime(Sys.time(), step_start_time, units = "secs")
      )
      train_time <- train_time + step_train_duration
      losses <- c(losses, total_loss)
      step <- step + 1

      if (!is.null(num_steps) && step >= num_steps) break

      if (step %% steps_per_eval == 0) {
        recent <- losses[max(1, length(losses) - steps_per_eval + 1):length(losses)]
        avg_loss <- mean(recent)
        if (!is.null(eval_func)) {
          model$eval()
          classifier <- NanoTabPFNClassifier$new(model, device)
          scores <- eval_func(classifier)
          eval_history[[length(eval_history) + 1]] <- list(
            time = train_time, step = step, loss = avg_loss, scores = scores
          )
          score_str <- paste(names(scores), sprintf("%.4f", unlist(scores)),
                             sep = " ", collapse = " | ")
          cat(sprintf("step %4d | time %6.1fs | loss %.4f | %s\n",
                      step, train_time, avg_loss, score_str))
          model$train()
        } else {
          cat(sprintf("step %4d | time %6.1fs | loss %.4f\n",
                      step, train_time, avg_loss))
        }
      }
    })
  }, interrupt = function(e) {
    message("Training interrupted by user.")
  })

  list(model = model, history = eval_history, losses = losses)
}

# ===========================================================================
# MAIN: Pretrain and evaluate on iris
# ===========================================================================

set.seed(0)
torch_manual_seed(0)

cat("\n========================================\n")
cat("       nanoTabPFN in R (torch)\n")
cat("========================================\n\n")

device <- if (cuda_is_available()) "cuda" else "cpu"
cat("Device:", device, "\n")

model <- NanoTabPFNModel(
  embedding_size       = 96,
  num_attention_heads  = 4,
  mlp_hidden_size      = 192,
  num_layers           = 3,
  num_outputs          = 2
)

cat("Parameters:", sum(sapply(model$parameters, function(p) prod(p$shape))), "\n")

# Synthetic prior dataloader (on-the-fly, no HDF5 file needed)
prior_loader <- SyntheticPriorDataLoader$new(
  num_steps      = 5000,
  batch_size     = 32,
  device         = device,
  max_num_classes = 2,
  min_features   = 3,
  max_features   = 5,
  max_seq_len    = 60
)
prior_iter <- make_synthetic_iterator(prior_loader)

cat("\n--- Pretraining ---\n")
result <- train_nanotabpfn(
  model    = model,
  prior    = prior_iter,
  lr       = 1e-3,
  device   = device,
  num_steps = 1500,
  steps_per_eval = 100
)

losses <- result$losses
cat("\n--- Pretraining Summary ---\n")
cat("Steps:    ", length(losses), "\n")
cat("Initial:  ", sprintf("%.4f", losses[1]), "\n")
cat("Final:    ", sprintf("%.4f", tail(losses, 1)), "\n")
cat("Best:     ", sprintf("%.4f", min(losses)), "\n")
cat("Improved: ", tail(losses, 1) < losses[1], "\n")

# ---------------------------------------------------------------------------
# Iris binary classification (setosa vs versicolor)
# ---------------------------------------------------------------------------
cat("\n--- Iris Classification Test ---\n")

X <- as.matrix(datasets::iris[, 1:4])
X <- scale(X)
y <- as.integer(datasets::iris$Species) - 1  # 0, 1, 2

# Binary: classes 0 and 1 only
idx <- y <= 1
X <- X[idx, ]
y <- y[idx]

n <- nrow(X)
train_idx <- sample(seq_len(n), size = floor(n / 2))
test_idx <- setdiff(seq_len(n), train_idx)

X_train <- X[train_idx, ]
y_train <- y[train_idx]
X_test  <- X[test_idx, ]
y_test  <- y[test_idx]

clf <- NanoTabPFNClassifier$new(result$model, device)
clf$fit(X_train, y_train)

prob <- clf$predict_proba(X_test)
pred <- clf$predict(X_test)
acc  <- mean(pred == y_test)

cat("Accuracy:", sprintf("%.2f%%", acc * 100), "\n")
cat("Predicted:", pred, "\n")
cat("Actual:   ", y_test, "\n")

# ---------------------------------------------------------------------------
# Sanity check on fresh synthetic data
# ---------------------------------------------------------------------------
cat("\n--- Synthetic Data Sanity Check ---\n")
synth <- SyntheticPriorDataLoader$new(num_steps = 1, batch_size = 1, device = device)
s_iter <- make_synthetic_iterator(synth)
b <- s_iter()
n_tot <- b$x$shape[2]
trn   <- as.integer(floor(n_tot * 0.6))
X_s   <- as.array(b$x$cpu()[1, , ])
y_s   <- as.array(b$y$cpu()[1, ])
clf2  <- NanoTabPFNClassifier$new(result$model, device)
clf2$fit(X_s[1:trn, ], y_s[1:trn])
p_s   <- clf2$predict(X_s[(trn + 1):n_tot, ])
cat("Accuracy:", sprintf("%.2f%%", mean(p_s == y_s[(trn + 1):n_tot]) * 100), "\n")

cat("\nDone!\n")
