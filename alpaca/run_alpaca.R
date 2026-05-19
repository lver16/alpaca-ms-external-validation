cores <- 1
Sys.setenv(
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = cores,
  ANTS_NUM_THREADS = cores
)
Sys.setenv(
  OMP_NUM_THREADS        = cores,
  OPENBLAS_NUM_THREADS   = cores,
  MKL_NUM_THREADS        = cores,
  VECLIB_MAXIMUM_THREADS = cores,
  NUMEXPR_NUM_THREADS    = cores
)

library(ALPaCA)
message("make_predictions loaded from: ", getAnywhere("make_predictions")$where[1])
message("Package path: ", system.file(package = "ALPaCA"))
library(ANTsR)
library(ANTsRCore)

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("Provide subject list file")
sublist_file    <- args[1]
subjects_to_run <- readLines(sublist_file)

input_root <- "/linux/luverheyen/data/dataset_renamed"
out_root   <- "/linux/luverheyen/data/alpaca_out"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

subjects <- file.path(input_root, subjects_to_run)

for (subdir in subjects) {
  sub_id <- basename(subdir)
  message("\n=== Processing ", sub_id, " ===")

  t1    <- file.path(subdir, "T1.nii.gz")
  flair <- file.path(subdir, "FLAIR.nii.gz")
  epi   <- file.path(subdir, "EPIm.nii.gz")
  phase <- file.path(subdir, "EPIp.nii.gz")

  missing <- c()
  if (!file.exists(t1))    missing <- c(missing, "T1")
  if (!file.exists(flair)) missing <- c(missing, "FLAIR")
  if (!file.exists(epi))   missing <- c(missing, "EPIm")
  if (!file.exists(phase)) missing <- c(missing, "EPIp")

  if (length(missing) > 0) {
    message("Skipping ", sub_id, " (missing: ", paste(missing, collapse = ", "), ")")
    next
  }

  sub_out <- file.path(out_root, sub_id)
  dir.create(sub_out, recursive = TRUE, showWarnings = FALSE)

  # 1) Preprocess
  ants_list <- ALPaCA::preprocess_images(
    t1_path       = t1,
    flair_path    = flair,
    epi_path      = epi,
    phase_path    = phase,
    output_dir    = sub_out,
    reorient      = TRUE,
    cores         = cores,
    verbose       = TRUE,
    return_images = TRUE
  )

  # 2) Chunked predictions
  orig_labeled <- as.array(ants_list$labeled_candidates)
  orig_eroded  <- as.array(ants_list$eroded_candidates)

  n_lesions <- max(orig_labeled)
  message("Total candidate lesions: ", n_lesions)

  if (n_lesions == 0) {
    message("No lesion candidates found for ", sub_id, ", skipping.")
    next
  }

  chunk_size <- 50
  chunks <- split(
    seq_len(n_lesions),
    ceiling(seq_len(n_lesions) / chunk_size)
  )
  message("Processing in ", length(chunks), " chunk(s) of max ", chunk_size)

  all_predictions   <- vector("list", length(chunks))
  all_probabilities <- vector("list", length(chunks))
  all_uncertainties <- vector("list", length(chunks))

  for (i in seq_along(chunks)) {
    chunk_labels <- chunks[[i]]
    message("  Chunk ", i, "/", length(chunks),
            " - lesions ", min(chunk_labels), " to ", max(chunk_labels))

    chunk_labeled <- orig_labeled * 0L
    for (new_label in seq_along(chunk_labels)) {
      chunk_labeled[orig_labeled == chunk_labels[new_label]] <- new_label
    }

    chunk_eroded <- orig_eroded * 0L
    for (new_label in seq_along(chunk_labels)) {
      chunk_eroded[orig_eroded == chunk_labels[new_label]] <- new_label
    }

    sub_ants                    <- ants_list
    sub_ants$labeled_candidates <- as.antsImage(chunk_labeled,
                                     reference = ants_list$labeled_candidates)
    sub_ants$eroded_candidates  <- as.antsImage(chunk_eroded,
                                     reference = ants_list$eroded_candidates)

    preds_i <- ALPaCA::make_predictions(
      ants_list      = sub_ants,
      output_dir     = sub_out,
      n_patches      = 20,
      n_models       = 10,
      rotate_patches = TRUE,
      verbose        = TRUE
    )

    # Save chunk mask before next chunk overwrites alpaca_mask.nii.gz on disk
    antsImageWrite(
      preds_i$alpaca_mask,
      file.path(sub_out, paste0("alpaca_mask_chunk_", i, ".nii.gz"))
    )

    # Store results in memory — no lesion_id column, same format as original
    all_predictions[[i]]   <- preds_i$predictions
    all_probabilities[[i]] <- preds_i$probabilities
    all_uncertainties[[i]] <- preds_i$prediction_uncertainties

    rm(sub_ants, chunk_labeled, chunk_eroded, preds_i)
    gc()
  }

  # -- Merge CSVs — identical format to original single-run output ------------
  final_predictions   <- do.call(rbind, all_predictions)
  final_probabilities <- do.call(rbind, all_probabilities)
  final_uncertainties <- do.call(rbind, all_uncertainties)

  # Reset row names to 1..N, exactly as a single make_predictions call would produce
  rownames(final_predictions)   <- NULL
  rownames(final_probabilities) <- NULL
  rownames(final_uncertainties) <- NULL

  write.csv(final_predictions,
            file.path(sub_out, "predictions.csv"),              row.names = TRUE)
  write.csv(final_probabilities,
            file.path(sub_out, "probabilities.csv"),            row.names = TRUE)
  write.csv(final_uncertainties,
            file.path(sub_out, "prediction_uncertainties.csv"), row.names = TRUE)

  # -- Merge alpaca_mask -------------------------------------------------------
  full_mask_arr <- orig_labeled * 0L

  for (i in seq_along(chunks)) {
    chunk_mask_arr <- as.array(
      antsImageRead(file.path(sub_out, paste0("alpaca_mask_chunk_", i, ".nii.gz")))
    )
    full_mask_arr[chunk_mask_arr > 0] <- chunk_mask_arr[chunk_mask_arr > 0]
    file.remove(file.path(sub_out, paste0("alpaca_mask_chunk_", i, ".nii.gz")))
  }

  antsImageWrite(
    as.antsImage(full_mask_arr, reference = ants_list$labeled_candidates),
    file.path(sub_out, "alpaca_mask.nii.gz")
  )

  message("Done: ", sub_id)
}

message("\nAll subjects processed.")