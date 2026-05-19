# =========================================================
# Main script for ALPaCA with FLAMES
# =========================================================
cores <- 1
Sys.setenv(
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = cores,
  ANTS_NUM_THREADS = cores
)
Sys.setenv(
  OMP_NUM_THREADS = cores,
  OPENBLAS_NUM_THREADS = cores,
  MKL_NUM_THREADS = cores,
  VECLIB_MAXIMUM_THREADS = cores,
  NUMEXPR_NUM_THREADS = cores
)
library(ALPaCA)
library(ANTsR)
library(ANTsRCore)
library(extrantsr)
library(oro.nifti)
library(neurobase)
library(fslr)
library(WhiteStripe)
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/get_lesion_labels.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/split_confluent.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/lesion_centers.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/label_lesion.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/extract_patch.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/frangi.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/gradient3D.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/hessian3D.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/make_predictions.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/R/rotate_patch.R")
source("/linux/luverheyen/code/ALPACA/ALPaCA/preprocess_images_flames.R")
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) stop("Provide subject list file")
sublist_file <- args[1]
subjects_to_run <- readLines(sublist_file)
input_root     <- "/linux/luverheyen/data/dataset_renamed"
out_root       <- "/linux/luverheyen/data/alpaca_out_flames_05_thresh"
flames_root    <- "/linux/luverheyen/data/flames_acls"
brainmask_root <- "/linux/luverheyen/data/synthstrip_raw"
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
subjects <- file.path(input_root, subjects_to_run)
for (subdir in subjects) {
  sub_id <- basename(subdir)
  message("\n=== Processing ", sub_id, " ===")
  t1    <- file.path(subdir, "T1.nii.gz")
  flair <- file.path(subdir, "FLAIR.nii.gz")
  epi   <- file.path(subdir, "EPIm.nii.gz")
  phase <- file.path(subdir, "EPIp.nii.gz")
  flames_file     <- file.path(flames_root, paste0(sub_id, "_ses-01_pred-prob.nii.gz"))
  brainmask_file  <- file.path(brainmask_root, paste0(sub_id, "_brainmask.nii.gz"))
  missing <- c()
  if (!file.exists(t1))             missing <- c(missing, "T1")
  if (!file.exists(flair))          missing <- c(missing, "FLAIR")
  if (!file.exists(epi))            missing <- c(missing, "EPIm")
  if (!file.exists(phase))          missing <- c(missing, "EPIp")
  if (!file.exists(flames_file))    missing <- c(missing, "FLAMES")
  if (!file.exists(brainmask_file)) missing <- c(missing, "brainmask")
  if (length(missing) > 0) {
    message("Skipping ", sub_id, " (missing: ", paste(missing, collapse = ", "), ")")
    next
  }
  sub_out <- file.path(out_root, sub_id)
  dir.create(sub_out, recursive = TRUE, showWarnings = FALSE)
  ants_list <- preprocess_images_flames(
    t1_path          = t1,
    flair_path       = flair,
    epi_path         = epi,
    phase_path       = phase,
    flames_path      = flames_file,
    output_dir       = sub_out,
    brainmask_path   = brainmask_file,
    reorient         = TRUE,
    cores            = cores,
    verbose          = TRUE,
    return_images    = TRUE,
    flames_threshold = 0.5,
    mincluster       = 30
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

    antsImageWrite(
      preds_i$alpaca_mask,
      file.path(sub_out, paste0("alpaca_mask_chunk_", i, ".nii.gz"))
    )

    all_predictions[[i]]   <- preds_i$predictions
    all_probabilities[[i]] <- preds_i$probabilities
    all_uncertainties[[i]] <- preds_i$prediction_uncertainties

    rm(sub_ants, chunk_labeled, chunk_eroded, preds_i)
    gc()
  }

  # -- Merge CSVs
  final_predictions   <- do.call(rbind, all_predictions)
  final_probabilities <- do.call(rbind, all_probabilities)
  final_uncertainties <- do.call(rbind, all_uncertainties)

  rownames(final_predictions)   <- NULL
  rownames(final_probabilities) <- NULL
  rownames(final_uncertainties) <- NULL

  write.csv(final_predictions,
            file.path(sub_out, "predictions.csv"),              row.names = TRUE)
  write.csv(final_probabilities,
            file.path(sub_out, "probabilities.csv"),            row.names = TRUE)
  write.csv(final_uncertainties,
            file.path(sub_out, "prediction_uncertainties.csv"), row.names = TRUE)

  # -- Merge alpaca_mask
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