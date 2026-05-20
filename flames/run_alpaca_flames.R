# =============================================================================
# run_flames_alpaca.R
# -----------------------------------------------------------------------------
# Description:
#   Runs the full FLAMES + ALPaCA pipeline on a list of subjects. This script
#   bypasses the MIMoSA lesion candidate generation step used in the original
#   ALPaCA pipeline and instead uses a pre-computed FLAMES probability map to
#   generate lesion candidates. For each subject, the script:
#     1. Preprocesses the MRI images using FLAMES probability map as candidate
#        generator (preprocess_images_flames)
#     2. Runs ALPaCA predictions in chunks to manage memory
#     3. Merges chunk results and saves final outputs to the output directory
#
# Usage:
#   Rscript run_flames_alpaca.R <sublist_file> <input_root> <out_root> \
#           <flames_root> <brainmask_root> <alpaca_src> \
#           [flames_threshold] [chunk_size]
#
# Arguments:
#   sublist_file      Path to a text file listing subject IDs (one per line)
#   input_root        Path to the root directory containing subject folders
#   out_root          Path to the root directory where outputs will be saved
#   flames_root       Path to the directory containing FLAMES probability maps
#   brainmask_root    Path to the directory containing brain masks
#   alpaca_src        Path to the ALPaCA source directory (ALPaCA/R/)
#   flames_threshold  (Optional) Probability threshold for FLAMES map binarization (default: 0.5)
#   chunk_size        (Optional) Number of lesions to process per chunk (default: 50)
#
# Input folder structure expected per subject:
#   <input_root>/<subject_id>/T1.nii.gz
#   <input_root>/<subject_id>/FLAIR.nii.gz
#   <input_root>/<subject_id>/EPIm.nii.gz
#   <input_root>/<subject_id>/EPIp.nii.gz
#
# Additional input files expected per subject:
#   <flames_root>/<subject_id>_ses-01_pred-prob.nii.gz   # FLAMES probability map
#   <brainmask_root>/<subject_id>_brainmask.nii.gz       # Brain mask
#
# Outputs per subject (saved in <out_root>/<subject_id>/):
#   - alpaca_mask.nii.gz           : final merged lesion mask
#   - predictions.csv              : binary predictions (Lesion, PRL, CVS)
#   - probabilities.csv            : raw predicted probabilities
#   - prediction_uncertainties.csv : standard deviation across models and patches
#   - prob.nii.gz                  : FLAMES probability map in EPI space
#   - labeled_candidates.nii.gz    : labeled lesion candidates from FLAMES
#   - eroded_candidates.nii.gz     : eroded lesion candidates
#   - Preprocessed images (T1, FLAIR, EPI, phase)
# =============================================================================

# -----------------------------------------------------------------------------
# Thread control — limit to single core to avoid conflicts with ANTs/ITK
# -----------------------------------------------------------------------------
cores <- 1
Sys.setenv(
  ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS = cores,
  ANTS_NUM_THREADS                     = cores,
  OMP_NUM_THREADS                      = cores,
  OPENBLAS_NUM_THREADS                 = cores,
  MKL_NUM_THREADS                      = cores,
  VECLIB_MAXIMUM_THREADS               = cores,
  NUMEXPR_NUM_THREADS                  = cores
)

# -----------------------------------------------------------------------------
# Load libraries
# -----------------------------------------------------------------------------
library(ALPaCA)
library(ANTsR)
library(ANTsRCore)
library(extrantsr)
library(oro.nifti)
library(neurobase)
library(fslr)
library(WhiteStripe)

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 6) {
  stop("Usage: Rscript run_flames_alpaca.R <sublist_file> <input_root> <out_root> ",
       "<flames_root> <brainmask_root> <alpaca_src> [flames_threshold] [chunk_size]")
}

sublist_file     <- args[1]
input_root       <- args[2]
out_root         <- args[3]
flames_root      <- args[4]
brainmask_root   <- args[5]
alpaca_src       <- args[6]
flames_threshold <- ifelse(length(args) >= 7, as.numeric(args[7]), 0.5)
chunk_size       <- ifelse(length(args) >= 8, as.integer(args[8]), 50)

# -----------------------------------------------------------------------------
# Source ALPaCA internal functions and FLAMES preprocessing adaptation
# -----------------------------------------------------------------------------
source(file.path(alpaca_src, "get_lesion_labels.R"))
source(file.path(alpaca_src, "split_confluent.R"))
source(file.path(alpaca_src, "lesion_centers.R"))
source(file.path(alpaca_src, "label_lesion.R"))
source(file.path(alpaca_src, "extract_patch.R"))
source(file.path(alpaca_src, "frangi.R"))
source(file.path(alpaca_src, "gradient3D.R"))
source(file.path(alpaca_src, "hessian3D.R"))
source(file.path(alpaca_src, "make_predictions.R"))
source(file.path(alpaca_src, "rotate_patch.R"))
source(file.path(alpaca_src, "preprocess_images_flames.R"))

# -----------------------------------------------------------------------------
# Read subject list and set up output directory
# -----------------------------------------------------------------------------
subjects_to_run <- readLines(sublist_file)
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
subjects <- file.path(input_root, subjects_to_run)

# -----------------------------------------------------------------------------
# Main loop — process each subject
# -----------------------------------------------------------------------------
for (subdir in subjects) {
  sub_id <- basename(subdir)
  message("\n=== Processing ", sub_id, " ===")

  # ---------------------------------------------------------------------------
  # Check that all required input files exist
  # ---------------------------------------------------------------------------
  t1             <- file.path(subdir, "T1.nii.gz")
  flair          <- file.path(subdir, "FLAIR.nii.gz")
  epi            <- file.path(subdir, "EPIm.nii.gz")
  phase          <- file.path(subdir, "EPIp.nii.gz")
  flames_file    <- file.path(flames_root,    paste0(sub_id, "_ses-01_pred-prob.nii.gz"))
  brainmask_file <- file.path(brainmask_root, paste0(sub_id, "_brainmask.nii.gz"))

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

  # Create subject output directory
  sub_out <- file.path(out_root, sub_id)
  dir.create(sub_out, recursive = TRUE, showWarnings = FALSE)

  # ---------------------------------------------------------------------------
  # Step 1: Preprocess images using FLAMES probability map
  # ---------------------------------------------------------------------------
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
    flames_threshold = flames_threshold,
    mincluster       = 30
  )

  # ---------------------------------------------------------------------------
  # Step 2: Split lesion candidates into chunks to manage memory
  # ---------------------------------------------------------------------------
  orig_labeled <- as.array(ants_list$labeled_candidates)
  orig_eroded  <- as.array(ants_list$eroded_candidates)

  n_lesions <- max(orig_labeled)
  message("Total candidate lesions: ", n_lesions)

  if (n_lesions == 0) {
    message("No lesion candidates found for ", sub_id, ", skipping.")
    next
  }

  chunks <- split(
    seq_len(n_lesions),
    ceiling(seq_len(n_lesions) / chunk_size)
  )
  message("Processing in ", length(chunks), " chunk(s) of max ", chunk_size)

  # Pre-allocate lists to collect results across chunks
  all_predictions   <- vector("list", length(chunks))
  all_probabilities <- vector("list", length(chunks))
  all_uncertainties <- vector("list", length(chunks))

  # ---------------------------------------------------------------------------
  # Step 3: Run ALPaCA predictions chunk by chunk
  # ---------------------------------------------------------------------------
  for (i in seq_along(chunks)) {
    chunk_labels <- chunks[[i]]
    message("  Chunk ", i, "/", length(chunks),
            " - lesions ", min(chunk_labels), " to ", max(chunk_labels))

    # Relabel candidates for this chunk (1 to N within chunk)
    chunk_labeled <- orig_labeled * 0L
    for (new_label in seq_along(chunk_labels)) {
      chunk_labeled[orig_labeled == chunk_labels[new_label]] <- new_label
    }

    chunk_eroded <- orig_eroded * 0L
    for (new_label in seq_along(chunk_labels)) {
      chunk_eroded[orig_eroded == chunk_labels[new_label]] <- new_label
    }

    # Replace candidates in ants_list with chunk-specific candidates
    sub_ants                    <- ants_list
    sub_ants$labeled_candidates <- as.antsImage(chunk_labeled,
                                     reference = ants_list$labeled_candidates)
    sub_ants$eroded_candidates  <- as.antsImage(chunk_eroded,
                                     reference = ants_list$eroded_candidates)

    # Run predictions for this chunk
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

    # Store predictions, probabilities and uncertainties for this chunk
    all_predictions[[i]]   <- preds_i$predictions
    all_probabilities[[i]] <- preds_i$probabilities
    all_uncertainties[[i]] <- preds_i$prediction_uncertainties

    # Free memory before next chunk
    rm(sub_ants, chunk_labeled, chunk_eroded, preds_i)
    gc()
  }

  # ---------------------------------------------------------------------------
  # Step 4: Merge chunk results and save final outputs
  # ---------------------------------------------------------------------------

  # Merge prediction CSVs — identical format to a single make_predictions call
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

  # Merge chunk masks into a single final alpaca_mask
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