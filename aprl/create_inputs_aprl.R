# =============================================================================
# run_mimosa_preprocessing.R
# -----------------------------------------------------------------------------
# Description:
#   Preprocesses multi-modal MRI images and generates lesion candidates using
#   MIMoSA for use with the ALPaCA pipeline. For each subject, the script:
#     1. Applies N4 bias correction to all modalities
#     2. Registers T1 and FLAIR to EPI space
#     3. Applies WhiteStripe intensity normalization
#     4. Runs MIMoSA to generate a lesion probability map
#     5. Builds binary truth masks from raw instance labels
#     6. Generates labeled lesion candidates from the MIMoSA probability map
#
# Usage:
#   Rscript run_mimosa_preprocessing.R <sublist_file> <dataset_root> \
#           <brainmask_root> <labels_root> <out_root> <mimosa_model_file>
#
# Arguments:
#   sublist_file       Path to a text file listing subject IDs, one per line
#                      (e.g. sub-001, sub-003). Session suffix _ses-01 is added
#                      automatically.
#   dataset_root       Path to the root directory containing subject folders
#   brainmask_root     Path to the directory containing brain masks
#   labels_root        Path to the directory containing raw instance label masks
#   out_root           Path to the root directory where outputs will be saved
#   mimosa_model_file  Path to the MIMoSA model .RData file
#
# Note:
#   The MIMoSA model file (.RData) must be downloaded separately from the
#   MIMoSA project. See the README for instructions.
#   By default the script reprocesses all subjects even if outputs already
#   exist. If you want to skip subjects that have already been processed
#   (e.g. after a crash on a large dataset), set skip_exists <- TRUE directly
#   in the script.
#
# Input folder structure expected per subject:
#   <dataset_root>/<subject_id>/T1.nii.gz
#   <dataset_root>/<subject_id>/FLAIR.nii.gz
#   <dataset_root>/<subject_id>/EPIm.nii.gz
#   <dataset_root>/<subject_id>/EPIp.nii.gz
#
# Additional input files required per subject:
#   <brainmask_root>/<subject_id>_brainmask.nii.gz
#   <labels_root>/<subject_id>_ses-01_mask-instances.nii.gz
#
# Label encoding expected in instance label masks:
#   0         = background
#   1000-1999 = PRL lesions
#   >= 2000   = non-PRL lesions
#
# Outputs per subject (saved in <out_root>/<subject_id>_ses-01/):
#   - prob.nii.gz                  : MIMoSA lesion probability map
#   - t1_final.nii.gz              : WhiteStripe-normalized T1
#   - flair_final.nii.gz           : WhiteStripe-normalized FLAIR
#   - epi_final.nii.gz             : z-score normalized EPI
#   - phase_final.nii.gz           : WhiteStripe-normalized phase
#   - mask_final.nii.gz            : brain mask in EPI space
#   - t1_n4_bet.nii.gz             : bias-corrected T1 in EPI space
#   - flair_n4_bet.nii.gz          : bias-corrected FLAIR in EPI space
#   - epi_n4_bet.nii.gz            : bias-corrected EPI
#   - phase_n4_bet_ws.nii.gz       : bias-corrected WhiteStripe phase
#   - prl_truth_mask.nii.gz        : binary PRL truth mask
#   - lesion_truth_mask.nii.gz     : binary lesion truth mask (PRL + non-PRL)
#   - labels_raw_copy.nii.gz       : copy of raw instance labels
#   - prob_50_binary.nii.gz        : MIMoSA probability map thresholded at 0.5
#   - prob_50_labeled.nii.gz       : labeled lesion candidates
# =============================================================================

options(error = function() {
  traceback(2)
  quit(save = "no", status = 1)
})

# -----------------------------------------------------------------------------
# Load libraries
# -----------------------------------------------------------------------------
suppressPackageStartupMessages({
  library(neurobase)
  library(mimosa)
  library(fslr)
  library(WhiteStripe)
  library(stringr)
  library(ANTsR)
  library(ANTsRCore)
  library(extrantsr)
  library(pbapply)
  library(oro.nifti)
})

source("00_lesion_helper_functions.R")
source("00_preprocessing_helper_functions.R")

# -----------------------------------------------------------------------------
# Parse command-line arguments
# -----------------------------------------------------------------------------
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 6) {
  stop("Usage: Rscript run_mimosa_preprocessing.R <sublist_file> <dataset_root> ",
       "<brainmask_root> <labels_root> <out_root> <mimosa_model_file>")
}

sublist_file      <- args[1]
dataset_root      <- args[2]
brainmask_root    <- args[3]
labels_root       <- args[4]
out_root          <- args[5]
mimosa_model_file <- args[6]

# Set to TRUE to skip subjects whose outputs already exist (useful after a crash)
skip_exists <- FALSE

if (!file.exists(sublist_file)) {
  stop("Subject list file does not exist: ", sublist_file)
}

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

# Check that a file exists and return its path, or stop with a clear message
check_file <- function(path, label) {
  if (!file.exists(path)) {
    stop(paste0("Missing ", label, ": ", path))
  }
  path
}

# Z-score normalize an image within a brain mask
safe_zscore <- function(img, brain_mask, label_for_print = "image") {
  img_arr  <- as.array(img)
  mask_idx <- as.array(brain_mask) > 0
  vals     <- img_arr[mask_idx]
  vals     <- vals[is.finite(vals)]

  if (length(vals) == 0) {
    stop(paste0("No finite voxels inside brain mask for ", label_for_print))
  }

  mu  <- mean(vals, na.rm = TRUE)
  sdv <- sd(vals, na.rm = TRUE)
  cat(label_for_print, " mean:", mu, " sd:", sdv, "\n")

  if (!is.finite(sdv) || sdv == 0) {
    stop(paste0("Standard deviation is zero/non-finite for ", label_for_print))
  }

  ((img - mu) / sdv) * brain_mask
}

# Build binary PRL and lesion truth masks from raw instance label file
# Label encoding: 0 = background, 1000-1999 = PRL, >= 2000 = lesion
make_truth_from_labels_raw <- function(labels_path, ref_epi, out_dir) {
  lab_nii  <- readNIfTI(labels_path, reorient = FALSE)
  lab_arr  <- round(lab_nii@.Data)
  storage.mode(lab_arr) <- "integer"

  prl_mask    <- (lab_arr >= 1000 & lab_arr < 2000)
  lesion_mask <- ((lab_arr >= 1000 & lab_arr < 2000) | (lab_arr >= 2000))

  prl_img    <- lab_nii
  lesion_img <- lab_nii

  prl_img@.Data    <- prl_mask * 1L
  lesion_img@.Data <- lesion_mask * 1L

  prl_rpi    <- oro2ants(orient_rpi(prl_img)$img)
  lesion_rpi <- oro2ants(orient_rpi(lesion_img)$img)

  prl_rpi    <- antsCopyImageInfo(ref_epi, prl_rpi)
  lesion_rpi <- antsCopyImageInfo(ref_epi, lesion_rpi)

  antsImageWrite(prl_rpi,    file.path(out_dir, "prl_truth_mask.nii.gz"))
  antsImageWrite(lesion_rpi, file.path(out_dir, "lesion_truth_mask.nii.gz"))
}

# -----------------------------------------------------------------------------
# Read and validate subject list
# -----------------------------------------------------------------------------
subject_bases <- readLines(sublist_file, warn = FALSE)
subject_bases <- trimws(subject_bases)
subject_bases <- subject_bases[subject_bases != ""]
subject_bases <- subject_bases[!grepl("^#", subject_bases)]

if (length(subject_bases) == 0) {
  stop("No valid subjects found in subject list file.")
}

# Append session suffix: sub-001 -> sub-001_ses-01
subject_list <- paste0(subject_bases, "_ses-01")

cat("Subjects to process:\n")
print(subject_list)

# -----------------------------------------------------------------------------
# Create output root directory
# -----------------------------------------------------------------------------
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# Main loop — process each subject
# -----------------------------------------------------------------------------
for (subject_id in subject_list) {

  cat("\n========================================\n")
  cat("Processing:", subject_id, "\n")
  cat("========================================\n")

  subject_base <- sub("_ses-.*$", "", subject_id)
  subject_dir  <- file.path(dataset_root, subject_base)
  reg_out_dir  <- file.path(out_root, subject_id)
  dir.create(reg_out_dir, recursive = TRUE, showWarnings = FALSE)

  # ---------------------------------------------------------------------------
  # Validate input files
  # ---------------------------------------------------------------------------
  t1_path    <- check_file(file.path(subject_dir, "T1.nii.gz"),    "T1")
  flair_path <- check_file(file.path(subject_dir, "FLAIR.nii.gz"), "FLAIR")
  epi_path   <- check_file(file.path(subject_dir, "EPIm.nii.gz"),  "EPIm")
  phase_path <- check_file(file.path(subject_dir, "EPIp.nii.gz"),  "EPIp")

  brainmask_path <- check_file(
    file.path(brainmask_root, paste0(subject_base, "_brainmask.nii.gz")),
    "brain mask"
  )

  labels_path <- check_file(
    file.path(labels_root, paste0(subject_id, "_mask-instances.nii.gz")),
    "instance label mask"
  )

  check_file(mimosa_model_file, "MIMoSA model file")

  # ---------------------------------------------------------------------------
  # Step 1: Preprocessing and MIMoSA probability map
  # Skip if all outputs already exist and skip_exists is TRUE
  # ---------------------------------------------------------------------------
  needed_outputs <- c(
    "prob.nii.gz", "t1_final.nii.gz", "flair_final.nii.gz",
    "epi_final.nii.gz", "phase_final.nii.gz", "mask_final.nii.gz",
    "t1_n4_bet.nii.gz", "flair_n4_bet.nii.gz",
    "epi_n4_bet.nii.gz", "phase_n4_bet_ws.nii.gz"
  )
  all_exist <- all(file.exists(file.path(reg_out_dir, needed_outputs)))

  if (!(skip_exists && all_exist)) {

    # N4 bias correction on EPI and phase
    epi   <- read_rpi(epi_path)
    phase <- read_rpi(phase_path)

    epi_n4 <- oro2ants(bias_correct(
      epi, correction = "N4", reorient = FALSE, verbose = FALSE
    ))
    phase_n4 <- oro2ants(bias_correct(
      phase, correction = "N4", reorient = FALSE, verbose = FALSE
    ))

    t1_n4    <- check_ants(t1_path)
    flair_n4 <- check_ants(flair_path)
    t1_brain <- check_ants(brainmask_path)

    # Prepare brain mask
    brain_mask <- t1_brain != 0
    brain_mask <- oro2ants(orient_rpi(ants2oro(brain_mask))$img)
    brain_mask[brain_mask > 0]  <- 1
    brain_mask[brain_mask <= 0] <- 0

    # Register T1 and FLAIR to EPI space
    t1_to_epi <- registration(
      filename = t1_n4, template.file = epi_n4,
      typeofTransform = "Rigid", remove.warp = FALSE, verbose = FALSE
    )
    flair_to_epi <- registration(
      filename = flair_n4, template.file = epi_n4,
      typeofTransform = "Rigid", remove.warp = FALSE, verbose = FALSE
    )

    phase_n4 <- antsCopyImageInfo(epi_n4, phase_n4)

    t1_reg <- antsApplyTransforms(
      fixed = epi_n4, moving = t1_n4,
      transformlist = c(t1_to_epi$fwdtransforms),
      interpolator = "lanczosWindowedSinc"
    )

    # Register brain mask to EPI space using T1 transform
    brain_mask <- antsApplyTransforms(
      fixed = epi_n4, moving = brain_mask,
      transformlist = c(t1_to_epi$fwdtransforms),
      interpolator = "nearestNeighbor"
    )
    brain_mask[brain_mask > 0]  <- 1
    brain_mask[brain_mask <= 0] <- 0

    flair_reg <- antsApplyTransforms(
      fixed = epi_n4, moving = flair_n4,
      transformlist = c(flair_to_epi$fwdtransforms),
      interpolator = "lanczosWindowedSinc"
    )

    # Diagnostics
    mask_idx <- as.array(brain_mask) > 0
    cat("Subject:", subject_id, "\n")
    cat("Brain mask voxels:", sum(mask_idx), "\n")
    cat("Range epi_n4:",   paste(range(as.array(epi_n4),   na.rm = TRUE), collapse = " / "), "\n")
    cat("Range phase_n4:", paste(range(as.array(phase_n4), na.rm = TRUE), collapse = " / "), "\n")
    cat("Range t1_reg:",   paste(range(as.array(t1_reg),   na.rm = TRUE), collapse = " / "), "\n")
    cat("Range flair_reg:",paste(range(as.array(flair_reg),na.rm = TRUE), collapse = " / "), "\n")

    if (sum(mask_idx) == 0) {
      stop(paste0("Brain mask is empty after registration for ", subject_id))
    }

    # Intensity normalization — z-score for EPI, WhiteStripe for T1/FLAIR/phase
    epi_final <- safe_zscore(epi_n4, brain_mask, "epi_n4")

    # WhiteStripe normalization for phase
    tmp <- ants2oro(phase_n4 * brain_mask)
    storage.mode(tmp@.Data) <- "double"
    tmp@.Data[!is.finite(tmp@.Data)] <- 0
    tmp@.Data[abs(tmp@.Data) <= 1]   <- 0
    cat("Phase tmp range:", paste(range(tmp@.Data, na.rm = TRUE), collapse = " / "), "\n")
    ind            <- whitestripe(tmp, "T2", stripped = TRUE, verbose = FALSE)
    phase_n4_bet_ws <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))
    phase_final    <- antsCopyImageInfo(epi_n4, phase_n4_bet_ws) * brain_mask

    # WhiteStripe normalization for T1
    tmp <- ants2oro(t1_reg * brain_mask)
    storage.mode(tmp@.Data) <- "double"
    tmp@.Data[!is.finite(tmp@.Data)] <- 0
    tmp@.Data[abs(tmp@.Data) <= 1]   <- 0
    cat("T1 tmp range:", paste(range(tmp@.Data, na.rm = TRUE), collapse = " / "), "\n")
    ind          <- whitestripe(tmp, "T1", stripped = TRUE, verbose = FALSE)
    t1_final_ws  <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))
    t1_final_ws  <- antsCopyImageInfo(epi_n4, t1_final_ws) * brain_mask

    # WhiteStripe normalization for FLAIR
    tmp <- ants2oro(flair_reg * brain_mask)
    storage.mode(tmp@.Data) <- "double"
    tmp@.Data[!is.finite(tmp@.Data)] <- 0
    tmp@.Data[abs(tmp@.Data) <= 1]   <- 0
    cat("FLAIR tmp range:", paste(range(tmp@.Data, na.rm = TRUE), collapse = " / "), "\n")
    ind            <- whitestripe(tmp, "T2", stripped = TRUE, verbose = FALSE)
    flair_final_ws <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))
    flair_final_ws <- antsCopyImageInfo(epi_n4, flair_final_ws) * brain_mask

    # Run MIMoSA to generate lesion probability map
    mimosa_obj <- mimosa_data(
      brain_mask = ants2oro(brain_mask),
      FLAIR      = ants2oro(flair_final_ws),
      T1         = ants2oro(t1_final_ws),
      gold_standard = NULL,
      normalize  = "no",
      cores      = 1,
      verbose    = FALSE
    )

    mimosa_df    <- mimosa_obj$mimosa_dataframe
    cand_voxels  <- mimosa_obj$top_voxels
    tissue_mask  <- mimosa_obj$tissue_mask

    load(mimosa_model_file)
    predictions_WS <- predict(mimosa_model, mimosa_df, type = "response")

    predictions_nifti_WS <- niftiarr(cand_voxels, 0)
    predictions_nifti_WS[cand_voxels == 1] <- predictions_WS

    # Smooth probability map with Gaussian kernel (sigma = 1.25mm)
    probmap <- oro2ants(fslsmooth(
      predictions_nifti_WS, sigma = 1.25,
      mask = tissue_mask, retimg = TRUE,
      smooth_mask = TRUE, verbose = FALSE
    ))
    probmap <- antsCopyImageInfo(epi_n4, probmap)

    # Save all preprocessed images and probability map
    antsImageWrite(t1_reg * brain_mask,          file.path(reg_out_dir, "t1_n4_bet.nii.gz"))
    antsImageWrite(flair_reg * brain_mask,        file.path(reg_out_dir, "flair_n4_bet.nii.gz"))
    antsImageWrite(epi_n4 * brain_mask,           file.path(reg_out_dir, "epi_n4_bet.nii.gz"))
    antsImageWrite(phase_n4_bet_ws * brain_mask,  file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz"))
    antsImageWrite(probmap,                        file.path(reg_out_dir, "prob.nii.gz"))
    antsImageWrite(t1_final_ws,                   file.path(reg_out_dir, "t1_final.nii.gz"))
    antsImageWrite(flair_final_ws,                file.path(reg_out_dir, "flair_final.nii.gz"))
    antsImageWrite(epi_final,                     file.path(reg_out_dir, "epi_final.nii.gz"))
    antsImageWrite(phase_final,                   file.path(reg_out_dir, "phase_final.nii.gz"))
    antsImageWrite(brain_mask,                    file.path(reg_out_dir, "mask_final.nii.gz"))

  } else {
    cat("Skipping preprocessing — all outputs already exist.\n")
  }

  # ---------------------------------------------------------------------------
  # Step 2: Build binary truth masks from raw instance labels
  # ---------------------------------------------------------------------------
  epi_final <- check_ants(file.path(reg_out_dir, "epi_final.nii.gz"))
  make_truth_from_labels_raw(labels_path, epi_final, reg_out_dir)

  # Save a copy of raw instance labels in EPI space
  raw_labels <- check_ants(labels_path)
  raw_labels <- antsCopyImageInfo(epi_final, raw_labels)
  antsImageWrite(raw_labels, file.path(reg_out_dir, "labels_raw_copy.nii.gz"))

  # ---------------------------------------------------------------------------
  # Step 3: Generate labeled lesion candidates from MIMoSA probability map
  # Threshold at 0.5 to obtain binary candidate mask, then label clusters
  # ---------------------------------------------------------------------------
  prob     <- check_ants(file.path(reg_out_dir, "prob.nii.gz"))
  prob_bin <- make_binary_mask(prob, 0.50)

  if (sum(prob_bin) == 0) {
    prob_labeled <- antsImageClone(prob_bin)
  } else {
    prob_bin_clone <- antsImageClone(prob_bin)
    prob_bin_clone[prob_bin_clone > 0] <- 1
    prob_labeled <- labelClusters(prob_bin_clone, fullyConnected = TRUE)
  }

  antsImageWrite(prob_bin,    file.path(reg_out_dir, "prob_50_binary.nii.gz"))
  antsImageWrite(prob_labeled, file.path(reg_out_dir, "prob_50_labeled.nii.gz"))

  cat("Done for", subject_id, "\n")
  cat("Outputs written to:", reg_out_dir, "\n")
}