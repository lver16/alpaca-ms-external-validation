#!/usr/bin/env Rscript

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

# =========================================================
# USER CONFIG
# =========================================================
dataset_root <- "/linux/luverheyen/data/dataset_renamed"
brainmask_root <- "/linux/luverheyen/data/synthstrip_raw"
combined_root <- "/linux/luverheyen/data/combined_labels"
out_root <- "/linux/luverheyen/data/processed_05_custom"

mimosa_model_path <- "mimosa_model.RData"

skip_exists <- TRUE
skip_all <- FALSE

# Subject to process
subject_id <- "sub-243_ses-01"

# Example:
# subject_id = sub-252_ses-01
# folder with MRI files = /linux/.../dataset_renamed/sub-252/
subject_base <- sub("_ses-.*$", "", subject_id)
subject_dir <- file.path(dataset_root, subject_base)

# =========================================================
# HELPERS
# =========================================================

check_file <- function(path, label) {
  if (!file.exists(path)) {
    stop(paste0("Missing ", label, ": ", path))
  }
  path
}

make_truth_from_combined <- function(combined_path, ref_epi, out_dir) {
  combined_nii <- readNIfTI(combined_path, reorient = FALSE)
  combined_arr <- round(combined_nii@.Data)
  storage.mode(combined_arr) <- "integer"

  # Assumed encoding:
  # 0 = background
  # 1000-1999 lesion only
  # 2000-2999 PRL
  # 3000-3999 CVS
  # 4000-4999 PRL+CVS

  prl_mask <- ((combined_arr >= 2000 & combined_arr < 3000) |
               (combined_arr >= 4000 & combined_arr < 5000))

  cvs_mask <- ((combined_arr >= 3000 & combined_arr < 4000) |
               (combined_arr >= 4000 & combined_arr < 5000))

  prl_img <- combined_nii
  cvs_img <- combined_nii
  prl_img@.Data <- prl_mask * 1L
  cvs_img@.Data <- cvs_mask * 1L

  prl_rpi <- oro2ants(orient_rpi(prl_img)$img)
  cvs_rpi <- oro2ants(orient_rpi(cvs_img)$img)

  antsCopyImageInfo(ref_epi, prl_rpi)
  antsCopyImageInfo(ref_epi, cvs_rpi)

  # mimic the original dilation for PRL coordinates
  prl_rpi <- iMath(prl_rpi, "GD", 2)

  # keep same categorical spirit as original CVS coords
  cvs_tmp <- antsImageClone(cvs_rpi)
  cvs_tmp[cvs_tmp > 0] <- 3

  antsImageWrite(prl_rpi, file.path(out_dir, "prl_coords.nii.gz"))
  antsImageWrite(cvs_tmp, file.path(out_dir, "cvs_coords.nii.gz"))
}

# =========================================================
# INPUT PATHS
# =========================================================
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)
reg_out_dir <- file.path(out_root, subject_id)
dir.create(reg_out_dir, recursive = TRUE, showWarnings = FALSE)

# Exact filenames in your dataset
t1_path <- check_file(file.path(subject_dir, "T1.nii.gz"), "T1")
flair_path <- check_file(file.path(subject_dir, "FLAIR.nii.gz"), "FLAIR")
epi_path <- check_file(file.path(subject_dir, "EPIm.nii.gz"), "EPIm (magnitude EPI)")
phase_path <- check_file(file.path(subject_dir, "EPIp.nii.gz"), "EPIp (phase)")

brainmask_path <- check_file(
  file.path(brainmask_root, paste0(subject_base, "_brainmask.nii.gz")),
  "SynthStrip brain mask"
)

combined_path <- check_file(
  file.path(combined_root, paste0(subject_id, "_combined.nii.gz")),
  "combined label file"
)

mimosa_model_path <- check_file(
  mimosa_model_path,
  "MIMoSA model file"
)

# =========================================================
# PREPROCESSING + MIMoSA PROBABILITY MAP
# =========================================================
needed_outputs <- c(
  "prob.nii.gz",
  "t1_final.nii.gz",
  "flair_final.nii.gz",
  "epi_final.nii.gz",
  "phase_final.nii.gz",
  "mask_final.nii.gz",
  "t1_n4_bet.nii.gz",
  "flair_n4_bet.nii.gz",
  "epi_n4_bet.nii.gz",
  "phase_n4_bet_ws.nii.gz"
)

all_exist <- all(file.exists(file.path(reg_out_dir, needed_outputs)))

if (!(skip_exists && all_exist)) {
  epi <- read_rpi(epi_path)
  phase <- read_rpi(phase_path)

  epi_n4 <- oro2ants(bias_correct(
    epi,
    correction = "N4",
    reorient = FALSE,
    verbose = FALSE
  ))

  phase_n4 <- oro2ants(bias_correct(
    phase,
    correction = "N4",
    reorient = FALSE,
    verbose = FALSE
  ))

  t1_n4 <- check_ants(t1_path)
  flair_n4 <- check_ants(flair_path)
  t1_brain <- check_ants(brainmask_path)

  brain_mask <- t1_brain != 0
  brain_mask <- oro2ants(orient_rpi(ants2oro(brain_mask))$img)

  # Register T1 and FLAIR to EPI
  t1_to_epi <- registration(
    filename = t1_n4,
    template.file = epi_n4,
    typeofTransform = "Rigid",
    remove.warp = FALSE,
    verbose = FALSE
  )

  flair_to_epi <- registration(
    filename = flair_n4,
    template.file = epi_n4,
    typeofTransform = "Rigid",
    remove.warp = FALSE,
    verbose = FALSE
  )

  phase_n4 <- antsCopyImageInfo(epi_n4, phase_n4)

  t1_reg <- antsApplyTransforms(
    fixed = epi_n4,
    moving = t1_n4,
    transformlist = c(t1_to_epi$fwdtransforms),
    interpolator = "lanczosWindowedSinc"
  )

  brain_mask <- antsApplyTransforms(
    fixed = epi_n4,
    moving = brain_mask,
    transformlist = c(t1_to_epi$fwdtransforms),
    interpolator = "nearestNeighbor"
  )

  flair_reg <- antsApplyTransforms(
    fixed = epi_n4,
    moving = flair_n4,
    transformlist = c(flair_to_epi$fwdtransforms),
    interpolator = "lanczosWindowedSinc"
  )

  # Normalize EPI
  epi_dist <- c(mean(epi_n4[brain_mask]), sd(epi_n4[brain_mask]))
  epi_final <- ((epi_n4 - epi_dist[1]) / epi_dist[2]) * brain_mask

  # WhiteStripe phase
  tmp <- ants2oro(phase_n4 * brain_mask)
  ind <- whitestripe(tmp, "T2", stripped = TRUE, verbose = FALSE)
  phase_n4_bet_ws <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))

  # Normalize phase
  phase_dist <- c(mean(phase_n4[brain_mask]), sd(phase_n4[brain_mask]))
  phase_final <- ((phase_n4 - phase_dist[1]) / phase_dist[2]) * brain_mask

  # WhiteStripe T1
  tmp <- ants2oro(t1_reg * brain_mask)
  ind <- whitestripe(tmp, "T1", stripped = TRUE, verbose = FALSE)
  t1_final <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))

  # WhiteStripe FLAIR
  tmp <- ants2oro(flair_reg * brain_mask)
  ind <- whitestripe(tmp, "T2", stripped = TRUE, verbose = FALSE)
  flair_final <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))

  # MIMoSA input + probability map
  mimosa_obj <- mimosa_data(
    brain_mask = ants2oro(brain_mask),
    FLAIR = ants2oro(flair_final),
    T1 = ants2oro(t1_final),
    gold_standard = NULL,
    normalize = "no",
    cores = 1,
    verbose = FALSE
  )

  mimosa_df <- mimosa_obj$mimosa_dataframe
  cand_voxels <- mimosa_obj$top_voxels
  tissue_mask <- mimosa_obj$tissue_mask

  load(mimosa_model_path)
  predictions_WS <- predict(mimosa_model, mimosa_df, type = "response")

  predictions_nifti_WS <- niftiarr(cand_voxels, 0)
  predictions_nifti_WS[cand_voxels == 1] <- predictions_WS

  probmap <- oro2ants(
    fslsmooth(
      predictions_nifti_WS,
      sigma = 1.25,
      mask = tissue_mask,
      retimg = TRUE,
      smooth_mask = TRUE,
      verbose = FALSE
    )
  )

  # Final renormalization like original workflow
  flair_dist <- c(mean(flair_final[brain_mask]), sd(flair_final[brain_mask]))
  flair_final <- ((flair_final - flair_dist[1]) / flair_dist[2]) * brain_mask

  t1_dist <- c(mean(t1_final[brain_mask]), sd(t1_final[brain_mask]))
  t1_final <- ((t1_final - t1_dist[1]) / t1_dist[2]) * brain_mask

  antsImageWrite(t1_reg * brain_mask, file.path(reg_out_dir, "t1_n4_bet.nii.gz"))
  antsImageWrite(flair_reg * brain_mask, file.path(reg_out_dir, "flair_n4_bet.nii.gz"))
  antsImageWrite(epi_n4 * brain_mask, file.path(reg_out_dir, "epi_n4_bet.nii.gz"))
  antsImageWrite(phase_n4_bet_ws, file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz"))

  antsImageWrite(probmap, file.path(reg_out_dir, "prob.nii.gz"))
  antsImageWrite(t1_final, file.path(reg_out_dir, "t1_final.nii.gz"))
  antsImageWrite(flair_final, file.path(reg_out_dir, "flair_final.nii.gz"))
  antsImageWrite(epi_final, file.path(reg_out_dir, "epi_final.nii.gz"))
  antsImageWrite(phase_final, file.path(reg_out_dir, "phase_final.nii.gz"))
  antsImageWrite(brain_mask, file.path(reg_out_dir, "mask_final.nii.gz"))
} else {
  cat("Skipping preprocessing because outputs already exist.\n")
}

# =========================================================
# BUILD PRL/CVS TRUTH FROM COMBINED LABELS
# =========================================================
epi_final <- check_ants(file.path(reg_out_dir, "epi_final.nii.gz"))
make_truth_from_combined(combined_path, epi_final, reg_out_dir)

# =========================================================
# LABEL LESIONS
# =========================================================
prob <- check_ants(file.path(reg_out_dir, "prob.nii.gz"))
prl_coords <- check_ants(file.path(reg_out_dir, "prl_coords.nii.gz"))

cvs_exists <- file.exists(file.path(reg_out_dir, "cvs_coords.nii.gz"))
if (cvs_exists) {
  cvs_coords <- check_ants(file.path(reg_out_dir, "cvs_coords.nii.gz"))
} else {
  cvs_coords <- NULL
}

contains_lesions <- cvs_exists

prob_05 <- make_binary_mask(prob, 0.05)
prob_30 <- make_binary_mask(prob, 0.30)

if (sum(prob_05) == 0) {
  prob_05_labeled <- antsImageClone(prob_05)
  lesion_type <- antsImageClone(prob_05)
} else {
  prob_05_bin <- antsImageClone(prob_05)
  prob_05_bin[prob_05_bin > 0] <- 1

  prob_05_labeled <- labelClusters(
    prob_05_bin,
    minClusterSize = 30,
    fullyConnected = TRUE
  )

  lesion_type <- annotate_lesion_mask(
    prob_05_labeled,
    cvs_exists,
    prl_coords,
    cvs_coords,
    contains_lesions
  )
}

antsImageWrite(prob_05_labeled, file.path(reg_out_dir, "prob_05.nii.gz"))
antsImageWrite(lesion_type, file.path(reg_out_dir, "lesion_labels.nii.gz"))

if (sum(prob_30) == 0) {
  prob_30_labeled <- antsImageClone(prob_30)
} else {
  prob_30_bin <- antsImageClone(prob_30)
  prob_30_bin[prob_30_bin > 0] <- 1

  prob_30_labeled <- labelClusters(
    prob_30_bin,
    minClusterSize = 100,
    fullyConnected = TRUE
  )
}
antsImageWrite(prob_30_labeled, file.path(reg_out_dir, "prob_30.nii.gz"))

# =========================================================
# LESION_LABELS_TRUE  (SAFE VERSION)
# =========================================================
lesion_labels <- check_ants(file.path(reg_out_dir, "lesion_labels.nii.gz"))
prl_coords <- check_ants(file.path(reg_out_dir, "prl_coords.nii.gz"))

cvs_exists2 <- file.exists(file.path(reg_out_dir, "cvs_coords.nii.gz")) ||
               file.exists(file.path(reg_out_dir, "cvs_coords_nl.nii.gz"))

if (cvs_exists2) {
  if (file.exists(file.path(reg_out_dir, "cvs_coords.nii.gz"))) {
    contains_lesions <- TRUE
    cvs_coords <- check_ants(file.path(reg_out_dir, "cvs_coords.nii.gz"))
  } else {
    contains_lesions <- FALSE
    cvs_coords <- check_ants(file.path(reg_out_dir, "cvs_coords_nl.nii.gz"))
  }
} else {
  cvs_coords <- NULL
}

# Convert to plain arrays
lesion_arr <- as.array(lesion_labels)
prl_arr <- as.array(prl_coords)

if (is.null(cvs_coords)) {
  prl_arr[lesion_arr != 0] <- 0
  prl_arr[prl_arr == 1] <- 1100
  prl_arr[prl_arr == 2] <- 1110

  lesion_labels_true_arr <- lesion_arr + prl_arr

} else {
  cvs_arr <- as.array(cvs_coords)

  prl_arr[lesion_arr != 0] <- 0
  cvs_arr[lesion_arr != 0] <- 0

  tmp_arr <- array(0, dim = dim(prl_arr))

  tmp_arr[(prl_arr > 0) | (cvs_arr > 0)] <- 1100
  tmp_arr[(prl_arr != 2) & (cvs_arr == 3)] <- 1101
  tmp_arr[(prl_arr == 2) & (cvs_arr != 3)] <- 1110
  tmp_arr[(prl_arr == 2) & (cvs_arr == 3)] <- 1111

  lesion_labels_true_arr <- lesion_arr + tmp_arr
}

# Rebuild as ANTs image and copy geometry
lesion_labels_true <- as.antsImage(lesion_labels_true_arr)
antsCopyImageInfo(lesion_labels, lesion_labels_true)

antsImageWrite(
  lesion_labels_true,
  file.path(reg_out_dir, "lesion_labels_true.nii.gz")
)



cat("Done for ", subject_id, "\n")
cat("Outputs written to: ", reg_out_dir, "\n")