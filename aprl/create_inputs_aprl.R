#!/usr/bin/env Rscript

options(error = function() {
  traceback(2)
  quit(save = "no", status = 1)
})

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
# ARGUMENTS
# =========================================================
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript script.R /path/to/subject_list.txt")
}

subject_txt <- args[1]

if (!file.exists(subject_txt)) {
  stop(paste0("Subject list file does not exist: ", subject_txt))
}

# =========================================================
# USER CONFIG
# =========================================================
dataset_root <- "/linux/luverheyen/data/dataset_renamed"
brainmask_root <- "/linux/luverheyen/data/synthstrip_raw"
labels_root <- "/linux/luverheyen/data/labels_raw"
out_root <- "/linux/luverheyen/data/processed_50_custom"

mimosa_model_file <- "mimosa_model.RData"

skip_exists <- TRUE
skip_all <- FALSE

# =========================================================
# READ SUBJECT LIST FROM TXT
# Expected format:
# sub-001
# sub-003
# ...
# =========================================================
subject_bases <- readLines(subject_txt, warn = FALSE)
subject_bases <- trimws(subject_bases)
subject_bases <- subject_bases[subject_bases != ""]
subject_bases <- subject_bases[!grepl("^#", subject_bases)]

if (length(subject_bases) == 0) {
  stop("No valid subjects found in txt file.")
}

# convert sub-001 -> sub-001_ses-01
subject_list <- paste0(subject_bases, "_ses-01")

cat("Subjects to process:\n")
print(subject_list)

# =========================================================
# HELPERS
# =========================================================
check_file <- function(path, label) {
  if (!file.exists(path)) {
    stop(paste0("Missing ", label, ": ", path))
  }
  path
}

safe_zscore <- function(img, brain_mask, label_for_print = "image") {
  img_arr <- as.array(img)
  mask_idx <- as.array(brain_mask) > 0

  vals <- img_arr[mask_idx]
  vals <- vals[is.finite(vals)]

  if (length(vals) == 0) {
    stop(paste0("No finite voxels inside brain mask for ", label_for_print))
  }

  mu <- mean(vals, na.rm = TRUE)
  sdv <- sd(vals, na.rm = TRUE)

  cat(label_for_print, " mean:", mu, " sd:", sdv, "\n")

  if (!is.finite(sdv) || sdv == 0) {
    stop(paste0("Standard deviation is zero/non-finite for ", label_for_print))
  }

  ((img - mu) / sdv) * brain_mask
}

make_truth_from_labels_raw <- function(labels_path, ref_epi, out_dir) {
  lab_nii <- readNIfTI(labels_path, reorient = FALSE)
  lab_arr <- round(lab_nii@.Data)
  storage.mode(lab_arr) <- "integer"

  # USER-SPECIFIED ENCODING:
  # 0 = background
  # 1000-1999 = PRL
  # >= 2000 = lesion
  #
  # We create:
  # - prl_truth: voxels in [1000,1999]
  # - lesion_truth: voxels in [1000,1999] OR >= 2000
  #   because PRL is also a lesion biologically
  prl_mask <- (lab_arr >= 1000 & lab_arr < 2000)
  lesion_mask <- ((lab_arr >= 1000 & lab_arr < 2000) | (lab_arr >= 2000))

  prl_img <- lab_nii
  lesion_img <- lab_nii

  prl_img@.Data <- prl_mask * 1L
  lesion_img@.Data <- lesion_mask * 1L

  prl_rpi <- oro2ants(orient_rpi(prl_img)$img)
  lesion_rpi <- oro2ants(orient_rpi(lesion_img)$img)

  prl_rpi <- antsCopyImageInfo(ref_epi, prl_rpi)
  lesion_rpi <- antsCopyImageInfo(ref_epi, lesion_rpi)

  antsImageWrite(prl_rpi, file.path(out_dir, "prl_truth_mask.nii.gz"))
  antsImageWrite(lesion_rpi, file.path(out_dir, "lesion_truth_mask.nii.gz"))
}

# =========================================================
# MAIN LOOP
# =========================================================
dir.create(out_root, recursive = TRUE, showWarnings = FALSE)

if (!skip_all) {

  for (subject_id in subject_list) {

    cat("\n========================================\n")
    cat("Processing:", subject_id, "\n")
    cat("========================================\n")

    subject_base <- sub("_ses-.*$", "", subject_id)
    subject_dir <- file.path(dataset_root, subject_base)

    reg_out_dir <- file.path(out_root, subject_id)
    dir.create(reg_out_dir, recursive = TRUE, showWarnings = FALSE)

    # =========================================================
    # INPUT PATHS
    # =========================================================
    t1_path <- check_file(file.path(subject_dir, "T1.nii.gz"), "T1")
    flair_path <- check_file(file.path(subject_dir, "FLAIR.nii.gz"), "FLAIR")
    epi_path <- check_file(file.path(subject_dir, "EPIm.nii.gz"), "EPIm")
    phase_path <- check_file(file.path(subject_dir, "EPIp.nii.gz"), "EPIp")

    brainmask_path <- check_file(
      file.path(brainmask_root, paste0(subject_base, "_brainmask.nii.gz")),
      "SynthStrip brain mask"
    )

    labels_path <- check_file(
      file.path(labels_root, paste0(subject_id, "_mask-instances.nii.gz")),
      "labels_raw lesion/PRL mask"
    )

    mimosa_model_file <- check_file(
      mimosa_model_file,
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

      brain_mask[brain_mask > 0] <- 1
      brain_mask[brain_mask <= 0] <- 0

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

      brain_mask[brain_mask > 0] <- 1
      brain_mask[brain_mask <= 0] <- 0

      flair_reg <- antsApplyTransforms(
        fixed = epi_n4,
        moving = flair_n4,
        transformlist = c(flair_to_epi$fwdtransforms),
        interpolator = "lanczosWindowedSinc"
      )

      mask_idx <- as.array(brain_mask) > 0

      cat("Subject:", subject_id, "\n")
      cat("Brain mask voxels:", sum(mask_idx), "\n")
      cat("Range epi_n4:", paste(range(as.array(epi_n4), na.rm = TRUE), collapse = " / "), "\n")
      cat("Range phase_n4:", paste(range(as.array(phase_n4), na.rm = TRUE), collapse = " / "), "\n")
      cat("Range t1_reg:", paste(range(as.array(t1_reg), na.rm = TRUE), collapse = " / "), "\n")
      cat("Range flair_reg:", paste(range(as.array(flair_reg), na.rm = TRUE), collapse = " / "), "\n")

      if (sum(mask_idx) == 0) {
        stop(paste0("Brain mask is empty after registration for ", subject_id))
      }

      # EPI final
      epi_final <- safe_zscore(epi_n4, brain_mask, "epi_n4")

      # WhiteStripe phase
      tmp <- ants2oro(phase_n4 * brain_mask)
      storage.mode(tmp@.Data) <- "double"
      tmp@.Data[!is.finite(tmp@.Data)] <- 0
      tmp@.Data[abs(tmp@.Data) <= 1] <- 0

      cat("Phase tmp class:", class(tmp@.Data), "\n")
      cat("Phase tmp range:", paste(range(tmp@.Data, na.rm = TRUE), collapse = " / "), "\n")
      cat("Phase tmp nonzero:", sum(tmp@.Data != 0, na.rm = TRUE), "\n")

      ind <- whitestripe(tmp, "T2", stripped = TRUE, verbose = FALSE)
      phase_n4_bet_ws <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))

      # Saved final phase image:
      # keep WhiteStripe-normalized phase because that is the one you explicitly compute
      phase_final <- antsCopyImageInfo(epi_n4, phase_n4_bet_ws) * brain_mask

      # WhiteStripe T1
      tmp <- ants2oro(t1_reg * brain_mask)
      storage.mode(tmp@.Data) <- "double"
      tmp@.Data[!is.finite(tmp@.Data)] <- 0
      tmp@.Data[abs(tmp@.Data) <= 1] <- 0

      cat("T1 tmp class:", class(tmp@.Data), "\n")
      cat("T1 tmp range:", paste(range(tmp@.Data, na.rm = TRUE), collapse = " / "), "\n")
      cat("T1 tmp nonzero:", sum(tmp@.Data != 0, na.rm = TRUE), "\n")

      ind <- whitestripe(tmp, "T1", stripped = TRUE, verbose = FALSE)
      t1_final_ws <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))
      t1_final_ws <- antsCopyImageInfo(epi_n4, t1_final_ws) * brain_mask

      # WhiteStripe FLAIR
      tmp <- ants2oro(flair_reg * brain_mask)
      storage.mode(tmp@.Data) <- "double"
      tmp@.Data[!is.finite(tmp@.Data)] <- 0
      tmp@.Data[abs(tmp@.Data) <= 1] <- 0

      cat("FLAIR tmp class:", class(tmp@.Data), "\n")
      cat("FLAIR tmp range:", paste(range(tmp@.Data, na.rm = TRUE), collapse = " / "), "\n")
      cat("FLAIR tmp nonzero:", sum(tmp@.Data != 0, na.rm = TRUE), "\n")

      ind <- whitestripe(tmp, "T2", stripped = TRUE, verbose = FALSE)
      flair_final_ws <- oro2ants(whitestripe_norm(tmp, ind$whitestripe.ind))
      flair_final_ws <- antsCopyImageInfo(epi_n4, flair_final_ws) * brain_mask

      # MIMoSA input + probability map
      mimosa_obj <- mimosa_data(
        brain_mask = ants2oro(brain_mask),
        FLAIR = ants2oro(flair_final_ws),
        T1 = ants2oro(t1_final_ws),
        gold_standard = NULL,
        normalize = "no",
        cores = 1,
        verbose = FALSE
      )

      mimosa_df <- mimosa_obj$mimosa_dataframe
      cand_voxels <- mimosa_obj$top_voxels
      tissue_mask <- mimosa_obj$tissue_mask

      load(mimosa_model_file)
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

      probmap <- antsCopyImageInfo(epi_n4, probmap)

      antsImageWrite(t1_reg * brain_mask, file.path(reg_out_dir, "t1_n4_bet.nii.gz"))
      antsImageWrite(flair_reg * brain_mask, file.path(reg_out_dir, "flair_n4_bet.nii.gz"))
      antsImageWrite(epi_n4 * brain_mask, file.path(reg_out_dir, "epi_n4_bet.nii.gz"))
      antsImageWrite(phase_n4_bet_ws * brain_mask, file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz"))

      antsImageWrite(probmap, file.path(reg_out_dir, "prob.nii.gz"))
      antsImageWrite(t1_final_ws, file.path(reg_out_dir, "t1_final.nii.gz"))
      antsImageWrite(flair_final_ws, file.path(reg_out_dir, "flair_final.nii.gz"))
      antsImageWrite(epi_final, file.path(reg_out_dir, "epi_final.nii.gz"))
      antsImageWrite(phase_final, file.path(reg_out_dir, "phase_final.nii.gz"))
      antsImageWrite(brain_mask, file.path(reg_out_dir, "mask_final.nii.gz"))

    } else {
      cat("Skipping preprocessing because outputs already exist.\n")
    }

    # =========================================================
    # BUILD TRUTH MASKS FROM labels_raw
    # =========================================================
    epi_final <- check_ants(file.path(reg_out_dir, "epi_final.nii.gz"))
    make_truth_from_labels_raw(labels_path, epi_final, reg_out_dir)

    # also keep a copy of raw labels
    raw_labels <- check_ants(labels_path)
    raw_labels <- antsCopyImageInfo(epi_final, raw_labels)
    antsImageWrite(raw_labels, file.path(reg_out_dir, "labels_raw_copy.nii.gz"))

    # =========================================================
    # BUILD APRL LESION CANDIDATES
    # APRL-style threshold = 0.20
    # =========================================================
    prob <- check_ants(file.path(reg_out_dir, "prob.nii.gz"))

    prob_20 <- make_binary_mask(prob, 0.50) # change from 0.2 to 0.5 to try 29/04/2026

    if (sum(prob_20) == 0) {
      prob_20_labeled <- antsImageClone(prob_20)
    } else {
      prob_20_bin <- antsImageClone(prob_20)
      prob_20_bin[prob_20_bin > 0] <- 1

      prob_20_labeled <- labelClusters(
        prob_20_bin,
        fullyConnected = TRUE
      )
    }

    antsImageWrite(prob_20, file.path(reg_out_dir, "prob_50_binary.nii.gz"))
    antsImageWrite(prob_20_labeled, file.path(reg_out_dir, "prob_50_labeled.nii.gz"))

    cat("Done for ", subject_id, "\n")
    cat("Outputs written to: ", reg_out_dir, "\n")
  }

} else {
  cat("skip_all = TRUE, nothing was processed.\n")
}