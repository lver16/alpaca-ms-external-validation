preprocess_images_flames <- function(
    t1_path,
    flair_path,
    epi_path,
    phase_path,
    flames_path,
    output_dir,
    brainmask_path = NULL,
    reorient = TRUE,
    cores = 1,
    verbose = FALSE,
    return_images = TRUE,
    flames_threshold = 0.05,
    mincluster = 30
) {
  needed_files <- c(t1_path, flair_path, epi_path, phase_path, flames_path)
  if (!all(file.exists(needed_files))) {
    stop("One or more input files do not exist.")
  }

  if (!is.null(brainmask_path) && !file.exists(brainmask_path)) {
    stop("brainmask_path does not exist.")
  }

  if (!file.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  # -----------------------------
  # Read images
  # -----------------------------
  if (reorient) {
    t1    <- oro2ants(read_rpi(t1_path,    verbose = verbose))
    flair <- oro2ants(read_rpi(flair_path, verbose = verbose))
    epi   <- oro2ants(read_rpi(epi_path,   verbose = verbose))
    phase <- oro2ants(read_rpi(phase_path, verbose = verbose))
  } else {
    t1    <- check_ants(t1_path)
    flair <- check_ants(flair_path)
    epi   <- check_ants(epi_path)
    phase <- check_ants(phase_path)
  }

  # -----------------------------
  # Bias correction
  # -----------------------------
  t1    <- n4BiasFieldCorrection(t1,    verbose = verbose)
  flair <- n4BiasFieldCorrection(flair, verbose = verbose)
  epi   <- n4BiasFieldCorrection(epi,   verbose = verbose)
  phase <- n4BiasFieldCorrection(phase, verbose = verbose)

  # -----------------------------
  # Register T1 and FLAIR to EPI
  # -----------------------------
  t1_tx <- antsRegistration(epi, t1, typeofTransform = "Rigid")
  t1_reg <- antsApplyTransforms(
    fixed = epi,
    moving = t1,
    transformlist = c(t1_tx$fwdtransforms),
    interpolator = "lanczosWindowedSinc"
  )

  flair_tx <- antsRegistration(epi, flair, typeofTransform = "Rigid")
  flair_reg <- antsApplyTransforms(
    fixed = epi,
    moving = flair,
    transformlist = c(flair_tx$fwdtransforms),
    interpolator = "lanczosWindowedSinc"
  )

  # keep same metadata convention as original ALPaCA
  phase <- antsCopyImageInfo(epi, phase)

  # -----------------------------
  # Brain mask
  # -----------------------------
  if (is.null(brainmask_path)) {
    mask <- fslbet_robust(t1_reg) > 0
  } else {
    if (reorient) {
      mask <- oro2ants(read_rpi(brainmask_path, verbose = verbose))
    } else {
      mask <- check_ants(brainmask_path)
    }
    mask <- antsCopyImageInfo(epi, mask)
    mask <- antsImageClone(mask > 0)
  }

  # Apply mask
  t1_reg    <- t1_reg * mask
  flair_reg <- flair_reg * mask
  epi       <- epi * mask
  phase     <- phase * mask

  # -----------------------------
  # Intensity normalization
  # -----------------------------
  t1_dist <- c(mean(t1_reg[mask]), sd(t1_reg[mask]))
  t1_final <- ((t1_reg - t1_dist[1]) / t1_dist[2]) * mask
  antsImageWrite(t1_final, file.path(output_dir, "t1_final.nii.gz"))

  flair_dist <- c(mean(flair_reg[mask]), sd(flair_reg[mask]))
  flair_final <- ((flair_reg - flair_dist[1]) / flair_dist[2]) * mask
  antsImageWrite(flair_final, file.path(output_dir, "flair_final.nii.gz"))

  epi_dist <- c(mean(epi[mask]), sd(epi[mask]))
  epi_final <- ((epi - epi_dist[1]) / epi_dist[2]) * mask
  antsImageWrite(epi_final, file.path(output_dir, "epi_final.nii.gz"))

  phase_dist <- c(mean(phase[mask]), sd(phase[mask]))
  phase_final <- ((phase - phase_dist[1]) / phase_dist[2]) * mask
  antsImageWrite(phase_final, file.path(output_dir, "phase_final.nii.gz"))

  # -----------------------------
  # Load FLAMES probability map
  # -----------------------------
  if (reorient) {
    flames_prob <- oro2ants(read_rpi(flames_path, verbose = verbose))
  } else {
    flames_prob <- check_ants(flames_path)
  }

  # Resample FLAMES to EPI space if needed
  same_dim <- all(dim(flames_prob) == dim(epi_final))
  same_spacing <- all(abs(antsGetSpacing(flames_prob) - antsGetSpacing(epi_final)) < 1e-6)
  same_origin <- all(abs(antsGetOrigin(flames_prob) - antsGetOrigin(epi_final)) < 1e-6)
  same_direction <- all(abs(as.vector(antsGetDirection(flames_prob) - antsGetDirection(epi_final))) < 1e-6)

  if (!(same_dim && same_spacing && same_origin && same_direction)) {
    if (verbose) message("Resampling FLAMES probability map to EPI space...")
    flames_prob <- resampleImageToTarget(
      image = flames_prob,
      target = epi_final,
      interpType = "linear"
    )
  } else {
    flames_prob <- antsCopyImageInfo(epi_final, flames_prob)
  }

  flames_prob <- flames_prob * mask
  antsImageWrite(flames_prob, file.path(output_dir, "prob.nii.gz"))

  # -----------------------------
  # Threshold FLAMES probability map
  # -----------------------------
  prob_bin <- antsImageClone(flames_prob > flames_threshold)

  # -----------------------------
  # Label / split candidates
  # -----------------------------
  if (sum(prob_bin) == 0) {
    labeled_candidates <- antsImageClone(prob_bin)
    eroded_candidates  <- antsImageClone(prob_bin)
  } else {
    labeled_candidates <- oro2ants(
      label_lesion(
        prob_map = flames_prob,
        bin_map = prob_bin,
        mincluster = mincluster
      )
    )
    eroded_candidates <- iMath(labeled_candidates, "GE", 1)
  }

  antsImageWrite(labeled_candidates, file.path(output_dir, "labeled_candidates.nii.gz"))
  antsImageWrite(eroded_candidates,  file.path(output_dir, "eroded_candidates.nii.gz"))

  if (return_images) {
    return(list(
      t1 = t1_final,
      flair = flair_final,
      epi = epi_final,
      phase = phase_final,
      prob_map = flames_prob,
      labeled_candidates = labeled_candidates,
      eroded_candidates = eroded_candidates
    ))
  } else {
    return(NULL)
  }
}