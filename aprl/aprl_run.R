#!/usr/bin/env Rscript

.libPaths(c("/linux/luverheyen/R/aprl_lib_clean", .libPaths()))

suppressPackageStartupMessages({
  library(caret)
  library(prlr)
  library(oro.nifti)
})

out_root <- "/linux/luverheyen/data/processed_50_custom"

subject_list <- c(
  "sub-001_ses-01",
  "sub-003_ses-01",
  "sub-005_ses-01",
  "sub-008_ses-01",
  "sub-017_ses-01",
  "sub-022_ses-01",
  "sub-024_ses-01",
  "sub-029_ses-01",
  "sub-031_ses-01",
  "sub-032_ses-01",
  "sub-035_ses-01",
  "sub-036_ses-01",
  "sub-038_ses-01",
  "sub-046_ses-01",
  "sub-051_ses-01",
  "sub-054_ses-01",
  "sub-055_ses-01",
  "sub-056_ses-01",
  "sub-057_ses-01",
  "sub-060_ses-01",
  "sub-102_ses-01",
  "sub-104_ses-01"

)

for (subject_id in subject_list) {
  cat("\n========================================\n")
  cat("Running APRL for:", subject_id, "\n")
  cat("========================================\n")

  reg_out_dir  <- file.path(out_root, subject_id)
  aprl_out_dir <- file.path(reg_out_dir, "aprl")

  # Properly create APRL output folder for this subject
  if (!dir.exists(reg_out_dir)) {
    stop(paste("Missing subject folder:", reg_out_dir))
  }

  if (!dir.exists(aprl_out_dir)) {
    ok <- dir.create(aprl_out_dir, recursive = TRUE, showWarnings = TRUE)
    if (!ok && !dir.exists(aprl_out_dir)) {
      stop(paste("Could not create APRL output folder:", aprl_out_dir))
    }
  }

  if (!dir.exists(aprl_out_dir)) {
    stop(paste("APRL output folder does not exist:", aprl_out_dir))
  }

  probmap_path <- file.path(reg_out_dir, "prob.nii.gz")
  lesmask_path <- file.path(reg_out_dir, "prob_50_binary.nii.gz")
  phase_path   <- file.path(reg_out_dir, "phase_n4_bet_ws.nii.gz")

  for (p in c(probmap_path, lesmask_path, phase_path)) {
    if (!file.exists(p)) stop(paste("Missing file:", p))
  }

  # Use oro.nifti directly — no neurobase/fslr needed
  probmap <- oro.nifti::readNIfTI(probmap_path, reorient = FALSE)
  lesmask <- oro.nifti::readNIfTI(lesmask_path, reorient = FALSE)

  findprls_out <- tryCatch({
    prlr::findprls(
      probmap   = probmap,
      lesmask   = lesmask,
      phasefile = phase_path,
      disc      = TRUE
    )
  }, error = function(e) {
    cat("ERROR for", subject_id, ":", conditionMessage(e), "\n")
    NULL
  })

  if (!is.null(findprls_out)) {
    # Re-check folder just before saving, for safety
    if (!dir.exists(aprl_out_dir)) {
      ok <- dir.create(aprl_out_dir, recursive = TRUE, showWarnings = TRUE)
      if (!ok && !dir.exists(aprl_out_dir)) {
        stop(paste("Could not create APRL output folder before saving:", aprl_out_dir))
      }
    }

    saveRDS(findprls_out, file.path(aprl_out_dir, "findprls_out.RDS"))

    oro.nifti::writeNIfTI(findprls_out$leslabels,
                          file.path(aprl_out_dir, "aprl_leslabels"))

    write.csv(as.data.frame(findprls_out$ria.df),
              file.path(aprl_out_dir, "aprl_ria.csv"),
              row.names = FALSE)

    write.csv(as.data.frame(findprls_out$preds),
              file.path(aprl_out_dir, "aprl_preds.csv"),
              row.names = FALSE)

    cat("Done:", subject_id, "\n")
    cat("Outputs in:", aprl_out_dir, "\n")
  }
}