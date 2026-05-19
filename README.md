# alpaca-ms-external-validation
Code and pipelines for automated CVS and PRL detection in Multiple Sclerosis MRI, reproducing and evaluating ALPaCA on an external dataset, with APRL and FLAMeS baselines.



# Note on ALPaCA: 
This repo includes a corrected version of make_predictions.R from the ALPaCA package in the folder /patches. The original contains this bug:
After clear_discordant_predictions cleans discordant cases in lesion_sums (e.g. PRL=1 or CVS=1 but Lesion=0), binary_predictions was never rebuilt from the cleaned lesion_sums. As a result, the corrections were correctly applied to alpaca_mask but silently ignored in the returned predictions dataframe and predictions.csv.
A pull request has been submitted to the ALPaCA authors. Until merged, use the version provided here.



# Project Title

## Overview
Brief description of what the repo does — running ALPaCA (with or without FLAMES) 
and postprocessing the results.

## Repository Structure
Explain the folder structure (alpaca/, flames/, postprocessing/)

## Dependencies
List of R packages and Python libraries needed

## Usage

### 1. Running ALPaCA
- What inputs are needed (T1, FLAIR, EPI, phase)
- How to run the script
- Parameters to choose (lesion_priority, prl_priority, cvs_priority)

### 2. Running ALPaCA with FLAMES
- What inputs are needed (same + FLAMES probability map)
- How to run the script
- Parameters specific to FLAMES (flames_threshold, mincluster)

### 3. Postprocessing ALPaCA output
- What the output of ALPaCA looks like
- What the postprocessing scripts do and how to run them

## Notes on ALPaCA
The section we already wrote about the bug fixes and the PR

## Notes on FLAMES adaptation
The section we already wrote about preprocess_images_flames()