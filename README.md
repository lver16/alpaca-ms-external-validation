# alpaca-ms-external-validation
Code and pipelines for automated CVS and PRL detection in Multiple Sclerosis MRI, reproducing and evaluating ALPaCA on an external dataset, with APRL and FLAMeS baselines.



# Note on ALPaCA: 
This repo includes a corrected version of make_predictions.R from the ALPaCA package in the folder /patches. The original contains this bug:
After clear_discordant_predictions cleans discordant cases in lesion_sums (e.g. PRL=1 or CVS=1 but Lesion=0), binary_predictions was never rebuilt from the cleaned lesion_sums. As a result, the corrections were correctly applied to alpaca_mask but silently ignored in the returned predictions dataframe and predictions.csv.
A pull request has been submitted to the ALPaCA authors. Until merged, use the version provided here.