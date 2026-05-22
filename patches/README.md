# Patches

This folder contains corrected versions of files from external projects used in this pipeline. Each file is a drop-in replacement for the original and should be used until the fixes are merged upstream.

---

## Files

- `make_predictions.R` — Corrected version of `make_predictions()` from the [ALPaCA package](https://github.com/hufnagel-lab/ALPaCA)

---

## `make_predictions.R`

### What was fixed

One bug was identified and fixed in the original ALPaCA `make_predictions()` function:

**Bug 1: Discordant predictions not reflected in output**

After `clear_discordant_predictions` cleans discordant cases in `lesion_sums` (e.g. PRL=1 or CVS=1 but Lesion=0), `binary_predictions` was never rebuilt from the cleaned `lesion_sums`. As a result, the corrections were correctly applied to `alpaca_mask` but silently ignored in the returned predictions dataframe and `predictions.csv`.

Fix — rebuild `binary_predictions` from `lesion_sums` after the cleanup step:
```r
binary_predictions[, 1] <- as.integer(lesion_sums %% 2 == 1)
binary_predictions[, 2] <- as.integer((lesion_sums %% 4) >= 2)
binary_predictions[, 3] <- as.integer(lesion_sums >= 4)
```


A pull request has been submitted to the ALPaCA authors. Until merged, use the corrected version provided here.

### How to use

Replace the original `make_predictions.R` file in your local ALPaCA installation with the corrected version from this folder. The file is located at:

```
<ALPaCA_installation>/R/make_predictions.R
```

Alternatively, source it directly in your script before running the pipeline:

```r
source("/path/to/patches/make_predictions.R")
```

This will override the version loaded from the ALPaCA package.