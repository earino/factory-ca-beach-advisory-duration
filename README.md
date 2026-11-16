---
license: other
license_name: upstream-terms-preserved
license_link: LICENSE.md
task_categories:
- tabular-classification
configs:
- config_name: default
  data_files:
  - split: train
    path: extract/public/train.csv
  - split: validation
    path: extract/public/eval.csv
  - split: test
    path: extract/private/holdout.csv
---

# Persistence of California beach water-quality postings: will a new posting still be in effect seven days later?

When a county health officer issues a new beach water-quality posting, will that posting remain in force for at least seven days?

Derived from California State Water Resources Control Board (Beach Monitoring Program, Division of Water Quality). This is not its official product.

Prediction time: The advisory issue date, day 0, taken from DateofAdvisory. Only the advisory class, issue date, beach identity, county, station inventory and earlier posting issue dates are used. For duration rates only earlier training-split records whose recorded removal date closed before this row and before 2016-01-01 are used. DateOpened, DURATION CALCULATED, the spatial extent fields and the derived duration are withheld.

Evaluation scope: `retrospective_research`. Public validation is development data. The downloadable labeled holdout is public; it is not cryptographically sealed. No final holdout benchmark score is claimed.

## Measured baseline

The existing runner was used unchanged on public validation. See `evaluation.json` for rows, controls, per-year results and limitations.

```json
{
  "full_model": {
    "auc": 0.8960809939048806,
    "brier": 0.11963301140244259
  },
  "majority_prior": {
    "auc": 0.5,
    "brier": 0.24869170914735844
  },
  "calendar_identity": {
    "auc": 0.4835145926947546,
    "brier": 0.25037123270507866
  }
}
```

## Limitations

- Current-vintage retrospective file: end dates reflect later edits and the extract was refreshed in October 2026, so it demonstrates what a model can learn from this register, not that the 7-day outcome was available to an operator in real time.
- Exact column names and value codings were not verified against the dictionary PDF within this selection pass; the build step must confirm them before the split is frozen, and the frozen boundaries assume the file reaches back to at least the 2005-2015 train window.
- No rainfall, tide, oceanographic or sampling-result covariates are used; weather is a major driver of posting persistence, so the model is deliberately limited to register-available fields.
- Reporting is by county health officers with heterogeneous protocols, so county and beach effects partly encode administrative practice rather than environmental persistence.
- The 7-day threshold is a policy choice, not a natural break; results should be reported against at least one alternative threshold (for example 3 and 14 days) to show sensitivity.
- Beaches outside participating jurisdictions and advisory episodes never entered into the state register are absent, so the population is the reporting frame, not all California beaches.
- Underlying episodes are submitted by county agencies; the ODC-By grant is the publisher's over this dataset and does not relicense any upstream county record on other terms, and attribution must travel with any redistribution. Our own contribution may be released CC0, but that does not alter upstream rights.
- Current-vintage retrospective file: removal dates reflect later edits and the extract was refreshed in 2026, so this shows what a model can learn from the register, not that the seven-day outcome was known to an operator at issuance.
- No full-row availability ledger was observed; exact header spellings, date fill rates and historical field availability remain assumed and are confirmed or refuted by the builder checks at run time.
- AdvisoryType does not expose a separable permanent-posting code in the observed publisher data, so permanent postings folded into the posting class cannot be removed by class alone; the population wording was corrected rather than pretending an unobservable exclusion.
- Repeated rows sharing a beach, issue date and class are removed deterministically by keeping the earliest source row; where a duplicate group carries differing removal dates the retained label is one of several recorded outcomes.
- Station matching uses an identifier tier, a beach-name-plus-county tier and a beach-name-only tier, so same-named beaches in different jurisdictions can over-merge and inflate station counts.
- Duration is a whole-calendar-day difference, so a posting whose removal date equals the issue date plus exactly seven days counts as persistent; sub-day timing is not representable.
- No rainfall, tide, oceanographic or sampling-result covariates are used; weather is a major driver of posting persistence, so the model is limited to register-available fields.
- The seven-day threshold is a policy choice, not a natural break; sensitivity to three and fourteen days is not evaluated here.
- Beaches outside participating jurisdictions and episodes never entered into the state register are absent, so the population is the reporting frame, not all California beaches.
- No uncertainty interval; observed differences do not establish universal headroom

## Loading and reproduction

Verify all downloaded files with `verify_dataset.py`. CSV splits live under `extract/`; `meta.json` declares the target and identifiers. `MANIFEST.json` binds data, source snapshot, code and measured evidence.

To reconstruct without network source drift, extract `source-snapshot.tar.gz` into a new cache directory, install `reproduction/requirements.txt` in an isolated Python environment, then run `python reproduction/dataset_run.py --output NEW_EMPTY_DIRECTORY --source-cache CACHE_DIRECTORY`. Compare the qualification artifact digest with this manifest. This performs full validation and baseline evaluation and may need worker resources.

See `LICENSE.md` for source terms and preserved attribution.
