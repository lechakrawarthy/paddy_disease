# Paddy disease representation with a convolutional autoencoder (project 29)

Setup: `pip install -r requirements.txt`, then download the Kaggle *Paddy Doctor* data so you have
`data/train.csv` and `data/train_images/<label>/<image_id>`.

Run from `src/`:

```
python 01_eda_split.py                      # EDA, dHash dedup, grouped stratified 70/15/15 split (seed 42)
python 02_baseline.py                       # B1: PCA128 + LogReg (val only)
python 03_train_ae.py                       # Stage 1, defaults: z=128, batch 64, lr 0.01, momentum 0
python 04_train_head.py --ae cae_z128_b64_lr0.01_m0.0     # E1 (val only)
```

Tuning, one factor at a time on validation (all rows land in `runs/experiment_log.csv`):

```
python 03_train_ae.py --lr 0.1 ; python 03_train_ae.py --lr 0.001      # E3
python 03_train_ae.py --batch 32 ; python 03_train_ae.py --batch 128   # E3
python 03_train_ae.py --latent 64 ; python 03_train_ae.py --latent 256 # E2
python 03_train_ae.py --momentum 0.9 --note "momentum ablation"
python 04_train_head.py --ae <run> --finetune --lr 0.001 --enc_lr 0.001  # E4 (optional)
```

Final, once configs are chosen (this is the only time the test split is used):

```
python 02_baseline.py --test
python 04_train_head.py --ae <ae_run> --test
python 05_final_report.py --ae <ae_run> --head <head_run>   # figs in reports/figs, summary in reports/final_results.txt
python 06_build_pdf.py                                      # reports/report.md -> standalone PDF
python 07_build_docx.py                                     # fills the GITAM ANN template -> .docx
```

Then open the `.docx` in Word: right-click the table of contents, choose **Update Field**, and save or export it as PDF.

Notes / deviations to state in the report:
- Near-duplicates are clustered (64-bit dHash, Hamming <= 2, tune with `--hamming`) and each cluster stays in one split rather than being deleted.
- Stage 2 trains on precomputed, un-augmented latents (encoder frozen), standardised with train statistics.
- Only the images are 128x128 cached in `artifacts/`; B1 downsamples these to 64x64.
