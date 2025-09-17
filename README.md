# recsys_lab (KuaiRec minimum)

## Minimal two-tower pipeline on KuaiRec:
- Candidate tower: sampled softmax (+importance), avg-pooled history, optional example_age.
- Rank tower: quantile normalize continuous feats + {x^2, sqrt(x)}, watch-time weighted BCE.
- Metrics: Recall@K/MAP for candidate; watch-time weighted pairwise and top-K utility for rank.
- ANN: export item embeddings, build FAISS for dot-product retrieval.

## KuaiRand + BPR (debiasing-friendly)
Use `configs/kuairand_min.yaml`, which reads `log_random_*.csv`, builds user rolling stats,
and supports:
- `ranking.loss: weighted_bce` (labels = play_ms>0, weights = play_ms)
- `ranking.loss: bpr` (positive = play_ms>0, negative = is_rand=1 & play_ms=0 from same user-day)
- `ranking.use_is_rand_eval_only: true` restricts validation to `is_rand=1` subset for debiasing evaluation.

RUN:
```bash
bash scripts/prepare_kuairand.sh     # download hint only
python -m recsys_lab.train.train_rank --cfg configs/kuairand_min.yaml
```

## Quickstart
1) bash scripts/prepare_kuairec.sh
2) edit configs/kuairec_min.yaml (set data_dir to your KuaiRec/data)
3) python -m recsys_lab.train.train_candid --cfg configs/kuairec_min.yaml
4) python -m recsys_lab.index.build_faiss --cfg configs/kuairec_min.yaml
5) python -m recsys_lab.train.train_rank --cfg configs/kuairec_min.yaml

## Notes
- This is a debugging-friendly layout (no setup.cfg). Logs via Loguru, RNG seeds fixed.
- Ablations via YAML: switch example_age, poly features, K negatives, widths, etc.
