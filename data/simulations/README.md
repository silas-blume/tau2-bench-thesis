# Simulation Evidence Corpus

This directory intentionally retains the raw evaluation outputs used during the secure-airline study. New simulation output remains ignored by Git so local runs do not create routine repository noise; a result is added deliberately when it is evidence worth preserving.

## Result Families

| Family | Meaning |
| --- | --- |
| `DEC*` | Declare-policy experiment matrix. |
| `DCR*` | DCR-policy experiment matrix. |
| `U_*` | Unconstrained or baseline comparison runs. |
| `full-*`, `sec-full-*`, `*-adv-*` | Larger regular and adversarial evaluations. |
| `secure_airline_*`, `soft-*` | Secure-agent and deliberately policy-softened-agent runs. |
| `dcr1-*`, `dcr2-*`, `r35_r39_fix*` | DCR loading, resolver, policy-refinement, and regression checks. |
| `smoke*`, `debug*`, `*_test*`, `*_backup*` | Preserved diagnostics and intermediate reproductions. |

The short names are historical experiment labels. Read each file's serialized configuration before comparing metrics; policy choice, model, task set, trial count, and soft-agent mode all affect the result.

## Creating New Output

Run results with an explicit name so tau2 writes `data/simulations/<name>.json`:

```bash
tau2 run \
  --domain airline \
  --agent secure_airline_agent \
  --sec-file policy_v3.yaml \
  --task-set adv \
  --num-trials 1 \
  --save-to my_secure_adversarial_run
```

See the root [README](../../README.md) for setup and the current DCR-YAML runtime status. The CSV file is an accompanying manual experiment summary; the JSON files remain the primary trajectory-level evidence.
