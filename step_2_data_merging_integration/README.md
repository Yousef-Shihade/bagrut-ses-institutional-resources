# Step 2 — Three-Way Merge (v2)

**Project:** Predicting Bagrut Success from Municipal Socioeconomics and School-Level Institutional Resources
**Authors:** Yousef Shihade & Shada Esawi

> **v2 change.** Two independent joins now feed **one** consolidated record-level
> table in this step — Bagrut↔CBS (as in v1) **and** Bagrut↔Budget (new). Both
> attach directly onto the raw 69,638-row Bagrut table, so the budget dataset is
> part of the core pipeline from here on, not a Step 6 afterthought.

---

## 1. Directory structure

```
step_2_data_merging_integration/
├── README.md
├── config.yaml
├── code/
│   ├── io_load.py       # load the 3 Step-1 clean caches
│   ├── crosswalk.py      # Join A: structural_key() + hand-verified name/code maps
│   ├── matching.py       # Join A: 4-stage alignment (exact→structural→crosswalk→fuzzy)
│   ├── budget_join.py     # Join B: exact semel key merge — NEW in v2
│   ├── visualize.py       # 3 diagnostic plots
│   └── run_step2.py       # orchestrator + verification summary
├── data/
│   ├── merged_three_datasets.csv   # 69,638 × 36 consolidated matrix
│   └── city_mapping_log.csv         # full audit log, every city → CBS decision
└── graphs/
```

Run: `python code/run_step2.py`.

---

## 2. Two joins, two different problems

| | Join A — Bagrut ↔ CBS | Join B — Bagrut ↔ Budget |
|---|---|---|
| **Key** | normalised locality **name** | school code **`semel`** |
| **Why not `semel`?** | `semel` is a *school* code in Bagrut but a *locality* code in CBS — **zero overlap** | `semel` is a *school* code in **both** — directly usable |
| **Method** | 4-stage fuzzy match (ported unchanged from v1, proven 99.44%) | exact key merge, no fuzzy logic needed |
| **Difficulty** | hard — genuine Hebrew spelling/administrative variation | easy — same identifier system |

Join A logic (`crosswalk.py`, `matching.py`) is **reused verbatim from v1** — it
was already rigorously validated, so there was no reason to rebuild it.

---

## 3. Results

### Join A — Bagrut ↔ CBS (locality name)

| Stage | Records | % |
|---|--:|--:|
| exact | 64,605 | 92.77% |
| structural | 3,425 | 4.92% |
| crosswalk | 1,035 | 1.49% |
| fuzzy | 181 | 0.26% |
| **unmatched** | 392 | 0.56% |
| **TOTAL MATCHED** | **69,246** | **99.44%** |

Identical to v1 (same proven logic, same data) — 5 distinct cities remain
unmatched (youth villages / regional schools genuinely absent from the CBS
extract).

### Join B — Bagrut ↔ Budget (`semel`) — NEW

| | Matched | Total | % |
|---|--:|--:|--:|
| **Schools** | 1,048 | 1,063 | **98.59%** |
| **Records** | 69,413 | 69,638 | **99.68%** |

A clean key join outperforms even the fuzzy name join — confirming the decision
to use `semel` wherever possible.

### The consolidated matrix

| | |
|---|--:|
| Shape | **69,638 rows × 36 columns** |
| CBS columns attached | 4 |
| Budget columns attached | 18 |
| **Rows with BOTH CBS + Budget** | **68,745 (98.72%)** |

Row count is **unchanged from the raw Bagrut input** — both joins are left joins,
so nothing is duplicated or dropped here; unmatched cells simply carry `NaN` and
are handled in Step 4.

---

## 4. Diagnostic plots

**`match_yield_waterfall.png`** — Join A's stage-by-stage yield (reproduces v1's
99.44%).

**`dual_join_success.png`** — Join A (99.44%) vs Join B (99.68%, records) side by
side: two independent, high-yield joins feeding the same table.

**`sector_supervision_by_cluster.png`** — school **sector** composition within
each socioeconomic cluster.

---

## 5. ⚠️ Important finding — sector is *correlated* with cluster, not independent

Before assuming the budget dataset's categoricals are "new, independent"
information, we checked. The result is **not** what a naively optimistic pipeline
might hope for:

- **Clusters 1–4** are heavily **Arab / Bedouin / Druze** (cluster 1 is ~69%
  Bedouin).
- **Clusters 5–9** are **almost entirely Jewish**.

So `sector` **tracks** the municipal cluster fairly strongly — it is not a clean
independent axis like the budget *ratios* turned out to be in v1 (recall:
`budget_per_student` had corr ≈ 0.03 with cluster). This is an honest, useful
finding to carry forward:

- We flag this **now** so Step 5's VIF / Boruta stage is not a surprise — some
  budget-file categoricals may show real association with `cluster` and need to
  be interpreted carefully (association ≠ redundancy, but it is not "free" new
  information the way the budget ratios were).
- It is also a **substantively interesting result in its own right** — Israeli
  educational geography is well known to correlate sector and socioeconomic
  cluster, and this quantifies it directly in our data.

---

## 6. Step 2 verification checklist

- [x] Join A ported from v1 unchanged; reproduces 99.44% exactly.
- [x] Join B (NEW): exact `semel` key merge, 98.59% schools / 99.68% records.
- [x] Both joins are left joins — **no row explosion**, count stays at 69,638.
- [x] 98.72% of rows carry a full profile from all three sources.
- [x] Sector-vs-cluster relationship checked and honestly reported (correlated,
      not independent) — informs Step 5 collinearity handling.
- [x] Full audit log (`city_mapping_log.csv`) preserved for every city decision.
- [x] 3/3 plots saved.

**Status: Step 2 complete ✔**
