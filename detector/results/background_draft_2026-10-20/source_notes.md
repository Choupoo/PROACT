# Source and scope notes

Checked on 20 October 2026. This file is an editorial record, not thesis prose.

## Sources used

| Citation key | Primary publication or author manuscript | Role in the chapter |
| --- | --- | --- |
| parisi2019continual | https://doi.org/10.1016/j.neunet.2019.01.012 | CL overview, replay / regularisation / architecture families |
| vandeven2019scenarios | https://arxiv.org/abs/1904.07734 | Task-, domain- and class-incremental scenarios |
| lopezpaz2017gem | https://proceedings.neurips.cc/paper_files/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html | Accuracy matrix and backward transfer |
| kirkpatrick2017ewc | https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101/ | EWC, Fisher diagonal and local posterior approximation |
| aljundi2018mas | https://arxiv.org/abs/1711.09601 | Output sensitivity and unlabelled importance estimation |
| chaudhry2018rwalk | https://arxiv.org/abs/1801.10112 | RWalk, Fisher/path information and intransigence |
| wang2021afec | https://proceedings.neurips.cc/paper/2021/hash/bc6dc48b743dc5d013b1abaebd2faed2-Abstract.html | Expansion-convergence and negative transfer |
| kim2023ancl | https://arxiv.org/abs/2303.09483 | Auxiliary plasticity reference and stability-plasticity interpolation |
| biggio2012poisoning | https://arxiv.org/abs/1206.6389 | Training-time poisoning and optimisation through learning |
| abbasi2024brainwash | https://arxiv.org/html/2311.11995v2 | Threat model, inversion, reckless and cautious settings; conference metadata also checked against CVPR record |
| tran2018spectral | https://proceedings.neurips.cc/paper_files/paper/2018/hash/280cf18baf4311c92aa5a042336587d3-Abstract.html | Related backdoor detection, not a claimed BrainWash baseline |
| proactsoftware | https://github.com/Choupoo/PROACT | Software provenance; local README and current source inspected |
| yin2020deepinversion | https://openaccess.thecvf.com/content_CVPR_2020/html/Yin_Dreaming_to_Distill_Data-Free_Knowledge_Transfer_via_DeepInversion_CVPR_2020_paper.html | Fixed-teacher input synthesis and BatchNorm constraints |
| smith2021dreaming | https://openaccess.thecvf.com/content/ICCV2021/html/Smith_Always_Be_Dreaming_A_New_Approach_for_Data-Free_Class-Incremental_Learning_ICCV_2021_paper.html | Limits of naive inversion replay; a different task from detection |
| koh2017influence | https://proceedings.mlr.press/v70/koh17a.html | Influence functions contrasted with simple gradient descriptors |
| hendrycks2017baseline | https://arxiv.org/abs/1610.02136 | Maximum-softmax OOD baseline; not a poisoning guarantee |
| guo2017calibration | https://proceedings.mlr.press/v70/guo17a.html | Probability calibration, distinct from threshold calibration |
| fawcett2006roc | https://doi.org/10.1016/j.patrec.2005.10.010 | Ranking versus operating-point evaluation |
| clopper1934binomial | https://doi.org/10.1093/biomet/26.4.404 | Binomial confidence limits; original paper also read from an archived PDF |
| gretton2012mmd | https://www.jmlr.org/papers/v13/gretton12a.html | MMD and population distribution comparison |
| rahimi2007random | https://proceedings.neurips.cc/paper_files/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html | Finite-dimensional approximation to shift-invariant kernels |
| kendall1938rank | https://doi.org/10.1093/biomet/30.1-2.81 | Rank-correlation provenance; publisher metadata verified |
| efron1981jackknife | https://doi.org/10.1214/aos/1176345462 | Jackknife background; author-institution record checked |
| chen2018bootstrap | https://arxiv.org/abs/1610.00032 | Nondegenerate U-statistic bootstrap framework, not a theorem for this detector |

Some CVF pages returned access errors; author manuscripts or indexed official
conference records were used instead. Verification covered relevant statements,
formulae and metadata, not a line-by-line reading of every paper. Classical
formulae were also checked algebraically.

## Deliberate boundaries

- No invented performance results, claims of being the first detector, or claims
  of universal robustness are included.
- A shared task identity is not an image-class label. Source clean/poison labels
  are not incoming image-class labels either.
- The BWT equation uses zero-based indexing and the same immediate-post-task
  diagonal baseline as the implementation.
- The generic quadratic penalty is not presented as the exact objective of all
  five CL methods. The EWC expression is explicitly a two-task form.
- The first-order gradient calculation is an explanatory Taylor expansion. It
  neither proves detector effectiveness nor reproduces a full poisoned update.
- Historical class labels used for supervised CL training or inversion are not
  described as forbidden incoming labels.
- The population MMD equation is background. The implemented finite-feature and
  permutation procedure belongs in Methods, with its own assumptions.
- Rank uses tau-a, not tau-b. Kendall correlations are not full copula equality.
  Bootstrap p-values are approximate and are not poisoning probabilities.
- Chen (2018) is cited as related statistical groundwork, not as certification of
  the implemented two-sample, multi-reference procedure or inversion sample IID.
- A Clopper--Pearson interval is not a task-shift guarantee. The formal detector's
  historical maximum and error-budget split belong in Methods.
- PROACT is cited as software only. No formal paper, authorship, venue or new
  defence contribution is invented.

## Local provenance used for project-specific statements

The existing thesis outline at
`/Users/zhangpan/Documents/Program/Materials/Thesis/thesis/thesis.tex`
was inspected, but not edited. It uses article/section structure and separates
Background, Problem Setting, Detection Methods, Experimental Setup and Results.

The detector implementation was checked at PROACT commit
`701b79e5582f62edfd3b16346284caee9b224431`. This draft adds only files beneath
`detector/results/background_draft_2026-10-20/`; no training source was changed.

Before the final submission, replace the software repository citation with a
stable release/commit and the official PROACT paper if the supervisor provides
one. Do not insert an assumed publication or author list.

## Compilation and presentation checks

- Local TeX Live 2024 / latexmk compilation completed successfully with BibTeX.
- Final preview: 10 A4 pages, comprising 8 pages of text and 2 of references.
- All 24 citation keys are used and defined; citation keys and 22 labels are unique.
- Final log: no undefined citations, undefined references, overfull boxes,
  underfull boxes or LaTeX/BibTeX warnings.
- Every final PDF page was rendered and visually checked, including equations
  and the reference list. Sparse spillover pages in the first layout were removed.
- The app's separate compiler timed out while downloading its initial TeX
  resources. The delivered PDF was built and checked with the installed local
  toolchain; this is not a claim that the app compiler succeeded.

The student should review the content and cited arguments before submission.
Rendering and citation checks establish document consistency, not institutional
compliance or the correctness of every substantive claim.

## Prose revision after chapter splitting

On 20 October 2026, the prose was revised to reduce repeated chapter signposting,
generic transitions and admonitory wording. Qualifications about label access,
task shift, inversion fidelity and approximate inference remain in context.
Displayed equation environments, labels and the complete set of 24 citation keys
were compared with the user's copied chapter and preserved. The bibliography
was not changed. A layout-only samepage group keeps the first probability-feature
paragraph together; the entropy expression is kept on one line.

Only the draft in this directory was edited. The separate thesis chapter at
`/Users/zhangpan/Documents/Program/Materials/Thesis/thesis/chapters/background.tex`
was inspected for comparison and left unchanged.
