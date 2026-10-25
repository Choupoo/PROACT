# Detection Methods and Experimental Setup

These are complete English chapter drafts for the existing article-based
thesis. The standalone `main.tex` compiles them as Sections 4 and 5, with the
same citation keys as the current thesis bibliography. This directory does
not modify the actual thesis sources or any experiment code.

## Files to integrate

- `chapters/methods.tex`: frozen predicted-class features, source logistic
  regression, historical sample/count calibration, MMD and Rank.
- `chapters/experimental_setup.tex`: completed CIFAR-100 configuration,
  information boundaries, data splits, attacks, ablations, simulations,
  metrics, reproducibility and the registered ImageNet extension.
- `references_methods_setup.bib`: two additional references (CIFAR and ResNet).
- `references.bib`: an unchanged copy of the current thesis bibliography,
  included only to make this standalone project compile.

Replace the contents of the existing `chapters/methods.tex` and
`chapters/experimental_setup.tex` with these chapter sources. Do not add
another `\section` around them: each file already supplies its own heading.
The current thesis already contains the two required `\input` statements.
Keep the existing document class, cover, background and remaining chapters.

Copy `references_methods_setup.bib` beside the thesis's existing
`references.bib`, and change its bibliography line to:

```latex
\bibliography{references,references_methods_setup}
```

Alternatively, append the two new entries to the original bibliography and
retain `\bibliography{references}`. Do not add duplicate citation keys.
The copied `references.bib` in this review project need not replace the thesis
bibliography. The main thesis already loads all packages required here.

The chapter root comments point to `../main.tex` for this review project. After
integration, they may be changed to `../thesis.tex`; comments do not affect
compilation through the main document.

## Evidence used

The completed-run description was checked against the supplied
`/Users/zhangpan/Downloads/proact-result/advisor_final_v1/` directory,
including its plan, summary, all 15 run-state files, frozen fit/calibration
records and environment record. The summary is complete with no missing
cells: all 15 cells have 36 completed stages. There are five implementations,
three seeds (6, 7, 8) and four target attack configurations.

Method equations were checked against `extract_features.py`,
`transfer_detector.py`, `train_detector.py`, `advisor_detector.py`,
`calibration.py`, `dataset_detector.py`, `rank_reference.py` and
`unsupervised.py`. Training and perturbation settings were checked against the
registered commands and the upstream scripts they execute. The archived
AFEC-EWC and ANCL-EWC source hashes match the inspected upstream files.

Primary literature was consulted for BrainWash task counts, MMD, random Fourier
features, the bootstrap motivation, CIFAR and residual networks. The full
reference details are in the two bibliography files. The bootstrap equations
describe this implementation, not a claimed exact reproduction of a theorem.

## Boundaries retained in the text

- The full supervised input has 84 coordinates; Rank/MMD have ten descriptors.
- Image classes, source poisoning labels and incoming poisoning labels have
  different roles. Detector fitting is not fully unsupervised.
- Held-out detector originals are still visible to attack optimisation.
- Task 4 is a diagnostic for a prespecified rule in the completed expansion,
  not a target-driven method-selection step.
- Historical count calibration does not guarantee the target clean error rate.
- Repeated bags are not independent datasets, attacks or training seeds.
- AFEC-EWC and ANCL-EWC are qualified repository-implementation results because
  of the inspected auxiliary-model/optimiser issue.
- ImageNet configuration is registered, but its GPU results are not claimed as
  completed. Refresh that status only after checking actual outputs.
- Full-task forgetting controls do not establish a benefit from detector-based
  filtering or quantify the damage from a 150-image detector bag.

No numerical target-performance findings are invented or moved into these
chapters. Performance interpretation belongs in Results and Discussion.

## Before sending to the supervisor

Read both chapters and check that their scope matches the thesis argument.
Retain the implementation qualification when presenting the auxiliary methods.
As the remaining experiments finish, update only verified completion status
and put their numerical findings in Results and Discussion. A complete draft
still requires the author's technical review; editing cannot certify an
AI-detector score or substitute for understanding the method.
