# Background and Related Work draft

This directory contains a chapter draft for the thesis, with a standalone
preview. It introduces the literature and concepts without reporting detector
performance.

## Files

| File | Purpose |
| --- | --- |
| background.tex | Chapter fragment using `\section`, compatible with the thesis's article structure |
| references.bib | The chapter's 24 cited references |
| main.tex | Standalone preview wrapper; not a replacement for the thesis main file |
| output/pdf/main.pdf | Compiled preview |
| source_notes.md | Citation provenance, scope and checks |

No training or detection source is part of this draft. Its files do not enter
the registered experiment's top-level Python fingerprint.

## Integrate into the thesis

Copy the chapter to `chapters/background.tex`, then use this line in the thesis
main file in place of the existing Background heading and placeholder outline:

```latex
\input{chapters/background}
```

The fragment already includes `\section{Background and Related Work}`; do not
add a second heading. Merge the bibliography by citation key into the existing
`references.bib`.

If not already configured, load natbib before hyperref and use one bibliography:

```latex
\usepackage[numbers,sort&compress]{natbib}
```

Before `\end{document}`:

```latex
\bibliographystyle{unsrtnat}
\bibliography{references}
```

Do not combine natbib/BibTeX and biblatex/Biber configurations. If only replacing
the prose with the revised version, the main file and bibliography need no
further change.

## Compile the preview

Run from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=output/pdf main.tex
```

An editor may instead compile `main.tex` as the preview's main document.

## Scope and review

Background covers literature, notation and concepts. The later chapters specify
label access and the threat model; the 84-feature detector, historical MAD rule
and Rank algorithm; experimental settings; and measured results and limitations.

AFEC/ANCL are introduced as published methods. Issues in their upstream
implementations should be discussed separately, not presented as properties of
the algorithms. The background equations use consistent notation rather than
reproducing every detail of each original method.

Before submission, check the chapter against the methods and be ready to explain
its equations and assumptions. PROACT is currently cited as software; add an
official publication if supplied, without inventing authorship or venue.

The 20 October prose revision retained displayed equations, labels and all 24
citation keys. The standalone draft and preview were updated; the separate thesis
directory was not automatically overwritten. Detailed source and compilation
records are in [source_notes.md](source_notes.md).
