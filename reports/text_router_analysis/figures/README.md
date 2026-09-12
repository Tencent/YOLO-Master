# Figure sources

The current method diagram and controlled-intervention figures are generated with
Python, NumPy and Matplotlib. From the repository root:

```bash
python scripts/plot_b1_method_figure.py
python scripts/plot_b1_publication_figures.py
```

The intervention plots read the retained CSV files in `../controlled_interventions`.
These commands generate SVG files only. The older
`plot_b1_controlled_interventions.py` retains the original report layout; use the
publication script above to regenerate the current PR figures.

The method diagram uses the existing project example photograph, retained here as
`method_input.png`. Feature-plane sizes schematically indicate spatial resolution;
they are not measured activations or drawn to scale. Output boxes, labels and
scores illustrate the output format, not measured model predictions. Gray modules
are frozen; the blue adapter is trainable and receives the neck's P5 output.

The current intervention figure has three panels: output-weight substitution,
fixed-expert AP, and dominant-expert use. The per-image figure compares F1 shares
with the resulting complete-prediction AP differences. Tied F1 does not imply
identical or correct predictions. All experimental values remain unchanged.
