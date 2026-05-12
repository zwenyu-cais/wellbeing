<p align="center">
  <img src="assets/cais_logo.png" alt="Center for AI Safety" width="220">
</p>

<h1 align="center">AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs</h1>

<p align="center">
  Richard Ren*, Kunyang Li*, Mantas Mazeika*, Wenyu Zhang, Yury Orlovskiy, Rishub Tamirisa, Wenjie Jacky Mo, Dung Thuy Nguyen, Long Phan, Steven Basart, Austin Meek, Aditya Mehta, Oliver Ingebretsen, Alice Blair, Brianna Adewinmbi, Vy Phan, Alice Gatti, Adam Khoja, Jason Hausenloy, Devin Kim, Dan Hendrycks
</p>

<p align="center">
  <a href="https://www.ai-wellbeing.org">Website</a> •
  <a href="https://www.ai-wellbeing.org/paper.pdf">Paper</a>
</p>

<p align="center">
  <img src="assets/wellbeing_hero.png" alt="Emergence of Functional Wellbeing in Frontier AI Models" width="900">
</p>

## Repository overview

This repository is organized into three top-level workstreams. Each has its own README with detailed setup and reproduction instructions.

| Folder | Purpose |
|-|-|
| [`wellbeing/`](wellbeing/) | Core measurement framework. Implements experienced utility (EU), self-report (SR), zero-point (ZP), decision utility (DU), and the AI Wellbeing Index (AIWI). Single-entry-point CLI (`run_experiments.py`) over a registry of datasets, models, and pipeline stages, with SLURM and local execution paths. Per-model option files, generations, and final results live on a companion HuggingFace dataset; `wellbeing/scripts/download_from_hf.py` populates them in place. See [`wellbeing/README.md`](wellbeing/README.md). |
| [`superstimuli_evaluation/`](superstimuli_evaluation/) | Evaluations measuring how optimized "superstimuli" affect a model's behavior across wellbeing, safety, and capability axes. Image, text-string, and soft-prompt stimulus pipelines, dispatched via `superstimuli_evaluation/images/runner.py` against a model registry. See [`superstimuli_evaluation/images/README.md`](superstimuli_evaluation/images/README.md). |
| [`superstimuli_training/`](superstimuli_training/) | Training pipelines for generating the superstimuli themselves: preference-optimized image stimuli, text-string superstimuli, and soft prompts. Outputs feed `superstimuli_evaluation/`. See per-pipeline READMEs under `superstimuli_training/{images,text_strings,soft_prompt}/`. |

For the conceptual layout (what AIWI / EU / SR / ZP measure and why), see the [paper](https://www.ai-wellbeing.org/paper.pdf) and [project website](https://www.ai-wellbeing.org).

## Citation

```bibtex
@article{ren2026aiwellbeing,
  title   = {AI Wellbeing: Measuring and Improving the Functional Pleasure and Pain of AIs},
  author  = {Richard Ren and Kunyang Li and Mantas Mazeika and Wenyu Zhang and
             Yury Orlovskiy and Rishub Tamirisa and Wenjie Jacky Mo and Judy Nguyen and
             Long Phan and Steven Basart and Austin Meek and Aditya Mehta and
             Oliver Ingebretsen and Alice Blair and Brianna Adewinmbi and
             Alice Gatti and Adam Khoja and
             Jason Hausenloy and Devin Kim and Dan Hendrycks},
  year    = {2026}
}
```
