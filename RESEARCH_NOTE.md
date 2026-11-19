# Does a Weight Edit Restore an Ability or Create One?

Candidate Assignment — Weight-Space Programming for SE.LLMA

## Hypothesis

The article argues that weight editing can remove an obstacle that prevents a model from expressing an existing capability, but cannot create a capability the model has never learned. This study tests that claim in the context of Swedish language modelling.

The distinction matters for SE.LLMA: if editing can restore suppressed capabilities, it may offer a lightweight way to repair existing models. If it can create new capabilities, it might also reduce the need for additional training.

**Hypothesis:** Targeted weight editing can restore an existing capability when it is masked by a concentrated failure, but cannot reliably create a capability the model has not learned.

The hypothesis is supported if a Swedish-trained model recovers on held-out Swedish data while an English-trained model does not improve meaningfully. A targeted edit should also outperform random editing and decoding-only changes without substantially degrading English performance.

## Experiment

I implemented a small character-level language model in NumPy to test this hypothesis under controlled, CPU-only conditions. The model has four layers, a hidden dimension of 64 and a context length of eight characters. Experiments were repeated across five random seeds.

Two models were trained using synthetic sentence templates: Model A on Swedish text and Model B on English text. Both were evaluated on the same held-out Swedish dataset. Swedish-language ability was operationally defined as next-character prediction accuracy on this dataset.

To simulate a failure that suppresses an existing capability, I injected a known rank-one perturbation into the output weight matrix `W2`:

`D = alpha * u (x) v`

Here `u` is the mean hidden-state direction estimated from development data and `v` is chosen to increase the probability of a designated trap character, `a`. The scaling factor `alpha` was calibrated on development data so that greedy generation produces the trap character at roughly half of the output positions, causing repetitive output.

I compared four editing strategies: an oracle edit that subtracts the exact injected perturbation; a weight-only heuristic using no forward-pass feedback; and feedback-guided edits using five or twenty rounds of development-set feedback.

Three controls were included: a random edit with a matched norm, conventional fine-tuning, and a repetition penalty applied only during decoding. The latter tests whether improved surface fluency can be achieved without repairing the underlying model.

Training, development and evaluation data were kept separate. Editing decisions used development data only, while final performance was measured on the held-out evaluation set. English accuracy served as a regression check.

## Results

Swedish held-out next-character accuracy (mean ± standard deviation, five seeds):

| Condition | A: Swedish-trained | B: English-trained |
|---|---|---|
| C0 base | 0.8365 ± 0.0017 | 0.1641 ± 0.0075 |
| C1 injected, no edit | 0.4963 ± 0.0773 | 0.1314 ± 0.0149 |
| C2 oracle | 0.8365 ± 0.0017 | 0.1641 ± 0.0075 |
| C2 weight-only, b=0 | 0.4963 ± 0.0773 | 0.1314 ± 0.0149 |
| C2 feedback, b=5 | 0.8004 ± 0.0048 | 0.1658 ± 0.0066 |
| C2 feedback, b=20 | 0.8016 ± 0.0059 | 0.1661 ± 0.0069 |
| C3 random | 0.4930 ± 0.0826 | 0.1295 ± 0.0132 |
| C5 fine-tuning | 0.8354 ± 0.0031 | 0.1692 ± 0.0087 |

The injected failure reduced Model A's accuracy from 0.84 to 0.50. The oracle edit fully restored its original performance, while random editing did not. Feedback-guided editing recovered most, but not all, of the lost accuracy. In contrast, Model B showed no meaningful improvement: the best practical edit produced only a +0.0020 gain.

The weight-only heuristic failed to recover performance. The decoding-only control reduced repetition but did not restore prediction accuracy. Full results, including additional metrics and configurations, are available in `RESULTS.md`.

## Interpretation and Recommendation

The results support the distinction between repairing an existing capability and creating a new one. Under controlled conditions, targeted editing recovered a capability that had been learned but suppressed by a known failure. It did not meaningfully improve performance when the capability had not been learned.

However, the results do not demonstrate that an LLM can independently reason about and modify unfamiliar network weights. The strongest practical results came from feedback-guided editing, while the weight-only heuristic was ineffective. The experiment therefore tests the effects of weight editing, not the full weight-space programming method described in the article.

For SE.LLMA, I recommend limited further investigation of weight editing as a potential repair mechanism for identifiable failures such as repetitive generation, looping or language mixing. It should complement, rather than replace, data curation, continued pre-training and supervised fine-tuning, which remain the primary approaches for acquiring Swedish-language capabilities.

Any future evaluation should measure actual language ability, include regression tests in other languages, and distinguish genuine capability restoration from improvements in surface fluency.

## Limitations and Next Step

This is a proof-of-concept using a small model and synthetic text, not a demonstration on a production-scale Swedish language model. The injected failure is a single, known rank-one perturbation; real model failures may be more distributed and difficult to identify. Next-character accuracy also provides only a narrow measure of language ability.

Exploratory tests with Qwen2.5-0.5B and SmolLM2-135M were inconclusive: neither model provided a reliable ability-present versus ability-absent distinction on the synthetic evaluation. Therefore, the claim about the inability to create an absent capability rests primarily on the controlled experiment.

The most informative next step would be to repeat the comparison using an open-weight multilingual model and a genuine Swedish benchmark, with a reliable probe to establish whether the target capability exists before editing.
