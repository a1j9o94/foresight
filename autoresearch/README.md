# Autoresearch for Foresight

Autonomous AI agent research loop inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## The Idea

Foresight hit a wall at Phase 3: 7 prediction architectures failed to beat simply copying the input frame. Instead of manually iterating, we apply Karpathy's autoresearch pattern:

1. Point an AI agent (Claude Code, Codex, etc.) at `program.md`
2. Agent modifies prediction architecture in a fixed file
3. Trains for 5 minutes, evaluates against copy baseline
4. Keeps improvements, discards failures
5. Repeats autonomously — ~12 experiments/hour, ~100 overnight

## Files

- `program.md` — Agent instructions (the "research org code")
- `experiments.jsonl` — Structured log of all experiments (created by agent)

## Running

```bash
# Point your coding agent at the program
# In Claude Code, Codex, etc:
"Read autoresearch/program.md and start experimenting."
```

## Why This Might Work

The original 7 experiments were designed by humans with specific hypotheses. Autoresearch explores the space more broadly — architectural variations, hyperparameter sweeps, and novel combinations a human researcher might not try. At 100 experiments per night, the search space coverage is dramatically higher.

## Connection to Black Cygnets

One research direction (see program.md #5) explicitly uses hallucination as a feature rather than a bug — generating candidate futures through controlled confabulation and filtering for plausibility. See the [Black Cygnets thesis](https://github.com/a1j9o94/obsidian-vault) for background.
