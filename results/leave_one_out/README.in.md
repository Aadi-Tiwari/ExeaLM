# Which circuit types help or hurt the hybrid's starting menu? Leave one out at 13M ([#59](https://github.com/exea-labs/ExeaLM/issues/59))

This file is written by `python experiments/leave_one_out_table.py --write` from the result files in this folder.
Edit `README.in.md`, not `README.md`.

**Question.** Leaving rotary attention (`RP`) out of the hybrid's starting circuits removed its weak seeds
(`results/funding/README.md`, sections 1 and 3). Is any other circuit type a net cost, or a net gain, in the starting menu?

**Setup.** The 13M hybrid of the `small-tonn` preset with the default menu (`HYBRID` in `exealm/config.py`:
`C` 4, `G` 0.25, `D` 0.25, `RP` 0, gated attention and QK-norm, attention layers 1, 2, 4 and 5), 3,000 steps of
32 x 512 tokens in 32-bit precision, learning rate 2e-3, 50% English, 30% Japanese and 20% code. For each type
`T` in `D I GI M MI MIM EP G SM OS` the starting weight of `T` is set to 0; the control sets nothing to 0. Each of the
11 configurations ran seeds 0, 1 and 2. The score is the mean over the four held-out
sets (English, Japanese Wikipedia, Japanese web, code) of bits per byte; lower is better.

**Data and hardware.** TBD: the packs, their sizes and sources, the GPU, and the PyTorch version.
The control was run again here, on the same data, code and GPU as the other configurations, so the table compares
like with like. The absolute numbers are not comparable with `results/funding` unless that file used the same held-out text.

**Reading.** A change against the control smaller than about 0.005 bits per byte is within seed noise at this size and
is called noise. A type whose removal helps on all three seeds by more than that hurts the menu; a type whose removal
hurts on all three seeds by more than that helps it.

{{TABLES}}

**What it shows.** TBD after the runs finish.

**Caveats.** TBD.

**Reproduce.**

```bash
python experiments/leave_one_out_data.py --fetch --raw data/raw --tokenizer results/tokenizer/bpe32k --out data/tokpack/bpe32k
python experiments/leave_one_out.py                    # resumable; finished runs are skipped
python experiments/leave_one_out_table.py --write
```
