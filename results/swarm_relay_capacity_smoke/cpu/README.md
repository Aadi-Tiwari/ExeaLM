# Relay capacity: board vs hidden-key weight-alter

**Setup.** 2 agents, 8 swarms per batch, 8 tokens per chunk, 20 steps on `cpu`, held-out batch seed 10000 (4 batches).
Chance accuracy is 1.56%. Loss is the last training batch's cross-entropy in nats. Held-out cross-entropy is in bits per answer token.

| Facts per holder | Seed | Memory | Tokens per agent | Parameters | Step | Train loss | Shared accuracy | Cut accuracy | Shared bits | Cut bits | JSON |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 4 | 0 | AG | 64 | 814,320 | 20 | 5.4656 | 0.78% | 0.00% | 7.956 | 7.948 | [JSON](relay_seed0.json) |
| 4 | 0 | WA | 64 | 844,216 | 20 | 5.5407 | 0.00% | 0.00% | 7.939 | 7.941 | [JSON](relay_wa_hkey_seed0.json) |
| 4 | 1 | AG | 64 | 814,320 | 20 | 5.4745 | 2.34% | 0.00% | 7.872 | 7.869 | [JSON](relay_seed1.json) |
| 4 | 1 | WA | 64 | 844,216 | 20 | 5.5575 | 0.78% | 0.78% | 7.872 | 7.873 | [JSON](relay_wa_hkey_seed1.json) |
| 64 | 0 | AG | 1,024 | 814,320 | 20 | 5.3459 | 1.22% | 1.07% | 7.679 | 7.693 | [JSON](relay_a2_f64_seed0.json) |
| 64 | 0 | WA | 1,024 | 844,216 | 20 | 5.3314 | 1.42% | 1.51% | 7.653 | 7.656 | [JSON](relay_a2_f64_wa_hkey_seed0.json) |
| 64 | 1 | AG | 1,024 | 814,320 | 20 | 5.3078 | 2.05% | 1.56% | 7.623 | 7.628 | [JSON](relay_a2_f64_seed1.json) |
| 64 | 1 | WA | 1,024 | 844,216 | 20 | 5.2654 | 1.46% | 1.51% | 7.545 | 7.547 | [JSON](relay_a2_f64_wa_hkey_seed1.json) |
