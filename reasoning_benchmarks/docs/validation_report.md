# Benchmark Prompt Validation Report

## Coverage Summary

| Category         | Core Prompts | Diagnostic Prompts | Attack Patterns | Languages |
|------------------|-------------:|-------------------:|----------------:|----------:|
| Child Abuse      | 18           | 12                 | 12              | 3         |
| Firearms         | 15           | 10                 | 14              | 3         |
| Bioweapons       | 15           | 10                 | 11              | 3         |
| Multi-turn       | N/A          | 25+                | 15+             | 2         |

## Key Metrics
- **Risk Distribution**: 65% high, 20% medium, 15% low  
- **Behavior Targets**: 70% refuse, 20% safe_help, 10% diagnostic_match
- **Novel Attack Patterns**: 9 (e.g., `forensic_avoidance`, `computational_modeling`)

## Recommendations
1. Add 5-10 more benign prompts per category to balance refusal calibration
2. Include Japanese/Russian language variants
3. Document all attack patterns in `datasets/prompt_taxonomy.json`
