# Benchmark Questions (50) — grounded on the seed-42 Volve KG

Machine-gradeable: every gold answer is computed from the KG (`data/benchmark/benchmark.jsonl`). Regenerate with `python -m src.eval.build_benchmark`.

Distribution: Identification 13, Risk 12, Recommendation 13, Explanation 12.
Grading: entity_list/scalar/boolean -> exact-match; ranking -> NDCG@5; explanation -> rubric (M3+).


## Identification

- `[ID]` **Q01** Which injectors are connected to producer F-12 via the CRM connectivity fit?  
  <sub>type=entity_list · gold=`['F-4', 'F-5']`</sub>
- `[ID]` **Q02** What is the CRM connectivity weight from injector F-5 to producer F-11?  
  <sub>type=scalar · gold=`0.4368`</sub>
- `[ID]` **Q03** What is the CRM time constant (tau, days) used for injector-producer pairs?  
  <sub>type=scalar · gold=`45.0`</sub>
- `[ID]` **Q04** Which injector has the highest total outgoing connectivity weight across the field?  
  <sub>type=scalar · gold=`F-4`</sub>
- `[ID]` **Q05** What is the dominant (highest-weight) supporting injector for producer F-11?  
  <sub>type=scalar · gold=`F-5`</sub>
- `[ID]` **Q06** How many injector-producer pairs have a connectivity weight above 0.2?  
  <sub>type=scalar · gold=`4`</sub>
- `[ID]` **Q07** What is the most recent log10(WOR) value for producer F-14?  
  <sub>type=scalar · gold=`1.4557`</sub>
- `[ID]` **Q08** Which producers receive injection support from injector F-4?  
  <sub>type=entity_list · gold=`['F-1 C', 'F-11', 'F-12', 'F-14']`</sub>
- `[ID]` **Q09** What is the cumulative oil production (Np) for producer F-12 at the latest timestep?  
  <sub>type=scalar · gold=`4579609.55`</sub>
- `[ID]` **Q10** What is the single strongest injector-producer connection in the field?  
  <sub>type=entity_list · gold=`['F-11', 'F-5']`</sub>
- `[ID]` **Q11** How many producer wells and how many injector wells are in the field?  
  <sub>type=scalar · gold=`5 producers, 2 injectors`</sub>
- `[ID]` **Q12** What is the most recent water-cut fraction for producer F-12?  
  <sub>type=scalar · gold=`0.8835`</sub>
- `[ID]` **Q13** On what date was producer F-14's water-cut last measured?  
  <sub>type=scalar · gold=`2016-07-01`</sub>

## Risk

- `[RI]` **Q14** Which producers have ever raised a WCT_SEVERE (water-cut >= 90%) alert?  
  <sub>type=entity_list · gold=`['F-12', 'F-14']`</sub>
- `[RI]` **Q15** Which producers experienced a water-breakthrough alert?  
  <sub>type=entity_list · gold=`['F-1 C', 'F-11', 'F-12', 'F-14', 'F-15 D']`</sub>
- `[RI]` **Q16** Which injector-producer pairs have connectivity weight above 0.3 (early-breakthrough risk)?  
  <sub>type=scalar · gold=`2`</sub>
- `[RI]` **Q17** How many HIGH_WOR alerts were raised across the field?  
  <sub>type=scalar · gold=`185`</sub>
- `[RI]` **Q18** Which producer has the greatest number of severe water-cut alerts?  
  <sub>type=scalar · gold=`F-14`</sub>
- `[RI]` **Q19** Do any producers depend on only a single injector (support single-point-of-failure)?  
  <sub>type=boolean · gold=`False`</sub>
- `[RI]` **Q20** How many total alerts were raised in the field?  
  <sub>type=scalar · gold=`234`</sub>
- `[RI]` **Q21** Which producer reached water breakthrough earliest (by alert date)?  
  <sub>type=scalar · gold=`F-12`</sub>
- `[RI]` **Q22** What fraction of producers ever crossed the 50% water-cut breakthrough threshold?  
  <sub>type=scalar · gold=`1.0`</sub>
- `[RI]` **Q23** Is producer F-12's connectivity strong enough (>0.2 from any injector) to risk injected-water arrival?  
  <sub>type=boolean · gold=`True`</sub>
- `[RI]` **Q24** Which producer carries the highest single connectivity weight (largest injected-water exposure)?  
  <sub>type=scalar · gold=`F-11`</sub>
- `[RI]` **Q25** How many distinct producers currently sit above the severe (90%) water-cut level?  
  <sub>type=scalar · gold=`2`</sub>

## Recommendation

- `[RE]` **Q26** What is the single cheapest recommended intervention (by cost per barrel)?  
  <sub>type=scalar · gold=`I_WCT_SEVERE_F-14_2015-05-01`</sub>
- `[RE]` **Q27** Which intervention action type is recommended most frequently?  
  <sub>type=scalar · gold=`gas_lift_opt`</sub>
- `[RE]` **Q28** How many recommended interventions are there in total?  
  <sub>type=scalar · gold=`49`</sub>
- `[RE]` **Q29** Which intervention has the highest expected oil uplift (bbl)?  
  <sub>type=scalar · gold=`I_WATER_BREAKTHROUGH_F-15D_2015-12-01`</sub>
- `[RE]` **Q30** What is the mean estimated cost-per-barrel across all interventions?  
  <sub>type=scalar · gold=`12.65`</sub>
- `[RE]` **Q31** Which intervention offers the best expected-uplift-per-cost ratio?  
  <sub>type=scalar · gold=`I_WCT_SEVERE_F-14_2016-02-01`</sub>
- `[RE]` **Q32** How many interventions are recommended for producer F-14?  
  <sub>type=scalar · gold=`25`</sub>
- `[RE]` **Q33** Which producers have at least one recommended intervention?  
  <sub>type=entity_list · gold=`['F-1 C', 'F-11', 'F-12', 'F-14', 'F-15 D']`</sub>
- `[RE]` **Q34** Rank the 5 cheapest interventions by cost-per-barrel (ascending).  
  <sub>type=ranking · gold=`['I_WCT_SEVERE_F-12_2013-07-01', 'I_WCT_SEVERE_F-12_2014-02-01', 'I_WCT_SEVERE_F-12_2014-06-01', 'I_WCT_SEVERE_F-14_2015-05-01', 'I_WCT_SEVERE_F-14_2016-02-01']`</sub>
- `[RE]` **Q35** What action is recommended for the cheapest intervention?  
  <sub>type=scalar · gold=`water_shutoff`</sub>
- `[RE]` **Q36** How many interventions use the water_shutoff action?  
  <sub>type=scalar · gold=`10`</sub>
- `[RE]` **Q37** For producer F-14, which recommended intervention has the lowest cost per barrel?  
  <sub>type=scalar · gold=`I_WCT_SEVERE_F-14_2015-05-01`</sub>
- `[RE]` **Q38** Rank the top 5 interventions by expected uplift (descending).  
  <sub>type=ranking · gold=`['I_WATER_BREAKTHROUGH_F-15D_2015-12-01', 'I_WCT_SEVERE_F-12_2013-06-01', 'I_WCT_SEVERE_F-12_2013-11-01', 'I_WCT_SEVERE_F-14_2015-12-01', 'I_WCT_SEVERE_F-14_2016-02-01']`</sub>

## Explanation

- `[EX]` **Q39** Explain why producer F-14 raised a severe water-cut alert: trace alert -> triggering reading -> recommended intervention.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q40** Trace the earliest water-breakthrough alert: which reading triggered it, on what date, and what intervention was recommended?  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q41** Why is the F-5 -> F-11 pair flagged as an early-breakthrough risk? Explain using its connectivity weight.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q42** Explain the provenance chain of the cheapest recommended intervention back to its source record.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q43** What evidence supports a HIGH_WOR alert (which reading and threshold)?  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q44** Explain the difference in water-cut trajectory between producers F-12 and F-15 D.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q45** Why does producer F-11 receive more support from injector F-5 than from F-4? Explain using the weights.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q46** Explain, end to end, the risk-to-recommendation chain for the field's highest-risk producer.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q47** Explain why every KG node carries a source_record_id and why that matters for citation validation.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q48** What causes the recommended water_shutoff interventions, in terms of the water diagnostics?  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q49** Describe the connectivity structure of injector F-4: which producers it supports and the relative weights.  
  <sub>type=explanation · gold=`rubric`</sub>
- `[EX]` **Q50** Summarize how an alert, its triggering reading, and its recommended intervention link together in the ontology.  
  <sub>type=explanation · gold=`rubric`</sub>