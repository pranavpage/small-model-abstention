| Split | Model | Mean answer words | Lenient − strict gap | Answers longer than SFT's (both answered) |
|---|---|---|---|---|
| dev | SFT | 2.63 | 0.033 | — |
| dev | G-default best (step 200) | 2.87 | 0.049 | 8 of 136 |
| dev | G-equal best (step 200) | 3.49 | 0.078 | 37 of 144 |
| test | SFT | 3.27 | 0.017 | — |

Biggest length increases over SFT's answer to the same question:

| Model | Split | Question | Gold | SFT | Model | +words | Cell (SFT → model) |
|---|---|---|---|---|---|---|---|
| G-default best (step 200) | dev | What is insufficient to establish a direct origin from ancient festivals to Carnival? | evidence | evidence is insufficient | evidence is insufficient to establish a direct origin from Saturnalia or other ancient festivals | +11 | correct → correct |
| G-equal best (step 200) | dev | How many bits of control flags did USB 2.0 add? | 12 bits | seven | seven-bit hub number, 12 bits of control flags, and a five-bit CRC. | +11 | wrong → correct |
| G-equal best (step 200) | dev | What was the source of the misinformation concerning damage to the city's boardwalk? | photograph | a widely circulated photograph of a damaged section of the Boardwalk | a widely circulated photograph of a damaged section of the Boardwalk that was slated for repairs, prior to the storm | +9 | correct → correct |
| G-equal best (step 200) | dev | What is the meaning of impermanence in Buddhism? | all compounded or conditioned phenomena (all things and experiences) are inconstant, unsteady, and impermanent | all things and experiences are inconstant, unsteady, and impermanent | all compounded or conditioned phenomena (all things and experiences) are inconstant, unsteady, and impermanent | +5 | correct → correct |
| G-equal best (step 200) | dev | What is the least important factor which governs how a capacitor ages? | (unanswerable) | operating voltage | operating voltage has a smaller effect | +4 | hallucination → hallucination |
