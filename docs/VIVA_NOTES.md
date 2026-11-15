# Viva / review preparation notes

Short answers to the questions examiners usually ask. Numbers come from `reports/`.

**1. Why not just report the 100 % accuracy?**
Because overlapping windows from the same recording appear in both train and test in a random split (leakage), and
CWRU is known to be saturated. We therefore add cross-load, leave-one-load-out, cross-severity and noise tests. The
cross-severity test (unseen 0.021" defects) drops to 42.7 % – this is the honest limit of the ML layer and the reason
for the physics check and human review.

**2. What is the 48 kHz issue?**
CWRU "Normal baseline" files are recorded at 48 kHz, fault files at 12 kHz. Feeding both as 12 kHz makes the model
learn the sampling rate. We verified it (shaft peak at 30 Hz only when read as 48 kHz) and resample with
`scipy.signal.resample_poly` (anti-aliased).

**3. Why calibration?**
The reviewer sees a confidence number; it must mean something. Isotonic calibration with folds grouped by load reduced
ECE from 0.120 to 0.016 on the unseen 3 HP load.

**4. How does the physics check work?**
Band-pass 2–5 kHz → Hilbert envelope → FFT. We measure the peak-to-background ratio at 1–3 harmonics of BPFO, BPFI and
2×BSF (±3 %). SNR > 5 = present (all healthy recordings < 4.3). If it strongly contradicts the ML label the agent
reports "Unknown – verify" (abstain policy).

**5. Why hybrid retrieval?**
Dense BGE embeddings capture meaning ("spall on outer ring" ≈ "outer race defect"); BM25 catches exact terms (BPFO,
ISO 15243, WO numbers). Reciprocal-rank fusion merges both without score calibration; a cross-encoder reranks the top
candidates.

**6. How do you prevent hallucination?**
(a) prompt allows only context, (b) every sentence must cite `[n]`, `[ML]` or `[PHYS]`, (c) `verify.py` deletes
citations to non-existent sources and measures coverage, (d) low relevance → "insufficient evidence", (e) human
review with accept/reject/edit log.

**7. Why a bounded agent instead of a free agent?**
Safety and reproducibility (Di Maggio 2025): only 3 whitelisted tools, ≤ 5 calls, invalid planner output falls back to
the fixed plan, every call is logged in the tool trace.

**8. What do the ablation results show?**
ML output is what makes the fault type correct (0.10 → 1.00); retrieval is what makes the explanation useful
(key-fact recall 0.35 → 1.00) and cited. The full system is best on both.

**9. What happens when the ML model is wrong?**
Stress test: 30 injected wrong labels – 80 % flagged, confidently wrong answers reduced from 100 % to 33 % with the
abstain policy, no loss on correct predictions. Ball faults are the weak spot because the physics check is
usually inconclusive for them (it only overrules a ball label when race-fault lines are very strong, SNR >= 25).

**10. Why add a CNN if the RandomForest is already 100 %?**
To test whether deep learning on time-frequency images is more robust. With noise augmentation the CNN keeps 97.5 %
accuracy at 5 dB SNR (RF without augmentation: 41 %). Our control experiment (RF trained with the *same*
augmentation: 96.5 %) shows the gain comes mostly from augmentation; the CNN only wins clearly at 0 dB (76 % vs 69 %).
Both fail on unseen defect sizes (~43 %), so the bottleneck is data, not model capacity.

**11. Why is temperature T constrained to ≥ 1?**
CWRU validation data is almost 100 % correct, so plain temperature scaling would choose T → 0 and make the CNN
always ~100 % confident – silencing the low-confidence warning on unfamiliar machines. We only allow calibration to
soften probabilities (a safety choice).

**12. How do you stop Gemini from hallucinating?**
The LLM never decides the facts. The fault type is locked to the classifier/physics result; each sentence must
cite a source ([n], [ML], [PHYS]) and share ≥ 30 % of its content words with it; every number must appear in the
cited source (1 % rounding tolerance). Unverified sentences are deleted; below 50 % verified the system shows the
offline, fully extractive answer. The grounding score is shown with every answer and logged in the evaluation.

**13. What is the "similar cases" panel?**
Case-based evidence: the signal's mean standardised feature vector is compared with all CWRU recordings and the
three closest are shown. The evaluation model only contains 0-2 HP cases, so 3 HP test files never match themselves.

**14. Can users type normally?**
Yes. A rule-based language layer (offline) expands short forms (u, wat, shud, pls…), corrects spelling against the
knowledge-base vocabulary (bering → bearing), maps everyday words to maintenance terms (grinding noise → vibration,
getting hot → temperature/lubrication) and detects intent (greeting, help, diagnosis, knowledge, out-of-scope).
On 24 informal questions: intent 100 % correct; relevant document in top-5 rises from 80 % to 100 %.

**15. SDG mapping** – SDG 9.4/9.5 (efficient, modernised industry), 8.2/8.8 (productivity, safe work),
12.5 (less waste from premature replacement).
