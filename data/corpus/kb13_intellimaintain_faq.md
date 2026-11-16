---
title: IntelliMaintain - Frequently Asked Questions About the System and Its Results
type: user guide (project knowledge base)
---

# 0. What a bearing is

A bearing is a machine element that supports a rotating shaft and lets it turn with very little friction. A rolling-element bearing, also called a rolling bearing or ball bearing, has an inner ring fixed on the shaft, an outer ring fixed in the housing, rolling elements such as balls or rollers between the two rings, and a cage that keeps the rolling elements evenly spaced. Bearings are used in almost every rotating machine, including electric motors, pumps, fans, compressors and gearboxes, and a bearing failure is one of the most common reasons for an unplanned machine stop.

# 1. What IntelliMaintain does

IntelliMaintain is an AI copilot for bearing maintenance. It diagnoses the condition of a rolling-element bearing from a vibration signal, explains how the fault develops, and recommends maintenance actions with a citation for every statement. Its output is advisory: a qualified engineer reviews every answer and accepts, rejects or edits it before any action is taken.

A diagnosis runs in four steps. First, a machine-learning classifier predicts the bearing condition from the vibration signal. Second, a physics check calculates the bearing defect frequencies from the shaft speed and looks for them in the envelope spectrum. Third, a retrieval system searches the maintenance documents, and optionally the web, for passages about the fault. Fourth, a language model or the offline generator writes the answer using only that evidence, and every sentence is checked against the source it cites.

# 2. What the citations mean

A citation in square brackets shows where each statement comes from. [ML] means the statement comes from the machine-learning classifier. [PHYS] means it comes from the physics check of the defect frequencies. A number such as [2] refers to evidence passage 2 in the evidence list, which comes from a maintenance manual, guideline or maintenance log. A citation such as [W1] refers to web evidence passage 1, which comes from an online source and is shown with its link. Sentences that cannot be matched to the source they cite are removed automatically.

# 3. What the confidence and anomaly score mean

The confidence is the calibrated probability of the predicted class, averaged over all 2048-sample windows of the signal. A confidence of 0.95 means that, on test data, predictions with this confidence were correct about 95 percent of the time. Below 0.60 the result is flagged as low confidence and should be verified by an analyst.

The anomaly score comes from an Isolation Forest trained only on the healthy (Normal) CWRU recordings. It measures how far the signal is from a healthy bearing: 0 means it looks like a healthy bearing and 1 means it is far from healthy. Every faulty CWRU recording scores close to 1, so a high anomaly score supports a fault diagnosis. It also works without the classifier: a signal with a high anomaly score that the classifier calls Normal should be checked by an analyst.

# 4. What the physics cross-check means

The physics check calculates the defect frequencies BPFO, BPFI and BSF from the bearing geometry and the shaft speed, then measures how strongly each frequency and its harmonics stand out in the envelope spectrum. A ratio above 5 means the frequency is present. If the physics check points to the same fault as the classifier, the result is consistent. If it points to a different fault, the result is a conflict and the answer asks for analyst verification instead of giving a fault type. If no frequency stands out, the result is inconclusive, which is common for normal bearings and for ball faults.

The shaft speed directly changes the defect frequencies, so a wrong speed can make the physics check miss a real fault. The sampling rate must match the recording, because the signal is resampled to 12 kHz before analysis. In Auto-detect mode the app reads both values from the file when it can: a header line such as # fs=12000 or # rpm=1730, a time column, the file name (for example 48k, 1750rpm or 3hp) or a CWRU file number. If the file does not give the shaft speed, the app estimates it from the vibration spectrum by finding the shaft-speed line and ignoring 50 or 60 Hz mains hum; on the 40 CWRU recordings this estimate is within 3 percent for 95 percent of them. When no clear line is found, the app assumes 1772 rpm and asks the user to check. In Manual mode the user types both values, for example from the motor nameplate or the data logger.

# 5. Which files can be uploaded

The app accepts vibration recordings as CSV, TXT, MAT or NPY files. A CSV file can contain a single column of acceleration values or several columns; the app uses the column with the vibration signal. The recording must contain at least 2048 samples at 12 kHz, and longer recordings of several seconds give more reliable results because more windows are averaged. The app also offers demo samples taken from the CWRU dataset.

# 6. What "insufficient evidence" means

Insufficient evidence means that neither the classifier nor the documents give enough information to answer the question reliably. IntelliMaintain does not guess in this case. The question can be rephrased, a vibration recording can be uploaded, or online search can be enabled so that web sources are searched as well.

# 7. Which models are used

The default classifier is a RandomForest trained on 29 features per window: 10 time-domain statistics such as RMS, kurtosis and crest factor, 3 spectral shape features and 16 frequency-band energies. Its probabilities are calibrated with isotonic regression. The second classifier is a convolutional neural network that classifies a 64 by 64 spectrogram image of each window. Both classify 10 CWRU classes, which are combined into four fault types: Normal, Inner race, Outer race and Ball.

The knowledge retrieval combines dense embedding search with BM25 keyword search and merges both rankings with reciprocal rank fusion. The language model is Google Gemini, used through its API with a low temperature. When no API key is configured, when the quota is used up or when the answer fails verification, an offline extractive generator builds the answer from the evidence instead, so the app always works.

# 8. Whether the machine can keep running

The decision to keep a machine running depends on the severity and the trend, not on a single measurement. A normal result means routine monitoring can continue. A fault in an early stage usually allows the machine to run with more frequent monitoring until a planned stop, while a rapid rise in vibration, temperature or noise means the machine should be stopped. The final decision belongs to the responsible engineer and must follow the site's safety rules, including lockout and tagout before any work on the machine.

# 9. Differences between the fault types in plain words

An outer race fault is damage on the stationary outer ring; it gives regular impacts of equal strength at BPFO and is usually the easiest fault to detect. An inner race fault is damage on the rotating inner ring; it gives impacts at BPFI whose strength rises and falls once per revolution, which adds sidebands at shaft speed. A ball fault is damage on a rolling element; it gives weaker, irregular impacts at twice the ball spin frequency with sidebands at the cage frequency and is the hardest to detect. A normal bearing shows no repeating impacts, low RMS and a kurtosis close to 3.
