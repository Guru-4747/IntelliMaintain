---
title: The CWRU Bearing Dataset and How IntelliMaintain Learns From It
type: project reference (project knowledge base)
---

# 1. What the CWRU dataset is

The Case Western Reserve University (CWRU) Bearing Data Center provides the most widely used public benchmark for bearing fault diagnosis. The test rig consists of a 2 hp Reliance Electric induction motor, a torque transducer and encoder, and a dynamometer that applies the load. Accelerometers mounted on the motor housing near the drive-end bearing and the fan-end bearing record the vibration, and the motor speed is recorded with each file.

Single-point faults were seeded into the drive-end bearing (SKF 6205-2RS JEM) with electro-discharge machining. The fault diameters used by IntelliMaintain are 0.007, 0.014 and 0.021 inch on the inner race, on a rolling element (ball) and on the outer race at the 6 o'clock position. Every condition was recorded at motor loads of 0, 1, 2 and 3 hp, which correspond to shaft speeds of about 1797, 1772, 1750 and 1730 rpm.

# 2. The data IntelliMaintain is trained on

IntelliMaintain uses the standard 10-class benchmark from the 12 kHz drive-end experiments: Normal, B_007, B_014, B_021, IR_007, IR_014, IR_021, OR_007, OR_014 and OR_021. The letters give the fault location (B for ball, IR for inner race, OR for outer race) and the number gives the fault diameter in thousandths of an inch. With 10 classes at 4 loads, the training set contains 40 recordings.

The CWRU normal baseline files were recorded at 48 kHz while the fault files were recorded at 12 kHz. IntelliMaintain anti-alias filters and decimates the normal files by 4 so that every class shares the same 12 kHz time base; without this step a classifier can learn the sampling rate instead of the bearing condition.

Each recording is cut into windows of 2048 samples (about 0.17 seconds at 12 kHz) with 50 percent overlap, which gives 4643 windows from the 40 recordings. Every window becomes one training example with the label of its recording.

# 3. Typical CWRU signal indicators for each class

The table below lists the range of window-average indicator values over the 4 load conditions for each CWRU class, as stored in the IntelliMaintain classifier. RMS is in g, kurtosis and crest factor have no unit, and a healthy Gaussian-like signal has a kurtosis close to 3.

| CWRU class | RMS (g) | Kurtosis | Crest factor |
| --- | --- | --- | --- |
| Normal | 0.056 to 0.071 | 2.779 to 3.047 | 3.362 to 3.673 |
| B_007 | 0.138 to 0.153 | 2.826 to 2.962 | 3.377 to 3.556 |
| B_014 | 0.130 to 0.145 | 7.113 to 9.016 | 5.933 to 6.423 |
| B_021 | 0.107 to 0.129 | 3.094 to 4.217 | 3.631 to 4.427 |
| IR_007 | 0.291 to 0.313 | 5.276 to 5.559 | 4.448 to 5.077 |
| IR_014 | 0.163 to 0.194 | 17.876 to 22.017 | 8.717 to 9.035 |
| IR_021 | 0.442 to 0.525 | 7.394 to 8.312 | 5.681 to 6.066 |
| OR_007 | 0.570 to 0.669 | 7.585 to 7.963 | 4.880 to 5.078 |
| OR_014 | 0.093 to 0.099 | 2.933 to 3.440 | 3.563 to 4.029 |
| OR_021 | 0.555 to 0.582 | 20.084 to 21.522 | 8.489 to 8.767 |

Normal recordings have the lowest RMS of all classes and a kurtosis close to 3, which matches a healthy bearing without impacts. Most fault classes raise RMS, kurtosis or crest factor because the rolling elements strike the defect. The relation to defect size is not simple: IR_014 and OR_021 have very high kurtosis, while OR_014 and B_007 have kurtosis close to 3. A single indicator therefore cannot separate all ten classes, which is why IntelliMaintain uses 29 features with a machine-learning model and adds the physics check.

# 4. How IntelliMaintain compares a new signal with CWRU

For every uploaded signal, IntelliMaintain computes the same 29 features for each window and averages them. It then measures the distance between this average and the average feature vector of each of the 40 CWRU recordings, after standardising every feature. The three closest recordings are reported as the most similar CWRU cases with a similarity between 0 and 1. If the closest recordings share the predicted fault type, the prediction is supported by case-based evidence; if they disagree, the result should be treated with caution.

# 5. How well the models perform on CWRU

With a random 80/20 split of the windows, the RandomForest reaches an accuracy of 1.0000. A random split is optimistic, because overlapping windows from the same recording can appear in both the training and the test set. In the cross-load test the models are trained on loads 0 to 2 hp and tested on the unseen 3 hp load, and the RandomForest still reaches an accuracy of 1.0000. In leave-one-load-out testing the mean accuracy is 0.9995.

The hardest test is cross-severity generalisation: the models are trained on 0.007 and 0.014 inch faults and tested on unseen 0.021 inch faults. Here the accuracy drops to 0.4268 for the RandomForest and 0.4359 for the spectrogram CNN, because a new defect size changes the signal more than a new load does. This is the honest limit of models trained only on CWRU.

Under added noise on the cross-load test, the RandomForest accuracy falls from 1.0000 on clean data to 0.4058 at 5 dB signal-to-noise ratio. The spectrogram CNN trained with noise augmentation keeps 0.9618 at 5 dB, and a RandomForest trained with the same augmentation reaches 0.9652, so most of the robustness comes from the augmentation.

Isotonic calibration reduced the expected calibration error of the RandomForest from 0.1203 to 0.0163 on the unseen 3 hp load, so its confidence values can be read as probabilities.

# 6. What the physics check achieves on CWRU

The envelope-spectrum physics check, used on its own without machine learning, identifies the fault type of 0.600 of the CWRU recordings. It identifies 1.000 of the inner race and normal recordings and 0.667 of the outer race recordings, but none of the ball faults, because ball defects produce weak and irregular impacts at twice the ball spin frequency. For this reason the physics check is used to confirm or question the classifier, not to replace it.

# 7. Limits of a model trained on CWRU

CWRU contains seeded faults from a single motor, a single bearing type and steady speeds between 1730 and 1797 rpm. A model trained only on CWRU should not be trusted blindly on a different machine, bearing, speed range or sensor position. Before use on a new machine, the model should be checked with labelled data from that machine, and low-confidence or conflicting results should always be verified by a vibration analyst.
