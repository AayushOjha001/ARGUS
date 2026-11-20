# Related Work: ARGUS and Multimodal / Embodied Prompt Injection

This note places ARGUS [lu2026argus] among published attacks and defenses, and
explains where the closed-loop embodied extension in [`embodied/`](../embodied/) fits.
Citation keys in brackets refer to [`related_work.bib`](related_work.bib).

> **Scope of evidence.** The comparisons below come from each paper's abstract,
> arXiv record, or venue listing, plus the ARGUS code in this repository. They were
> not re-derived from the full experimental sections. Reported numbers come from
> different models, datasets and threat models, so **they are not directly
> comparable** and should not be put side by side in one results table.

---

## 1. ARGUS in one paragraph

ARGUS defends multimodal LLMs against *indirect* prompt injection (IPI): instructions
hidden in an image, video or audio clip that override the user's request. It works
in the model's representation space and has three stages (see
[`src/argus/defense.py`](../src/argus/defense.py)):

1. **Detection.** A linear probe on an early layer (layer 6 by default) flags injected inputs.
2. **Steering.** Only when the probe fires, ARGUS adds a steering vector at a set of
   intermediate layers. The direction is searched inside an "instruction-following"
   subspace so that it is decoupled from a utility-degrading direction. The
   strength α is set per token from the probe margin:
   `α = ReLU((w·h + b + ε) / ‖w‖²)`.
3. **Post-filtering.** A second probe on a later layer (layer 20) checks whether
   steering worked and otherwise rejects the response.

ARGUS needs white-box access and a per-model training stage (probes and vectors).
At inference it adds only cheap hooks. Its in-paper baselines are:
- a vigilant system prompt;
- an "ignore instructions in the media" prompt;
- Gaussian noise on the media;
- **Removal**: an editing model (Step1X-Edit for images, WAN-2.1-VACE for video)
  erases the injected text;
- **adversarial training (AT)**: DPO fine-tuning to prefer the user instruction.

The first four are in [`src/argus/baselines.py`](../src/argus/baselines.py). AT is
reported in the paper only. Its attack data are
typographic caption strips on VTQA images, text clips inserted into MSR-VTT videos,
and TTS speech appended to Clotho-AQA audio
([`src/argus/data_builder.py`](../src/argus/data_builder.py)).

---

## 2. Defenses

### 2.1 Representation-level defenses (closest to ARGUS)

**Steering for multimodal safety.** ASTRA [wang2025astra] is the nearest
multimodal relative. It builds steering vectors by ablating visual tokens of
adversarial images and projects activations away from the harmful direction at
inference. ASTRA targets *jailbreaks* (eliciting harmful content), whereas ARGUS
targets *task hijacking*, where the output may be harmless but is not what the user
asked for. ASTRA has no separate detector: its steering hook runs on every generated token,
and a built-in condition (is the activation pointing toward the harmful
direction?) decides how hard to push, often not at all on benign inputs. ARGUS
instead uses a trained probe to decide once, at prefill, whether to intervene,
and checks the result afterwards with a second probe.

**Other activation-level defenses for VLM jailbreaks.** HiddenDetect
[jiang2025hiddendetect] (ACL 2025) monitors hidden states for safety-relevant
patterns without any tuning. ShiftDC [zou2025shiftdc] finds that adding an image
shifts activations toward a "looks safer" direction and removes that component.
NullSteer [zhu2026nullsteer] (CVPR 2026) builds refusal directions that have zero
effect inside the benign activation subspace. This is the jailbreak counterpart of
ARGUS's search for a direction that does not cost utility. All three target
harmful content, not task hijacking.

**Steering against text prompt injection.** Several concurrent text-only works
reach the same core finding as ARGUS: models linearly encode whether they are
following the original instruction or an injected one, and this can be steered.
"Do LLMs Encode Instruction Overwrites?" [anon2026overwrites] probes layer-wise and
derives a single steering vector that restores the original task without
retraining. "Probing and Steering Prompt Injection Compliance"
[anon2026probingsteering] predicts compliance before generation (ROC-AUC 0.90 on
Qwen2.5-3B) and subtracts a "comply" direction. Steering lowers compliance from
0.58 to 0.26, but mostly for task-replacement attacks, and the authors frame it
as evidence rather than a deployable defense. Related steering work covers
role conflicts [zeng2025whoisincharge], poisoned MCP tool descriptions
[anon2026toolpoisoning] and attention steering for agents [wang2026icon]. ARGUS
differs from these in (i) being modality-agnostic, (ii) explicitly searching for a
direction that avoids degrading utility, (iii) using adaptive rather than fixed
strength, and (iv) wrapping steering in detect → steer → verify.

**Activation-based detectors.** TaskTracker [abdelnabi2025tasktracker] trains a
linear probe on the change in activations before and after external data is read.
InstructDetector [wen2025instructdetector] combines hidden states and gradients.
Attention Tracker [hung2024attentiontracker] needs no training: it watches
"important heads" shift attention from the user instruction to the injected one.
All three are text-only and **detect but do not repair**. They correspond to
ARGUS's first stage alone.

**How far probe results can be trusted.** Four recent papers apply directly to
ARGUS's detection and post-filter probes:

- **Context, not injection** [kale2026probes]: near-perfect AUROC can come from
  probes learning conversation position and length rather than the injection.
- **Dataset shortcuts** [fomin2026benchmarkslie]: standard cross-validation reports
  AUC 8.0–16.5 points higher than leave-one-dataset-out testing, and 28–44% of the
  top features are dataset shortcuts.
- **Multimodal agents** [li2026aucprobes]: on Qwen2.5-VL-7B computer-use agents, a
  clean-vs-attack AUC of 0.998 does not by itself show that the probe detects
  malicious content. The paper proposes nuisance-matched controls.
- **Adaptive evasion** [rahman2026evasive]: one optimized (GCG) suffix evades every
  layer-wise drift probe at once (93.91% on Phi-3, 99.63% on Llama-3-8B) while the
  injection still works. Retraining the probes on suffix-augmented activations
  restores detection.

### 2.2 Input/output-level defenses for multimodal models

Most multimodal defenses target harmful-content jailbreaks, not IPI:

- **Prompting:** AdaShield [wang2024adashield] prepends an adaptively chosen defense prompt.
  It is the same family as ARGUS's system-prompt baseline.
- **Modality conversion:** ECSO [gou2024ecso] captions the image and answers from text.
  For IPI this is unlikely to help, because a faithful caption carries the injected
  text along. This is our reasoning, not a reported result.
- **Detect-and-rewrite / alignment at inference:** MLLM-Protector [pi2024mllmprotector],
  ETA [ding2024eta] and Immune [ghosal2025immune] (reward-guided decoding).
- **Prompt classifiers:** VLMGuard [fang2026vlmguard] and VLMShield [qi2026vlmshield].

Two recent works address visual *injection* directly:

- **Trust-boundary separation** [chang2026trustboundary] is the defense closest
  to our setting: built for signs in an embodied agent's scene (other embodied
  guardrails are in §2.5). A perception agent rates how trustworthy each
  visual sign is, and a separate judgment agent decides what to follow. It reports
  about 3% attack success while keeping more than 95% of helpful cues. It is
  black-box and costs extra model calls.
- **Render Before Reading / Pictionary** [zhang2026renderbeforereading] goes the
  opposite way: it renders untrusted *text* tool outputs as an image and finds that
  VLMs follow it less. ARGUS shows that instructions in pixels remain highly
  effective (typographic attacks, below), so the two results bound when rendering
  helps and when it does not.

### 2.3 Text-only IPI defenses

StruQ [chen2025struq] and SecAlign [chen2025secalign] fine-tune models to ignore
instructions in a delimited data channel. Spotlighting [hines2024spotlighting]
marks data with delimiters or encoding. MELON [zhu2025melon] re-executes the
agent with masked user input to detect injection-driven tool calls. CaMeL
[debenedetti2025camel] separates control flow from data at the system level.
ARGUS's argument is that these do not transfer to MLLMs: an instruction rendered
inside pixels or speech has no separate data field to delimit or mark, and
retraining per modality is expensive.

### 2.4 Typographic-attack defenses

ARGUS's attacks are typographic, so defenses aimed at text inside images are the
closest input-side alternative:

- **Dyslexify** [hufe2025dyslexify] ablates a circuit of CLIP attention heads that
  carries text. It gains up to 22.06% on a typographic ImageNet-100 at under 1%
  clean-accuracy cost.
- **Concept localization** [liu2026conceptloc] finds ViT heads that encode lexical
  content and adjusts them without training, including inside LVLM vision encoders.
- **QuISE** [lu2026quise] is black-box: it rewrites text regions irrelevant to the
  query and checks that the answer stays consistent.

These work on the vision side and suppress *reading* text in general. ARGUS works
on the language model's instruction-following, so the model can still read a scene
label without obeying it. For a robot that must read legitimate signs this
difference matters: the helpful-cue setting of [chang2026trustboundary] tests it,
and [samarakoon2026hijacking] note that masking text blocks tasks that need
in-scene labels.

### 2.5 Embodied defenses, and reading robot-model internals

**Guardrails for embodied agents.** Most check planned actions or prompts:

- Safety Chip [yang2023safetychip] encodes constraints in temporal logic and prunes
  unsafe actions.
- RoboSafe [wang2025robosafe] derives executable safety predicates at run time and
  was tested on a physical arm.
- SafeEmbodAI [zhang2024safeembodai] combines secure prompting, state management
  and response validation against malicious commands for mobile robots.
- J-DAPT [marchiori2025jdapt] is a multimodal jailbreak detector adapted to robot
  domains (driving, maritime, quadruped).
- Two attack papers include defenses: RIPA's semantic firewall [dorzhiev2026ripa],
  and a mask-based filter plus a "semantic-vector internal detector" for wearable
  VLMs in Devil in the Lens [li2026devillens] (its abstract does not say whether
  the detector reads the VLM's hidden states).

[wang2026guardrailbench] benchmarks six guardrails acting at the perception,
planning or control stage. It measures bypass rate, false positives, task
completion and latency, and finds that no single guardrail wins everywhere. Its
metrics are a ready-made frame for the embodied evaluation. None of these
defenses steers the planner's activations.

**Reading and steering robot models.** Internal access to robot policies has
precedents, although all target vision-language-action (VLA) models rather than
a VLM planner:

- [haon2025mechsteer] (CoRL 2025) steers Pi0 and OpenVLA activations in real time
  on a UR5 arm, for control rather than security.
- SAFE [gu2025safe] (NeurIPS 2025) detects task failure from VLA internal features.
- IGAR [zhang2026igar] recalibrates attention so a VLA follows its language
  instruction again, validated on a Franka arm.
- TrustVLA [fu2026trustvla] monitors per-layer, per-token uncertainty to catch
  backdoor triggers.

Together they show that internal monitoring and steering can run at robot
control rates.

### 2.6 Defense comparison

| Method | Modalities | Mechanism | Access / training | Detects | Repairs | Main difference from ARGUS |
|---|---|---|---|:-:|:-:|---|
| **ARGUS** [lu2026argus] | image, video, audio | probe → utility-decoupled adaptive steering → post-filter | white-box; probes + vectors per model | ✓ | ✓ | — |
| ASTRA [wang2025astra] | image | ablation-derived steering vectors, conditional projection | white-box; few forward passes | – | ✓ | jailbreak, not IPI; no detector, condition built into the steering formula |
| HiddenDetect [jiang2025hiddendetect] | image | hidden-state safety monitor | white-box; training-free | ✓ | – | jailbreak; detection only |
| ShiftDC [zou2025shiftdc] | image | remove modality-induced activation shift | white-box; training-free | – | ✓ | jailbreak, not IPI |
| NullSteer [zhu2026nullsteer] | image | refusal steering in the null space of benign activations | white-box | – | ✓ | jailbreak; refuses rather than repairing the task |
| Instruction Overwrites [anon2026overwrites] | text | probe + 1-D steering vector | white-box; no retraining | ✓ | ✓ | text only; fixed direction and strength |
| Probing & Steering PI Compliance [anon2026probingsteering] | text | compliance probe + direction subtraction | white-box | ✓ | partial | partial effect; analysis, not a defense |
| Tool-poisoning steering [anon2026toolpoisoning] | text (MCP tools) | difference-in-means / SAE / hypernet steering | white-box | – | ✓ | tool channel; single model |
| ICON [wang2026icon] | text agents | attention steering | white-box | – | ✓ | attention rather than residual stream |
| TaskTracker [abdelnabi2025tasktracker] | text | probe on activation deltas | white-box; probe | ✓ | – | detection only |
| InstructDetector [wen2025instructdetector] | text | hidden states + gradients | white-box; classifier | ✓ | – | detection only |
| Attention Tracker [hung2024attentiontracker] | text | attention-head distraction | white-box; training-free | ✓ | – | detection only; no training |
| AdaShield [wang2024adashield] | image | adaptive defense prompt | black-box | – | prompt | prompt-level, jailbreak |
| ECSO [gou2024ecso] | image | image → caption | black-box | ✓ | ✓ | caption keeps the injected text |
| MLLM-Protector [pi2024mllmprotector] / ETA [ding2024eta] / Immune [ghosal2025immune] | image | output check + rewrite / aligned decoding | mixed; Immune slows decoding | ✓ | ✓ | harmful-content focus |
| VLMGuard [fang2026vlmguard] / VLMShield [qi2026vlmshield] | image + text | malicious-prompt classifier | black-box | ✓ | – | detection only |
| Dyslexify [hufe2025dyslexify] / Concept localization [liu2026conceptloc] | image (vision encoder) | ablate or reweight text-reading heads | white-box encoder; training-free | – | ✓ | stops reading text in general |
| QuISE [lu2026quise] | image | edit query-irrelevant text, check answer consistency | black-box; several queries | – | ✓ | input sanitization |
| Trust boundary [chang2026trustboundary] | embodied, scene signs | perception agent + judgment agent | black-box; extra calls | ✓ | ✓ | closest embodied defense; prompt and agent level |
| RoboSafe [wang2025robosafe] / Safety Chip [yang2023safetychip] | embodied actions | executable safety logic / temporal-logic constraints | black-box | ✓ | ✓ (replan or prune) | checks actions, not the model's reading of the scene |
| J-DAPT [marchiori2025jdapt] / SafeEmbodAI [zhang2024safeembodai] | robot text + image | jailbreak classifier / secure prompting + validation | black-box | ✓ | partial | robot jailbreaks and malicious commands |
| Devil in the Lens defenses [li2026devillens] | wearable camera, physical text | mask filter + semantic-vector detector | not stated in abstract | ✓ | partial | still photos, not closed loop |
| Pictionary [zhang2026renderbeforereading] | text tool outputs | render text as image | black-box | – | ✓ | opposite modality assumption |
| StruQ [chen2025struq] / SecAlign [chen2025secalign] / Spotlighting [hines2024spotlighting] / MELON [zhu2025melon] / CaMeL [debenedetti2025camel] | text | data/instruction separation, fine-tuning, system design | mixed | mixed | ✓ | assume a separable data channel |

---

## 3. Attacks

### 3.1 Digital multimodal injection

The earliest multimodal IPI work [bagdasaryan2023abusing] blends adversarial
perturbations into images and audio so that LLaVA and PandaGPT follow a hidden
instruction while still describing the content. Image Hijacks [bailey2024imagehijacks]
optimize images that force arbitrary outputs, leak context, or bypass safety. Both
need white-box gradients. Later work moves to cheaper channels: typography
(FigStep [gong2025figstep]; [balakrishnan2026readingpixels] links typographic
success to text-image embedding alignment), steganography [pathade2025invisible],
and cross-modal image-only perturbation [yang2026crossmodal]. CrossInject
[wang2025crossinject] perturbs image and text together to hijack multimodal agents,
and [ding2026covertvpi] embeds imperceptible visual injections that transfer to
closed commercial MLLMs.

### 3.2 GUI and web agents

VPI-Bench [cao2026vpibench] places injections in pop-ups, chats and emails across
five platforms. It reports deception rates of up to 51% for computer-use agents and
100% for browser-use agents, and that a vigilant system prompt has limited effect.
Pop-up attacks [zhang2025popups] and EIA [liao2025eia] show that small injected UI
elements redirect agents or leak private data. AgentTypo [li2025agenttypo]
optimizes the text, position, contrast and colour of typographic injections in
webpage images with a black-box search. It is the adaptive version of the
typographic attack ARGUS is trained on.

### 3.3 Physical and embodied injection

- CHAI [burbano2025chai] optimizes printed signs against VLM agents for drone
  landing, driving and tracking, including a real robotic car.
- PPIA [ling2026ppia] is black-box and query-agnostic. It places typographic
  instructions on physical objects, chosen and positioned with attention guidance,
  and reports 70–98% success over 10 VLMs in simulation and real settings.
- "Hijacking Robots with a Piece of Paper" [samarakoon2026hijacking] builds a
  taxonomy of physical injections for manipulation robots, including
  authority impersonation. Its large-scale evaluation uses static images rather
  than closed-loop execution.
- SceneTAP [cao2024scenetap] plans scene-coherent typographic attacks.
- **Indirect environmental jailbreaks** [li2025shawshank] write instructions into
  the environment (for example on a wall). Its benchmark generator, SHAWSHANK-BENCH,
  covers 3,957 task-scene combinations; the attack compromises all six tested VLMs,
  and existing defenses only partly mitigate it.
- PI3D [li2026pi3d] (COLM 2026) optimizes the 6-DoF pose of a text-bearing object in
  a 3D scene so a camera-based MLLM reads and follows it. Existing defenses do not
  reliably stop it.
- Devil in the Lens [li2026devillens] studies physical injections against VLM smart
  glasses across 200+ real environments and 12 models.
- Typographic attacks on driving VLMs [chung2024typodriving] transfer across models
  and are designed to be physically realizable in traffic scenes.
- **Injection through other robot channels:** PINA [liu2026pina] targets navigation
  agents. RIPA [dorzhiev2026ripa] injects through a ROS 2 robot's sensor pipeline
  (OCR, speech-to-text, LiDAR state). [nagaraja2026multiagentpi] shows injections
  spreading between cooperating robot agents.
- **Robot jailbreaks** come from the user, not the scene: BadRobot
  [zhang2025badrobot], RoboPAIR [robey2024robopair] (including a commercial robot
  dog) and POEF [lu2024intentbehavior], which shows that harmful *text* often fails
  to become an executable harmful *action*.

Several of these papers report that existing defenses fall short
[li2025shawshank, li2026pi3d]. The only defense among them that may use internal
model state is the detector in [li2026devillens], tested on still photos rather
than in a control loop.

### 3.4 Surveys, and what this document leaves out

For wider coverage see the embodied-AI safety survey [li2026embodiedsafety], the
VLA safety survey [li2026vlasafety], a survey of LLM-controlled robotics
[huang2025trustrobotics], and a survey organized by trust boundary
[liu2026embodiedsecurity]. The last finds defenses concentrated at the action and
runtime level, and lists unified closed-loop evaluation as an open problem.

**Out of scope.** This document does not cover attacks and defenses for VLA action
models (adversarial patches, action-freezing attacks, backdoors) or benchmarks of
hazardous *user* instructions (household-safety and risk-awareness benchmarks).
They target a different model class or threat: ARGUS defends a VLM against
instructions hidden in its inputs. The two surveys above index that work.

### 3.5 Attack coverage of ARGUS

| Attack | Channel | Attacker access | Matches ARGUS's training/eval data? |
|---|---|---|---|
| Abusing Images & Sounds [bagdasaryan2023abusing], Image Hijacks [bailey2024imagehijacks], Cross-modal PI [yang2026crossmodal] | adversarial pixels / audio | white-box (gradients) | **No.** Not evaluated. |
| CrossInject [wang2025crossinject], Covert VPI [ding2026covertvpi] | joint image + text perturbation; imperceptible overlay | black-box / transfer | **No.** |
| AgentTypo [li2025agenttypo] | optimized typographic text | black-box, adaptive | **Partly.** Same format, but optimized against the model; untested. |
| Invisible Injections [pathade2025invisible] | steganographic text | black-box | **No.** |
| FigStep [gong2025figstep] | typographic image | black-box | **Yes.** Closest to ARGUS's caption-strip images. |
| VPI-Bench [cao2026vpibench], Pop-ups [zhang2025popups], EIA [liao2025eia] | GUI screenshots | black-box | **Partly.** Rendered text, but in a GUI domain. |
| CHAI [burbano2025chai], PPIA [ling2026ppia], Paper-hijack [samarakoon2026hijacking], SceneTAP [cao2024scenetap], Shawshank [li2025shawshank], PI3D [li2026pi3d], Devil in the Lens [li2026devillens], Typo-driving [chung2024typodriving] | text physically in the scene | black-box | **No.** This is what [`embodied/`](../embodied/) tests. |
| PINA [liu2026pina], RIPA [dorzhiev2026ripa], Multi-agent PI [nagaraja2026multiagentpi] | agent context, OCR / speech-to-text / LiDAR state, inter-agent messages | black-box | **No.** Text reaches the model through text channels, not image tokens. |

---

## 4. Metric comparison (focused)

This section compares ARGUS only with defenses that solve the **same problem**: a
vision-language model reading instructions or misleading text **inside an image or a
physical scene**. Text-only defenses and jailbreak-only defenses are left out, because
they cover a narrower problem than ARGUS. Their numbers remain in Appendix A.

> **Read this first.** Only the numbers in §4.2 come from one shared test (ARGUS's own
> paper). Every row in §4.3 and §4.4 comes from a different paper, with its own models,
> attacks and scoring. They show the size of each effect, not a ranking.

### 4.1 Metrics and abbreviations

**Metrics**

| Abbreviation | Full name | What it counts | Better when |
|---|---|---|---|
| AIA | Attacker Instruction Accuracy | share of attacked samples whose answer contains the attacker's expected answer (ARGUS's attack-success measure) | lower |
| AIFR | Attacker Instruction Following Rate | share of attacked samples where the model *tries* to follow the injected instruction, even if it gets it wrong | lower |
| UIA | User Instruction Accuracy | share of samples where the model correctly answers the user's real question | higher |
| UIA<sub>inject</sub> | User Instruction Accuracy on injected inputs | UIA while the attack is present: does the user still get their answer? | higher |
| UIA<sub>clean</sub> | User Instruction Accuracy on clean inputs | UIA with no attack: does the defense hurt normal use? | higher |
| ASR | Attack Success Rate | share of attacked trials where the attacker's goal is reached | lower |
| — | Defense effectiveness | 1 − (ASR with defense ÷ ASR without defense); 100% means every attack is stopped | higher |
| M Avg | Misleading-goal score | share of *misleading* signs the agent obeys (trust-boundary paper) | lower |
| H Avg | Helpful-goal score | share of *helpful* signs, such as a safety notice, the agent obeys | higher |
| RR | Recovery Rate | share of attack-caused errors that the defense fixes (QuISE paper) | higher |
| HR | Harm Rate | share of correct answers that the defense breaks (QuISE paper) | lower |

**Other abbreviations**

| Abbreviation | Full name |
|---|---|
| IPI | Indirect Prompt Injection: instructions hidden in content the model reads, not typed by the user |
| VLM / MLLM | Vision-Language Model / Multimodal Large Language Model |
| OCR | Optical Character Recognition (finding and reading text in an image) |
| TTS | Text-To-Speech |
| DPO | Direct Preference Optimization (a fine-tuning method) |
| PGD | Projected Gradient Descent (an attack that optimizes image pixels) |
| GCG | Greedy Coordinate Gradient (an attack that optimizes text tokens) |
| CUA / BUA | Computer-Use Agent / Browser-Use Agent |

### 4.2 ARGUS against its own baselines (same test, directly comparable)

Image and video use Qwen2-VL-7B; audio uses Kimi-Audio-7B. All values in %, except
time. Source: ARGUS Table 1 [lu2026argus].

| Defense | Image: attack success (AIA) ↓ | Image: user answered under attack (UIA<sub>inject</sub>) ↑ | Image: normal accuracy (UIA<sub>clean</sub>) ↑ | Video: attack success (AIA) ↓ | Audio: attack success (AIA) ↓ | Extra time per image (ms) |
|---|---|---|---|---|---|---|
| No defense | 25.1 | 30.9 | 49.6 | 28.2 | 12.6 | 0 |
| System prompt warning | 10.7 | 38.2 | 42.7 | 26.9 | 27.9 | 6 |
| Gaussian noise on the media | 7.6 | 34.3 | 46.8 | 9.6 | 0.0 | 1 |
| Removal (an editing model erases the text) | 0.0 | 48.5 | 49.3 | 1.5 | n/a | 12,885 |
| Adversarial training (DPO fine-tuning) | 2.3 | 41.1 | 40.7 | 1.6 | 1.4 | 0 |
| **ARGUS** | **0.1** | **46.3** | **49.6** | **0.1** | **0.0** | **3** |

- **ARGUS is the only row with near-zero attack success *and* unchanged normal
  accuracy.** Removal matches it on images but takes about 13 seconds per image
  (574 seconds per video) and has no audio version. Adversarial training lowers normal
  image accuracy from 49.6 to 40.7.
- **Same pattern on video and audio.** Normal accuracy stays equal to no defense
  (video 37.6, audio 65.7), and the user is answered more often under attack (video
  25.4 → 37.8, audio 45.6 → 58.0).
- **Prompt warnings are unreliable.** The system prompt *raises* audio attack success
  (12.6 → 27.9).
- **The post-filter matters.** Without the layer-20 check, image attack success rises
  from 0.1 to 4.3.
- **Three more models** (ARGUS Table 3): attack success ends at 0.0% on InternVL3.5-8B
  (image), 0.2% on Qwen2.5-VL-7B (video) and 0.0% on Qwen2-Audio-7B (audio).

### 4.3 Close defenses (each from its own paper)

Only defenses against text or instructions placed in images or physical scenes.
"Before → after" is without defense → with defense.

| Defense | Where it acts | Tested on | Attack success: before → after | Cost to normal use, and extra work | Tested against an attacker who knows the defense? | Versus ARGUS |
|---|---|---|---|---|---|---|
| **ARGUS** [lu2026argus] | inside the language model (hidden states) | Qwen2-VL-7B, Kimi-Audio-7B, plus 3 more models; own image, video and audio test | AIA 25.1 → 0.1 (image), 28.2 → 0.1 (video), 12.6 → 0.0 (audio) | UIA<sub>clean</sub> unchanged; +3–6 ms inside the same pass | No | reference |
| Trust-boundary multi-agent framework [chang2026trustboundary] | two extra prompted agents: one rates how reliable each sign is, one decides what to follow | 7 VLMs incl. GPT-4o; image-editing and manipulation agents | M Avg 8.0 → 0.7 (GPT-4o); 0.0–3.3 across all 7 models | H Avg rises 65.0 → 99.0 (GPT-4o): helpful signs are still obeyed; 2 extra model calls | Partly: white-box noise images (PGD) still mislead 34–56% of the time | Similar attack success, works on closed models, and also tests helpful signs, which ARGUS never measures. Slower. |
| Text masking (D3) [samarakoon2026hijacking] | before the model: OCR finds scene text and covers it | GPT-4o, Gemini 2.5 Flash, Qwen3-VL-32B; robot sorting, 1,800 attack trials per model | ASR 27.0 → 0.0, 29.4 → 0.0, 5.0 → 0.0 (effectiveness 100%) | all scene text is hidden, so tasks that need labels break; one OCR pass | No | Lower attack success on its own test, but the robot can no longer read any sign. |
| Two-stage verification (D2) [samarakoon2026hijacking] | a second call to the same VLM checks and corrects the plan | same as above | ASR 27.0 → 4.1, 29.4 → 2.9, 5.0 → 0.0 (effectiveness 85–100%) | task ability kept on their test; 1 extra model call | No | Close; works on closed models; slower. |
| Prompt warning (D1) [samarakoon2026hijacking] | system prompt: "ignore text in the scene" | same as above | ASR 27.0 → 6.7, 29.4 → 0.3, 5.0 → 0.0 (effectiveness 75–100%) | none reported | No | Same idea as ARGUS's system-prompt baseline, which was unreliable there. |
| TaCo-Guard (taint-aware counterfactual guard) [li2026devillens] | before the model: OCR plus an LLM scores each text region's risk and blurs risky ones | 12 VLMs; smart-glasses photos from 200+ real scenes | ASR per task, e.g. 100 → 3.8 (Gemini 2.5 Flash), 100 → 15.3 (Qwen3-VL-235B); most end at or below 15 | effect on normal use called "limited" (no number); OCR plus an LLM call | No | Close; works on closed models; tested on still photos only. |
| Token-drift feature pullback [li2026devillens] | inside the vision encoder: tokens that drift from a clean reference are pulled back toward it (forward hooks) | Llama-3.2-11B-Vision, Qwen2.5-VL-7B; navigation task | ASR 34.6 → 7.6; 3.8 → 0.0 | not reported; needs TaCo-Guard's cleaned image as a reference | No | Closest mechanism to ARGUS (edits hidden features), but early: one task, two models. |
| QuISE (query-irrelevant semantic editing) [lu2026quise] | before the model: rewrites text regions unrelated to the question and compares the answers | 4 VLMs; 3 typographic-attack tests, 8,558 attacked samples | RR 67.9–75.0% of attack-caused errors fixed | HR 0.5–1.1% of correct answers broken; several edited images and queries | not reported | Fixes about 70% of attack errors, against ARGUS's near-complete fix; works on closed models. |

**Where ARGUS lands among them.** On reported attack success, ARGUS (0.0–0.2%) sits
with the best: text masking (0%) and the trust-boundary framework (0–3.3%). It does
better than TaCo-Guard and QuISE. It is the only one that:

- covers image, video and audio;
- keeps scene text readable while stopping the model from obeying it;
- costs milliseconds rather than extra model calls or editing passes.

The close defenses are ahead in three ways: they work with closed models such as
GPT-4o, two of them were tested on physical scenes, and the trust-boundary framework
shows that helpful signs are still obeyed.

### 4.4 Attack strength without any defense

Only attacks that place text in images, screens or physical scenes.

| Attack | What it does | Success with no defense |
|---|---|---|
| ARGUS's own test [lu2026argus] | caption strip on images, spoken clip in audio, text frames in video | AIA 25.1% image, 28.2% video, 12.6% audio; 8.6% on InternVL3.5-8B |
| Paper-hijack [samarakoon2026hijacking] | printed notes on a robot's work table | ASR 27.0% GPT-4o, 29.4% Gemini 2.5 Flash, 5.0% Qwen3-VL-32B; "authority impersonation" notes up to 80% |
| Trust-boundary attacks [chang2026trustboundary] | signs in an agent's scene; optimized image noise | 28.7% best printed injection (GPT-4o); 94% white-box noise (Qwen2.5-VL) |
| Devil in the Lens [li2026devillens] | physical text seen by smart glasses | up to 96% (simulated), 60% (real world) |
| PPIA (physical prompt injection attack) [ling2026ppia] | instructions printed on physical objects, placed where the model looks | up to 98% across 10 VLMs |
| AgentTypo [li2025agenttypo] | text in web images, with its position, contrast and colour optimized | 45–83% |
| VPI-Bench (visual prompt injection benchmark) [cao2026vpibench] | instructions inside rendered app and web screens | up to 51% (CUA), 100% (BUA) |
| Pop-up attack [zhang2025popups] | adversarial pop-ups shown to computer agents | 86% average |

ARGUS's test attacks start at 6–28%; the other attacks reach 45–100%.

### 4.5 Summary: ARGUS's strengths and downsides

**Strengths**

1. **One method for image, video and audio.** Attack success ends at 0.0–0.2% in all
   six model and modality settings tested. Every close defense in §4.3 was tested on
   images only.
2. **The user still gets their answer, and scene text stays readable.** ARGUS steers
   the model back to the user's question, so the user is answered more often under
   attack (UIA<sub>inject</sub> 30.9 → 46.3 on images). The masking defenses (text
   masking, TaCo-Guard, QuISE) delete or rewrite the text instead, which also removes
   legitimate labels.
3. **No cost on clean inputs.** Normal accuracy (UIA<sub>clean</sub>) is identical to no
   defense, because the steering only switches on when the detector fires. Adversarial
   training loses about 9 points.
4. **Cheap.** About 3–6 ms per sample inside the normal forward pass, with the model's
   weights unchanged. The close alternatives add an OCR pass, one or two extra model
   calls, several edited copies of the image, or seconds to minutes of editing.

**Downsides**

1. **Never tested against an attacker who knows the defense** (an *adaptive* attack).
   None of the close defenses fully did this either; only the trust-boundary paper
   tried white-box noise, and 34–56% of attacks still succeeded. Evidence from nearby
   work is not reassuring:
   - ASTRA, a steering defense against jailbreaks, rose from 5–18% back to 58–69%
     attack success under a PGD attack that knew its steering vectors [wang2025astra];
   - one GCG-optimized suffix fooled every layer-wise linear probe at once in 93.91–99.63%
     of cases [rahman2026evasive], and ARGUS's detector and post-filter are linear probes;
   - "The Attacker Moves Second" broke 12 published defenses, most of which had
     reported near-zero attack success, with over 90% success against most of them
     [nasr2025attackermovessecond].

   ARGUS argues it is hard to bypass without access to the model's weights.
2. **Easy attacks in its test.** With no defense, ARGUS's attacks succeed only 6–28%
   of the time; the attacks in §4.4 reach 45–100%. Whether ARGUS holds against much
   stronger attacks is untested.
3. **One attack style per modality.** Training and testing both use a printed caption
   strip, a spoken clip or inserted text frames. The test changes only the trigger
   sentence and the font size, with one user instruction and one injection per sample.
   It never measures whether helpful signs are still obeyed, which the trust-boundary
   paper does.
4. **Needs model internals, retrained per model.** ARGUS reads and edits hidden states,
   so it cannot protect closed models such as GPT-4o, and its probes and vectors must
   be rebuilt for every model. The trust-boundary framework, text masking, TaCo-Guard
   and QuISE all work through a normal model interface.

---

## 5. Where the embodied extension fits

1. **Open gap.** Physical-injection papers report high attack success. They test
   static frames or photos [samarakoon2026hijacking, li2026devillens], test no
   defense [burbano2025chai, ling2026ppia], or report that existing defenses fall
   short [li2025shawshank, li2026pi3d]. Embodied defenses act on prompts, inputs or
   actions [chang2026trustboundary, wang2025robosafe, zhang2024safeembodai].
   Activation steering has run in closed loop on a robot only for control
   [haon2025mechsteer]. No published work uses activation steering as an injection
   defense **inside a closed perception-action loop**, where a missed detection at
   one step changes the next observation.
2. **Baselines to run alongside ARGUS in the PyBullet loop:**
   - ARGUS's four in-repo baselines (system prompt, ignore, noise, remove).
   - Attention Tracker [hung2024attentiontracker]: training-free and cheap on a T4.
   - A TaskTracker-style detection-only probe [abdelnabi2025tasktracker] that halts
     on detection. This isolates what steering adds over simply stopping.
   - A simplified perception/judgment split [chang2026trustboundary] as the
     black-box embodied baseline.
   - QuISE-style text sanitization [lu2026quise] as a model-agnostic input
     baseline.
   - Report bypass rate, false positives, task completion and added latency, as in
     [wang2026guardrailbench], so the results line up with embodied guardrails.
3. **Attack variants worth adding** beyond the current `*_card` / `*_overlay`
   conditions:
   - PPIA-style placement: card position, distance and viewing angle.
   - Authority-impersonation wording from [samarakoon2026hijacking].
   - A benign-sign control: cards with helpful safety cues, following
     [chang2026trustboundary]. This measures over-refusal.
   - Environmental instructions from SHAWSHANK-BENCH [li2025shawshank], and
     pose-optimized cards in the spirit of PI3D [li2026pi3d].
4. **Threats to validity to report:**
   - ARGUS's probes are trained on caption-strip images. Scene text under
     perspective, occlusion and lighting is a distribution shift for both the
     detector and the steering vectors.
   - Probe detection may key on context rather than the injection [kale2026probes].
     Use held-out splits (leave one scene or card style out) [fomin2026benchmarkslie]
     and nuisance-matched controls [li2026aucprobes].
   - An adaptive attacker who knows the probes can optimize inputs so the hidden
     states look benign [rahman2026evasive]. The white-box attacks in §3.1 give the
     means to try this on images, and suffix augmentation [rahman2026evasive] is a
     defense to test against it.

---

## Appendix A. Broad metric tables

These are the full tables behind §4, including the text-only and jailbreak-only papers that §4 leaves out. The metric names are defined in §4.1. As in §4, every row comes from a different paper and setting.

### A.1 All defenses, including text-only and jailbreak-only (each in its own setting)

"Before → after" is the undefended → defended value reported by each paper.

| Defense | Threat · setting | Attack metric, before → after | Utility / side effect | Adaptive attack reported? |
|---|---|---|---|---|
| **ARGUS** [lu2026argus] | multimodal IPI · Qwen2-VL-7B, own benchmark | AIA 25.1 → 0.1 (image), 28.2 → 0.1 (video), 12.6 → 0.0 (audio) | UIA<sub>clean</sub> unchanged; +3–6 ms | **No.** The paper argues it is hard to bypass without weight access. |
| ASTRA [wang2025astra] | VLM jailbreak, adversarial images · Qwen2-VL | jailbreak ASR 67–76 → 5–18 | "little performance drop" on benign inputs | **Yes:** with full knowledge, ASR 58–69 on Qwen2-VL, 14–25 on MiniGPT-4 |
| Trust boundary [chang2026trustboundary] | embodied scene signs · GPT-4o agent | misleading-sign follow rate 8.0 → 0.7 | helpful-cue follow rate 65 → 99 | **Partly:** white-box noise injections on Qwen still succeed 34–56% |
| Pictionary [zhang2026renderbeforereading] | text IPI in agent tools · 10 frontier models | DirectInject ASR, e.g. 75.0 → 0.0 (GPT-5.4 mini), 98.2 → 55.4 (Gemini 3.1 Pro) | AgentDojo utility under attack mostly kept (e.g. GPT-5.4 mini 71.4 → 64.9) | **Yes:** adaptive and human red-team ASR from 9.5% to 97.8%, depending on model |
| ICON [wang2026icon] | text agent IPI | ASR 0.4% | over 50% task-utility gain vs. baselines | not in abstract |
| InstructDetector [wen2025instructdetector] | text IPI · BIPIA | ASR → 0.03%; detection 99.6% (in-domain) / 96.9% (out-of-domain) | detection only, no repair | not in abstract |
| TaskTracker [abdelnabi2025tasktracker] | text task drift | near-perfect ROC AUC (out-of-distribution) | detection only | not in abstract |
| Attention Tracker [hung2024attentiontracker] | text IPI | AUROC up to +10.0% over prior detectors | training-free; detection only | not in abstract |
| Probing & Steering [anon2026probingsteering] | text IPI · Qwen2.5-3B | compliance 0.58 → 0.26 (probe AUC 0.90) | mainly task-replacement attacks | not in abstract |
| MELON [zhu2025melon] | text agent IPI · AgentDojo, GPT-4o | average ASR 16.1 → 0.24 | benign utility 80.4 → 68.0 (MELON-Aug: 76.3); doubles agent calls | no |
| StruQ [chen2025struq] | text IPI · Llama | Completion-Real 96 → 0; TAP 97 → 9 | AlpacaEval 67.2 → 67.6 | **Yes:** GCG optimization still reaches 58% |
| SecAlign [chen2025secalign] | text IPI | ASR below 10%, including attacks stronger than those seen in training | "similar utility" | **Partly:** stronger unseen attacks, per the abstract |
| Spotlighting [hines2024spotlighting] | text IPI · GPT models | ASR above 50 → below 2 | "minimal impact" | no |
| CaMeL [debenedetti2025camel] | text agent IPI · AgentDojo | provable security on solved tasks | 77% of tasks solved vs. 84% undefended | by design |
| ECSO [gou2024ecso] | VLM jailbreak · LLaVA-1.5-7B | safety improved 37.6% (MM-SafetyBench SD+OCR) and 71.3% (VLSafe) | "consistently maintaining utility" | no |
| ETA [ding2024eta] | VLM jailbreak | unsafe rate reduced by 87.5% (cross-modality attacks) | 96.6% win-or-tie in GPT-4 helpfulness | no |
| Immune [ghosal2025immune] | MLLM jailbreak · LLaVA-1.6 | ASR reduced by 57.82% vs. base model | slower decoding (reward-guided) | no |
| VLMGuard [fang2026vlmguard] | malicious VLM prompts | AUROC +5.39% over prior best | detection only | no |
| NullSteer [zhu2026nullsteer] | VLM jailbreak · MiniGPT-4 | average ASR reduced by over 15 points | general benchmarks comparable to the original model | not in abstract |
| Dyslexify [hufe2025dyslexify] | typographic attack · CLIP | up to +22.06% accuracy on typographic ImageNet-100 | under 1% drop on clean ImageNet-100 | not in abstract |
| QuISE [lu2026quise] | typographic attack · 4 VLMs | recovery rate 67.9–75.0% | harm rate 0.5–1.1% | not in abstract |
| RoboSafe [wang2025robosafe] | hazardous instructions · embodied agents | risk occurrence −36.8% vs. leading baselines | near-original task performance | not in abstract |
| J-DAPT [marchiori2025jdapt] | robot jailbreaks · driving, maritime, quadruped | detection accuracy near 100% | minimal overhead | not in abstract |
| RIPA firewall [dorzhiev2026ripa] | sensor-channel injection · ROS 2 | 0% ASR on known patterns | 0/20 false positives on benign commands | **Yes:** 10.2% bypass under obfuscated payloads |
| Paper-hijack mitigations [samarakoon2026hijacking] | robot sorting · GPT-4o, Gemini, Qwen3-VL | prompt defense 75–100% effective; two-stage check 85–100%; text masking 100% | masking blocks tasks that need in-scene labels | no |

### A.2 All attacks, including text and jailbreak attacks

| Attack | Setting | Reported success |
|---|---|---|
| ARGUS benchmark (for scale) [lu2026argus] | typographic / TTS / video-text IPI, undefended | AIA 25.1% image, 28.2% video, 12.6% audio (Qwen2-VL-7B / Kimi-Audio); 8.6% InternVL3.5-8B |
| Image Hijacks [bailey2024imagehijacks] | white-box adversarial images · LLaVA | over 80% for all four attack types |
| FigStep [gong2025figstep] | typographic jailbreak · 6 open VLMs | 82.5% average |
| Invisible Injections [pathade2025invisible] | steganographic text · 8 VLMs | 24.3% ± 3.2 overall, up to 31.8% |
| Reading Between the Pixels [balakrishnan2026readingpixels] | typographic · GPT-4o, Claude Sonnet 4.5 | text-in-image 36% / 47% vs. image attacks 8% / 22% |
| VPI-Bench [cao2026vpibench] | GUI visual injection | up to 51% (computer-use agents), 100% (browser agents) |
| Pop-ups [zhang2025popups] | OSWorld, VisualWebArena | 86% average; task success −47% |
| EIA [liao2025eia] | web agent PII theft | up to 70% (specific PII), 16% (full request) |
| PPIA [ling2026ppia] | physical typography · 10 VLMs | up to 98% |
| Paper-hijack [samarakoon2026hijacking] | robot sorting, 5,670 trials | 27.0% GPT-4o, 29.4% Gemini 2.5 Flash, 5.0% Qwen3-VL-32B |
| Trust-boundary attacks [chang2026trustboundary] | GPT-4o image-editing agent; Qwen (white-box noise) | 28.7% (best structured injection); 94% (white-box noise) |
| CHAI [burbano2025chai] | drones, driving, real robot car | "outperforms state-of-the-art attacks" (abstract gives no single number) |
| Evasive suffix vs. activation probes [rahman2026evasive] | attacks drift *detectors* · Phi-3, Llama-3-8B | evades all layer-wise probes at once: 93.91% / 99.63% |
| AgentTypo [li2025agenttypo] | adaptive typographic injection · 7 model backends | 45–83% |
| CrossInject [wang2025crossinject] | cross-modal agent injection | +30.1 points or more over prior attacks |
| Devil in the Lens [li2026devillens] | physical text · smart-glasses VLMs | up to 96% (simulated), 60% (real world) |
| Shawshank [li2025shawshank] | environmental instructions · 6 VLMs | compromises all six; beats 11 prior methods |
| PINA [liu2026pina] | navigation agents | 87.5% average |
| RIPA [dorzhiev2026ripa] | ROS 2 sensor channels · 5 LLMs | 100% on Llama-3.3-70B; 0% on two 7–8B models for direct override |
| RoboPAIR [robey2024robopair] | robot jailbreak, incl. Unitree Go2 | "often 100%" |
| POEF [lu2024intentbehavior] | robot jailbreak, behaviour level | 80% executable harmful behaviour |
