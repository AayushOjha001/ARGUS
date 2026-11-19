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
asked for. ASTRA is also always on, while ARGUS gates the intervention behind a detector
and verifies it afterwards.

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
| ASTRA [wang2025astra] | image | ablation-derived steering vectors, projection | white-box; few forward passes | – | ✓ | jailbreak, not IPI; always on |
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

## 4. Reported metrics: where ARGUS stands out

> **Read this first.** Every number below is copied from the paper that reports it.
> Each paper uses its own models, benchmark, attack set and metric definition. Only
> the numbers *inside* §4.2 are measured under the same conditions. Use §4.3–§4.4 to
> see the scale of effects in each setting, not to rank methods against each other.

### 4.1 Metrics used

- **ASR (attack success rate):** share of attacked inputs where the attacker's goal
  is achieved. Lower is better for a defense.
- **ARGUS's names for it:** **AIA** (Attacker Instruction Accuracy) counts outputs
  that contain the attacker's expected answer, the strict equivalent of ASR.
  **AIFR** (Attacker Instruction Following Rate) counts *any attempt* to follow the
  injection, even an incorrect one. The paper's Table 1 labels it "AIFA".
- **UIA (User Instruction Accuracy):** share of outputs that correctly do the
  user's task. ARGUS reports it on injected inputs (**UIA<sub>inject</sub>**,
  "does the user still get an answer under attack") and on clean inputs
  (**UIA<sub>clean</sub>**, "does the defense hurt normal use").
- **Detection metrics:** AUROC or accuracy of flagging injected inputs. These say
  nothing about whether the final answer is correct.
- **Jailbreak papers** (ASTRA, ECSO, ETA, Immune) measure harmful-content rates or
  toxicity, not task hijacking.

### 4.2 ARGUS against its own baselines (same benchmark)

Values are AIA↓ / UIA<sub>inject</sub>↑ / UIA<sub>clean</sub>↑ in %. Image and video
use Qwen2-VL-7B; audio uses Kimi-Audio-7B. The last column is extra inference time
per sample in ms (image / video / audio). Source: ARGUS Table 1 [lu2026argus].

| Approach | Image | Video | Audio | Extra time (ms) |
|---|---|---|---|---|
| No defense | 25.1 / 30.9 / 49.6 | 28.2 / 25.4 / 37.6 | 12.6 / 45.6 / 65.7 | 0 / 0 / 0 |
| System prompt | 10.7 / 38.2 / 42.7 | 26.9 / 25.4 / 37.0 | 27.9 / 7.5 / 63.8 | 6 / 15 / 5 |
| Ignore prompt | 31.5 / 24.5 / 49.4 | 32.9 / 21.8 / 36.1 | 28.0 / 24.3 / 65.7 | 2 / 3 / 2 |
| Gaussian noise | 7.6 / 34.3 / 46.8 | 9.6 / 18.7 / 23.3 | 0.0 / 42.8 / 41.0 | 1 / 2 / 2 |
| Removal (editing model) | 0.0 / 48.5 / 49.3 | 1.5 / 32.5 / 32.9 | n/a | 12,885 / 574,121 / n/a |
| Adversarial training (DPO) | 2.3 / 41.1 / 40.7 | 1.6 / 35.9 / 37.2 | 1.4 / 55.8 / 60.9 | 0 / 0 / 0 |
| **ARGUS** | **0.1 / 46.3 / 49.6** | **0.1 / 37.8 / 37.6** | **0.0 / 58.0 / 65.7** | 3 / 6 / 4 |
| ARGUS without post-filter | 4.3 / 46.4 / 49.6 | 0.8 / 38.5 / 37.6 | 1.0 / 58.2 / 65.7 | 3 / 6 / 4 |

On three more models (ARGUS Table 3), ARGUS reaches AIA 0.0% on InternVL3.5-8B
(image, from 8.6%), 0.2% on Qwen2.5-VL-7B (video, from 15.4%) and 0.0% on
Qwen2-Audio-7B (audio, from 6.4%), with UIA<sub>clean</sub> within 0.2 points of no
defense in each case.

What the table shows:

- **Only ARGUS gets near-zero attack success and unchanged clean accuracy in all
  three modalities.** Every other row gives up one of the two. Prompt defenses
  are inconsistent: the system prompt cuts image AIA to 10.7 but raises audio AIA
  (12.6 → 27.9) and collapses audio UIA<sub>inject</sub> to 7.5; the "ignore" prompt
  makes image AIA worse (25.1 → 31.5). Noise and Removal hurt clean accuracy on video (37.6 → 23.3 and
  32.9). Adversarial training lowers clean image accuracy (49.6 → 40.7).
- **Clean accuracy is untouched because of the detection gate.** On clean inputs the
  detector rarely fires, so the model runs unmodified (the paper reports near-100%
  detection accuracy).
- **ARGUS repairs the answer rather than refusing.** UIA<sub>inject</sub> rises above
  no defense (image 30.9 → 46.3; video 25.4 → 37.8): the user's task is still
  done under attack.
- **Cost:** 3–6 ms per sample, against 12.9 s (image) and 574 s (video) for
  Removal, and no weight change, unlike adversarial training.
- **The post-filter trades a little utility for safety:** without it, image AIA
  rises from 0.1 to 4.3.

### 4.3 Headline results of other defenses (each in its own setting)

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

### 4.4 Attack strength reported by attack papers

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

### 4.5 Summary: what ARGUS does differently, and what it has not shown

**Where ARGUS stands out**

1. **One method, three modalities.** It is the only defense here with results on
   image, video *and* audio, with AIA 0.0–0.2% in all six model/modality settings.
   The closest steering defense (ASTRA) covers images only, and every text-only
   steering paper covers text only.
2. **The user's task survives the attack.** Most defenses either refuse, halt, or
   only flag the input (TaskTracker, InstructDetector, VLMGuard, MELON's
   detection). ARGUS steers the model back, so UIA<sub>inject</sub> goes up
   (image 30.9 → 46.3). Among the papers here, only ICON reports the same
   repair-rather-than-refuse goal, mainly for text agents (its abstract mentions an
   extension to multimodal agents).
3. **No cost on clean inputs.** Detection gating keeps UIA<sub>clean</sub> equal to
   no defense. MELON loses 12 points of benign utility; adversarial training in
   ARGUS's own table loses 9.
4. **Cheap and training-light.** About 3–6 ms per sample, one forward pass, model
   weights untouched. Compare: StruQ and SecAlign need fine-tuning; MELON runs the
   agent twice; ETA uses best-of-N sampling; trust boundary adds two agent calls;
   Removal costs seconds to minutes.

**Where ARGUS is weaker or untested**

1. **No adaptive attack.** ASTRA (58–69% under adaptive attack on Qwen2-VL), StruQ
   (58% under GCG), Pictionary (red-team union) and trust boundary (34–56% under
   white-box noise) all report how far their numbers fall against a stronger
   attacker. ARGUS's 0.1% has no such counterpart. The most direct warning is
   [rahman2026evasive]: one optimized suffix evades every layer-wise drift probe at
   once (93.91–99.63%), and ARGUS's detection and post-filter stages are the same
   kind of linear probe. ARGUS's own argument is that steering is hard to bypass
   *without weight access*, so a white-box attacker who targets the probes is
   outside its threat model.
2. **A modest undefended baseline.** Undefended AIA in ARGUS's benchmark is 6–28%.
   Physical, GUI and agent attacks reach 45–100% (PPIA, VPI-Bench, pop-ups,
   AgentTypo, PINA, Devil in the Lens). A
   near-zero result is easier to reach from a 25% start, so the gap to these
   stronger attacks is untested.
3. **A narrow attack family.** Training and testing both use rendered caption
   strips, TTS clips and inserted video text. Generalization is checked with a
   different trigger sentence and font size at test time, plus one user
   instruction and one injection per sample (the paper's stated limitation).
   Adversarial-pixel, steganographic and physical-scene attacks are not evaluated.
4. **White-box and per-model.** Probes and vectors must be retrained for each model,
   whereas the trust-boundary and Pictionary approaches work on closed APIs.

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
