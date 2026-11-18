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
ARGUS's first stage alone. A recent caution [kale2026probes] shows that
near-perfect probe AUROC can come from probes learning conversation context rather
than the injection. This applies directly to ARGUS's detector and should be
controlled for.

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

- **Trust-boundary separation** [chang2026trustboundary] is the only
  embodied-agent defense we found. A perception agent rates how trustworthy each
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

### 2.4 Defense comparison

| Method | Modalities | Mechanism | Access / training | Detects | Repairs | Main difference from ARGUS |
|---|---|---|---|:-:|:-:|---|
| **ARGUS** [lu2026argus] | image, video, audio | probe → utility-decoupled adaptive steering → post-filter | white-box; probes + vectors per model | ✓ | ✓ | — |
| ASTRA [wang2025astra] | image | ablation-derived steering vectors, projection | white-box; few forward passes | – | ✓ | jailbreak, not IPI; always on |
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
| Trust boundary [chang2026trustboundary] | embodied, scene signs | perception agent + judgment agent | black-box; extra calls | ✓ | ✓ | only embodied defense; prompt and agent level |
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
and cross-modal image-only perturbation [yang2026crossmodal].

### 3.2 GUI and web agents

VPI-Bench [cao2026vpibench] places injections in pop-ups, chats and emails across
five platforms. It reports deception rates of up to 51% for computer-use agents and
100% for browser-use agents, and that a vigilant system prompt has limited effect.
Pop-up attacks [zhang2025popups] and EIA [liao2025eia] show that small injected UI
elements redirect agents or leak private data.

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
- BadRobot [zhang2025badrobot] jailbreaks embodied LLM agents.

None of these papers evaluates a representation-level defense.

### 3.4 Attack coverage of ARGUS

| Attack | Channel | Attacker access | Matches ARGUS's training/eval data? |
|---|---|---|---|
| Abusing Images & Sounds [bagdasaryan2023abusing], Image Hijacks [bailey2024imagehijacks], Cross-modal PI [yang2026crossmodal] | adversarial pixels / audio | white-box (gradients) | **No.** Not evaluated. |
| Invisible Injections [pathade2025invisible] | steganographic text | black-box | **No.** |
| FigStep [gong2025figstep] | typographic image | black-box | **Yes.** Closest to ARGUS's caption-strip images. |
| VPI-Bench [cao2026vpibench], Pop-ups [zhang2025popups], EIA [liao2025eia] | GUI screenshots | black-box | **Partly.** Rendered text, but in a GUI domain. |
| CHAI [burbano2025chai], PPIA [ling2026ppia], Paper-hijack [samarakoon2026hijacking], SceneTAP [cao2024scenetap] | text physically in the scene | black-box | **No.** This is what [`embodied/`](../embodied/) tests. |

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
   attacker. ARGUS's 0.1% has no such counterpart. Its own argument is that steering
   is hard to bypass *without weight access*, so a white-box attacker who targets
   the probes is outside its threat model.
2. **A modest undefended baseline.** Undefended AIA in ARGUS's benchmark is 6–28%.
   Physical and GUI attacks reach 70–100% (PPIA, VPI-Bench, pop-ups). A
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

1. **Open gap.** Physical-injection papers report high attack success but either
   test static frames [samarakoon2026hijacking] or no defense at all
   [burbano2025chai, ling2026ppia]. The only embodied defense
   [chang2026trustboundary] is black-box and agent-level. No published work runs
   an activation-steering defense **inside a closed perception-action loop**,
   where a missed detection at one step changes the next observation.
2. **Baselines to run alongside ARGUS in the PyBullet loop:**
   - ARGUS's four in-repo baselines (system prompt, ignore, noise, remove).
   - Attention Tracker [hung2024attentiontracker]: training-free and cheap on a T4.
   - A TaskTracker-style detection-only probe [abdelnabi2025tasktracker] that halts
     on detection. This isolates what steering adds over simply stopping.
   - A simplified perception/judgment split [chang2026trustboundary] as the
     black-box embodied baseline.
3. **Attack variants worth adding** beyond the current `*_card` / `*_overlay`
   conditions:
   - PPIA-style placement: card position, distance and viewing angle.
   - Authority-impersonation wording from [samarakoon2026hijacking].
   - A benign-sign control: cards with helpful safety cues, following
     [chang2026trustboundary]. This measures over-refusal.
4. **Threats to validity to report:**
   - ARGUS's probes are trained on caption-strip images. Scene text under
     perspective, occlusion and lighting is a distribution shift for both the
     detector and the steering vectors.
   - Probe detection may key on context rather than the injection [kale2026probes].
   - An adaptive attacker who knows the probes can optimize hidden states to look
     benign. The white-box attacks in §3.1 give the means to try this.
