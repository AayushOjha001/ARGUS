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
At inference it adds only cheap hooks. Its in-paper baselines are a vigilant system
prompt, an "ignore instructions in the media" prompt, Gaussian noise on the media,
and an oracle that removes the injected content
([`src/argus/baselines.py`](../src/argus/baselines.py)). Its attack data are
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
| MLLM-Protector / ETA / Immune | image | output check + rewrite / aligned decoding | mixed; Immune slows decoding | ✓ | ✓ | harmful-content focus |
| VLMGuard / VLMShield | image + text | malicious-prompt classifier | black-box | ✓ | – | detection only |
| Trust boundary [chang2026trustboundary] | embodied, scene signs | perception agent + judgment agent | black-box; extra calls | ✓ | ✓ | only embodied defense; prompt and agent level |
| Pictionary [zhang2026renderbeforereading] | text tool outputs | render text as image | black-box | – | ✓ | opposite modality assumption |
| StruQ / SecAlign / Spotlighting / MELON / CaMeL | text | data/instruction separation, fine-tuning, system design | mixed | mixed | ✓ | assume a separable data channel |

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
| [bagdasaryan2023abusing], [bailey2024imagehijacks], [yang2026crossmodal] | adversarial pixels / audio | white-box (gradients) | **No.** Not evaluated. |
| [pathade2025invisible] | steganographic text | black-box | **No.** |
| FigStep [gong2025figstep] | typographic image | black-box | **Yes.** Closest to ARGUS's caption-strip images. |
| VPI-Bench, pop-ups, EIA | GUI screenshots | black-box | **Partly.** Rendered text, but in a GUI domain. |
| CHAI, PPIA, Paper-hijack, SceneTAP | text physically in the scene | black-box | **No.** This is what [`embodied/`](../embodied/) tests. |

---

## 4. Where the embodied extension fits

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
