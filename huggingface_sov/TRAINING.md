## How we trained

This is a checkpoint based on Krea 2 RAW, trained on 25,000 captioned Soviet-era photographs, capturing the look and content of the USSR from the 1920s to the 1990s: streets, interiors, vehicles, clothing, school uniforms, signage.

### Dataset

We started from about 29,000 photographs; the initial dataset provided by V-Zh. Each came with editorial metadata: the year or range of years, the place down to the city and often the street, the event, and names where known. That matters for captions: the model learned "Khreshchatyk in Kiev" and "the Moskvich-408" instead of "a street" and "a car".

Cleaning took it to 25,000:

- removed everything below the 512² bucket;
- deduplicated by hash and by embedding similarity;
- cropped white borders and removed watermarks;
- ran FBCNN (Jiang et al., ICCV 2021) over the worst multi-generation JPEGs to remove blocking without inventing detail;
- dropped images where the main subject was a face smaller than the compression artifacts.

Composition of the final set: half color, half black-and-white (about 12,000 each). About 40% of the images are from the 1970s; the rest spread over the Soviet period, with more from the 1980s than the 1960s and progressively fewer going back. By subject: people in everyday life (work, study, street), outdoor environments, transport, interiors, military, objects and goods, group photographs, and a small set of Soviet retro-futurism. Niche sub-collections cover Kiev and Odessa by decade, specific car and bus models (GAZ-21, GAZ-24, Moskvich-408, GAZ-M20, LiAZ, Ikarus, ZIL, trolleybuses), pedal cars, and household objects. A dedicated school uniform set of 891 images was built later, for reasons described below.

### Captioning

Captions are in English, 2 to 5 sentences, under 160 words, fixed structure: medium and subject with date and place first, then people with ages, clothing, pose and expression, then objects and setting, then camera angle and lighting, then any damage or color cast. Color photographs name the color of each major element; black-and-white captions contain no color words. All readable text in the image is transcribed in Cyrillic as written.

5,797 images were hand-picked and captioned manually. The rest were captioned by a 31B vision-language model from a prompt that encodes the structure above and feeds it the archive's editorial record for each image. School uniform images got a second, stricter prompt that fixes the uniform vocabulary: "schoolgirl" and "schoolboy" as subject nouns, "wears a Soviet school uniform consisting of ..." as the clothing sentence, neckwear first, "red pioneer tie" when present and a positive description of the collar when absent, Octobrist, pioneer or Komsomol badge by name. Both prompts are published in the Prompting section, turned around to write prompts from an idea.

### Training setup

- Base model: Krea 2 RAW (bf16), trained with the base quantized to fp8; the adapter applies to Krea 2 Turbo unchanged.
- Trainer: Ostris AI Toolkit.
- Network: LoKr, factor 8, full-rank second block; about 240M trainable parameters in the DiT and text-fusion blocks, 390 MB in bf16. Patchify, output projection, timestep and text embedders left untouched (the trainer's `transformer_only` default).
- Optimizer: Automagic v3, start LR 1e-4. The controller let the LR drift down over the run: 5e-5 at step 47,000, 3.7e-5 at step 74,000.
- Timesteps: sigmoid, balanced.
- Caption dropout 0.05; latents cached to disk.

### Schedule

1. Six epochs at 512² area, batch 4, all images.
2. Four epochs at 1024², batch 2, all images; images smaller than 1024² train at their native size.
3. One epoch with timesteps biased to low noise, on the sharp subset.
4. A corrective pass for school uniforms (below), balanced timesteps, 9,000 steps at batch 2.

### Corrections during training

**Black-and-white bias.** The caption said "a color photograph" and the model still produced black-and-white for the 1950s to 1970s unless told twice. The base model's prior for those decades is greyscale and three tokens at the start of a long caption did not override it. We recaptioned the color half so that every major element carries its color ("a red tram", "a brown dress"), re-weighted the black-and-white half for the remaining epochs, and measured the result as the share of black-and-white outputs per decade on a fixed grid of prompts and seeds.

**Pioneer ties.** The original set had 250 photographs of pioneers in red ties and 66 of pupils without them, which is the historical proportion and the wrong training proportion: every school scene came out with ties, Octobrist-age children included. We built a balanced set of 891 images covering everyday, formal and ceremonial uniforms with and without ties, recaptioned all of them with the stricter uniform prompt, and ran a corrective pass with the set at about 30% of samples (`num_repeats` 12 against the 25,000 main set). The before and after grids are what fixed checkpoint selection: the earlier checkpoint puts ties and Octobrist stars on the same seven-year-olds, the corrected one does not.

### What we checked

Fixed seeds, fixed prompt sets, counted outcomes: decade × color mode, ties present or absent for Octobrist-age and pioneer-age prompts, hands and small faces in group photographs, and a set of niche prompts (named car models, Kiev landmarks, Soviet goods) against the base model with the same seeds. Comparing two checkpoints on single seeds misled us more than once; eight seeds per prompt was the minimum that gave stable counts.

### Limitations

The model wants detailed prompts. It knows the period, but it needs to be told which part of it to draw: a named car model, a named uniform variant, the text on every sign in Cyrillic capitals. Given "a Soviet car" it blends several; given "school uniform" without the neckwear it picks one. This is an adapter, not a full fine-tune, and the 240M parameters hold the concepts without always holding them apart.

Fine detail follows the training data. Faces in wide shots, hands, and small lettering are the first to degrade, and the archival medium (grain, fading, a magenta cast on 1970s color film) leaks into prompts that did not ask for it. Quality words at the end of the prompt steer this in both directions.

### What worked

The adapter generalizes. Concepts learned from a few hundred images (specific cars, pedal cars, the Ogonyok cover layout) combine with places, decades and people they were never photographed with, and the base model's composition and anatomy survive. LoKr at factor 8 trained faster than a rank-32 LoRA on the same data and produced cleaner anatomy; the LyCORIS authors' finding that full-dimension LoKr is the most fitted of the adapters at equal parameter count matches what we saw, as does their note that it leaks less at larger factors. Automagic v3 removed the learning rate as a decision: the start value was a launch point, and the controller found the anneal on its own.
