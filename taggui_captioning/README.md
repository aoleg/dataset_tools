# TagGUI captioning prompt

A prompt for [TagGUI](https://github.com/jhc13/taggui) and a local vision-language model. It turns a photo and its short editor's caption into an English text-to-image prompt, and keeps the editor's caption after it. It is written for the datasets that [telegram_dataset](../telegram_dataset/README.md) makes from channels of historical photographs.

## Why

The editor's caption knows what the model cannot see: the place, the year, the event and the names of the people. The model sees what the editor did not write: the people, their clothing and pose, the setting, the camera angle and the light. The prompt asks for both in one paragraph.

TagGUI replaces `{{tags}}` in the prompt with the image's existing caption, that is, the text of its `.txt` file. Here that is the cleaned Russian caption from `telegram_dataset`, sometimes followed by English quality words. The model is not told it gets "tags" or "metadata". The prompt says what the text is: a caption by a channel editor that may be empty, may describe a whole album rather than this photo, and may contain remarks or questions. Without this, the model treats an album caption as facts about each photo, and it names people who are not in the frame.

The model writes a text-to-image prompt, not a description. When it is asked to describe, it writes about the caption ("The caption describes the first television in a village") and about the photo ("This photograph shows ..."). A generation prompt starts with the subject, which is also how people write prompts for the trained model. One worked example in the prompt fixes the form. Its label, "not the content", stops the model from copying it.

## How

In TagGUI, set these before you start the run:

1. Paste the prompt below into the auto-captioner prompt field.
2. Set the caption position to "Insert before first tag". The English prompt then comes first and the editor's caption stays at the tail of the file. With the default, the new text is appended after the Russian caption.
3. Set "Remove tag separators in caption" to OFF. TagGUI treats a caption as a list of tags separated by commas. With this option on, it removes every comma from the generated text so that the text stays one tag, and the result reads "High-angle view bright daylight". The commas cannot be put back afterwards. With the option off, TagGUI shows the English split into many tags, but it writes them back joined with the separator, so the file text is the same as the model's text. Check one image after you change this setting.

Test the prompt on a few dozen images before a full run: some with their own caption, some with a caption copied from an album, some with a question as the caption, and some with no caption.

TagGUI puts its separator between the new text and the old caption, so the file reads "... bright daylight., Вид на башни-близнецы, 1985". Fix this join in a pass after the run.

The trainer reads the whole file as one prompt. Ostris AI Toolkit does not split a caption at line breaks, and its `shuffle_tokens` and `token_dropout` options split at commas, which scrambles a sentence caption, so keep them off.

## The prompt

```
Write a text-to-image prompt that would generate this historical photograph.

Existing caption: {{tags}}
The Russian part was written by a channel editor. It may be empty, it may describe a whole series of photos rather than this one, and it may contain remarks or questions. English words in it, if any, describe image quality.

Rules:
- Plain English, present tense, one paragraph of 2 to 5 sentences, at most 160 words.
- Start with the medium and the main subject, for example "A black-and-white photograph of ..." or "A color poster of ...". Put the decade, place, event and names of people from the Russian caption into this first sentence, in standard English spellings. Leave out the editor's opinions, greetings and questions.
- Then describe what is visible: people with approximate age (an age range for groups), clothing, pose and expression; objects and setting and where they are relative to each other.
- End with the camera angle and lighting, then the quality words from the caption and any visible scratches, tears, stains or fading. If there are none, say nothing about quality.
- Name a person only if the caption identifies them and the image agrees. If the caption gives no decade, estimate it from clothing, vehicles and architecture.
- Transcribe any readable signs or anything that contains text, including Cyrillics.
- Never mention the caption, the editor, the viewer, "this image" or "this photograph shows". Do not use "appears to", "seems" or "likely". Describe only what is present.

Example of the form, not the content:
A black-and-white photograph from the 1960s of the first television set in a village. A group of men aged 20 to 50, some shirtless and others in light-colored shirts and trousers, sit and stand on a concrete porch outside a small white-walled building with a corrugated roof. They all look toward a small television set on a wooden stand to the left. Eye-level view, bright daylight.
```

## Example result

```
A color photograph from 1985 of the Twin Towers in New York City. The towering skyscrapers stand in the distance at the end of a city street lined with multi-story brick and stone buildings. In the foreground, people in mid-80s casual attire walk on the sidewalks, a man rides a bicycle, and several vintage cars and a red truck are parked or driving along the asphalt road. Eye-level perspective, bright daylight., Вид на башни-близнецы, 1985
```
