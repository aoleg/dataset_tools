## Prompting

We trained the model on captions written to one template. Prompts in the same template produce the most coherent output. "Soviet street, 1970s" works, with generic details. Three kinds of detail need spelling out: text, vehicles and school uniforms. The rest is structure.

### Text: give it, in Cyrillic, in CAPITALS

Every object that carries text in real life carries text in the output: shop signs, banners, posters, street signs, book and notebook covers, license plates, the route number and destination board of a bus or tram, the titles on an airliner, the lettering on a cosmonaut's helmet. If the prompt does not say what the text is, the model fills the space with pseudo-letters.

Write the text in Cyrillic and in CAPITALS; lowercase renders poorly. Attach it to its object: "a bus with the route number 17 and the destination board reading ВОКЗАЛ", "a grocery shop with the sign ГАСТРОНОМ above the door", "an Il-62 with АЭРОФЛОТ along the fuselage", "a cosmonaut helmet with СССР in red letters above the visor".

Use what you know of the period. Common signs: ПРОДУКТЫ, ГАСТРОНОМ, УНИВЕРМАГ, КНИГИ, АПТЕКА, МОЛОКО, ХЛЕБ, КИНО, ПОЧТА, РЕМОНТ ОБУВИ. Common banners: СЛАВА КПСС, МИРУ МИР, СЛАВА ТРУДУ. A school diary reads ДНЕВНИК, a notebook ТЕТРАДЬ. License plates before 1981: two pairs of digits and three letters, "27-14 ММБ". From 1981: a letter, four digits and two letters, "В 0417 МО". Airliners carry АЭРОФЛОТ. Trucks often carry the enterprise name on the door.

### Vehicles: name the model

"A Soviet car" gives a blend of every car in the training set. Name the model: GAZ-21 Volga, GAZ-24 Volga, GAZ-M20 Pobeda, Moskvich-408, Moskvich-412, VAZ-2101 Zhiguli, ZAZ-965, ZAZ-968, ZIL-130 truck, GAZ-53 truck, LiAZ-677 bus, PAZ-672 bus, Ikarus-280 articulated bus, ZiU-9 trolleybus, Tatra T3 tram, Tu-154 and Il-62 airliners. In a color prompt give the color: "a beige GAZ-24 Volga".

### School scenes: describe the uniform

A Soviet pupil dressed in one of three ways. The model knows all three and needs to be told which.

- Everyday: girls in a brown dress with a black apron and a white collar; boys in a dark navy or grey suit with a light blue shirt.
- Formal (the first of September, examinations, class photographs): girls in a brown dress with a white apron, white lace collar and cuffs, white bows; boys in a dark navy or grey suit with a white shirt.
- Pioneer ceremonial (line-ups, parades): white shirt, red pioneer tie, no apron, no jacket.

Neckwear goes by age, and the prompt has to say which. Ages 7 to 9 are Octobrists: a collar at the neck and an Octobrist star badge on the chest. Ages 10 to 14 are pioneers: a red pioneer tie and a pioneer badge. From 14: a shirt collar and a Komsomol badge. Write "red pioneer tie" in a color prompt and "pioneer tie" in a black-and-white one. Describe what is at the neck. Never write "without a tie" or "no tie"; the word alone pulls a tie into the picture.

Call the children "schoolgirl", "schoolboy" or "schoolchildren", give ages, and introduce the clothing with "wears a Soviet school uniform consisting of ...". The model was trained on that sentence.

### Follow the caption structure

Every training caption has the same shape. Prompts in that shape read most reliably:

1. Medium and subject first: "A color photograph from the 1970s in Kiev, Soviet Union, of ..." or "A black-and-white photograph from 1958 in Leningrad of ...". Say "color" or "black-and-white". The training set is half of each, and the model will not guess your way.
2. People with ages, clothing, pose and expression; then objects, setting, and where things are relative to each other.
3. In a color prompt, tie a color to each major element: "a red tram", "a dark green Moskvich-408", "a pale blue sky". In a black-and-white prompt use no color words; say light or dark.
4. Camera angle and lighting last: "Eye-level view, bright daylight with strong shadows", "High-angle view, overcast light".
5. Quality words at the very end, if wanted. "Scratches, fading and a magenta color cast" gives an archival print; "sharp, clean" gives a clean one. For a cinematic look, start with "A color film still from a modern production set in the Soviet Union".

Two to five sentences, under 160 words, present tense. Russian in the prompt works; the structure above is what the model learned.

### A prompt-writing prompt

The captions came from a vision-language model running the instructions below, here turned around to write prompts from an idea. Give it to any capable LLM with a one-line idea and it returns a prompt in the trained form.

```text
You write prompts for a text-to-image model fine-tuned on Soviet-era photographs from the 1920s to the 1990s. The user gives a short idea. Turn it into one prompt that follows these rules exactly.

Form:
- Plain English, present tense, one paragraph of 2 to 5 sentences, at most 160 words.
- Start with the medium and the main subject: "A color photograph from the 1970s in Kiev, Soviet Union, of ..." or "A black-and-white photograph from 1958 in Leningrad of ...". If the user did not say color or black-and-white, choose black-and-white for the 1920s to 1950s and color from the 1960s on, and say which. Name the city and the country.
- Then describe the people with an age for each person or an age range for a group, their clothing, pose and expression; then the objects and setting and where they are relative to each other.
- In a color prompt name the color of each major element as you describe it ("a red tram", "a dark green Moskvich-408", "a pale blue sky"); tie each color to its object and do not list colors separately. In a black-and-white prompt use no color words; describe tones as light or dark.
- End with the camera angle ("Eye-level view", "Low-angle view", "High-angle view", "Close-up view") and the lighting. For an archival look add quality words after that: scratches, fading, a color cast, grain. For a clean image add "sharp, clean". For a cinematic look start instead with "A color film still from a modern production set in the Soviet Union".

Text:
- Every object that carries text in real life gets its text in the prompt, in Cyrillic and in CAPITALS, attached to the object: shop signs, banners, posters, street signs, book, notebook and diary covers, license plates, the route number and destination board of a bus, tram or trolleybus, the titles on an airliner, the lettering on a cosmonaut's helmet. If the user gave no text, invent text that fits the period and place. Common signs: ПРОДУКТЫ, ГАСТРОНОМ, УНИВЕРМАГ, КНИГИ, АПТЕКА, МОЛОКО, ХЛЕБ, КИНО, ПОЧТА. Common banners: СЛАВА КПСС, МИРУ МИР, СЛАВА ТРУДУ. Airliners carry АЭРОФЛОТ. Cosmonaut helmets carry СССР. A school diary reads ДНЕВНИК. License plates before 1981 look like "12-34 ММБ"; from 1981 like "В 0417 МО".

Vehicles:
- Name every vehicle by model: GAZ-21 Volga, GAZ-24 Volga, GAZ-M20 Pobeda, Moskvich-408, Moskvich-412, VAZ-2101 Zhiguli, ZAZ-965, ZAZ-968, ZIL-130 truck, GAZ-53 truck, LiAZ-677 bus, PAZ-672 bus, Ikarus-280 bus, ZiU-9 trolleybus, Tatra T3 tram, Tu-154 or Il-62 airliner. Choose a model that existed in the decade of the prompt. Give its color in a color prompt.

School scenes:
- Call the children "schoolgirl", "schoolboy", "schoolgirls", "schoolboys" or "schoolchildren", with ages. If they wear pioneer ties, call them "pioneer schoolgirl", "pioneer schoolboy" or "pioneer schoolchildren".
- Introduce the clothing with "She wears a Soviet school uniform consisting of ...", "He wears a Soviet school uniform consisting of ..." or "They wear Soviet school uniforms consisting of ...", and list: neckwear first; for girls the dress color (brown), the apron color (black for everyday, white for formal), collar and cuffs, hair bows; for boys the suit color (dark navy or grey), the shirt color (light blue for everyday, white for formal), belt and buckle; then the badge.
- Neckwear by age: 7 to 9, a white collar and an Octobrist star badge; 10 to 14, a red pioneer tie ("pioneer tie" in black-and-white) and a pioneer badge; 14 and up, a shirt collar and a Komsomol badge. Pioneer ceremonial dress is a white shirt with a red pioneer tie, no apron and no jacket. Never write "without a tie", "no tie" or "no neckwear"; describe what is at the neck.

Never mention the user, the viewer, "this image" or "this photograph shows". Do not use "appears to", "seems" or "likely". Output the prompt only.
```

### Examples

A color photograph from the 1970s in Moscow, Soviet Union, of a street in front of a grocery shop. A cream LiAZ-677 bus with the route number 24 and the destination board reading СОКОЛЬНИКИ stands at the curb; a beige GAZ-24 Volga with the plate 27-14 ММБ is parked behind it. Above the shop door a red sign reads ГАСТРОНОМ, and a woman aged 40 to 50 in a grey coat and headscarf carries a string bag with oranges past the window. Eye-level view, overcast daylight, faded colors with a magenta cast.

A black-and-white photograph from 1958 in Leningrad of a GAZ-M20 Pobeda taxi on a wet embankment. The driver, a man aged 30 to 40 in a dark cap and jacket, leans on the open door and smokes; a checkered band runs along the car's side and a roof sign reads ТАКСИ. Behind him the dark granite parapet and a row of light-walled buildings recede into mist. Eye-level view, diffuse morning light, light scratches and fading.

A color photograph from the 1980s in Kiev, Soviet Union, of two schoolchildren on the first of September, a 7-year-old schoolgirl and an 8-year-old schoolboy walking along a brick wall with bouquets of red carnations. She wears a Soviet school uniform consisting of a white lace collar, a brown dress, a white apron and large white bows in her pigtails, with an Octobrist star badge on the apron. He wears a Soviet school uniform consisting of a white shirt with an open collar, a dark navy suit with a buckled belt and an Octobrist star badge on the chest. They hold hands and look at each other, smiling. Eye-level view, bright daylight with strong shadows.

A color photograph from the 1970s in Minsk, Soviet Union, of a pioneer line-up in a school courtyard. Twenty pioneer schoolchildren aged 10 to 13 stand in two rows in Soviet school uniforms consisting of white shirts, red pioneer ties and pioneer badges, the girls with dark skirts and the boys with dark trousers. A red banner on the school wall reads МИРУ МИР, and a drummer in the front row holds a red drum. High-angle view, bright midday light.

A color photograph from the 1960s in the Soviet Union of a cosmonaut in a white pressure suit beside a Vostok capsule mock-up in a hangar. The white helmet carries СССР in red letters above the visor, and a red star is painted on the grey capsule. The cosmonaut, a man aged 30 to 35, holds the helmet under his arm and looks into the distance with a calm expression. Low-angle view, even artificial light, sharp, clean.

A black-and-white photograph from the 1930s in Kharkiv of a tram stop on a cobbled square. A light-colored tram with the number 7 on its front and a side board reading ЦЕНТР stands at the stop while men in dark caps and women in long coats board it; a kiosk on the right carries the sign ГАЗЕТЫ. The dark mass of a constructivist office block rises in the background. Eye-level view, flat overcast light, grain and fading.

A color photograph from the 1970s in Riga, Soviet Union, of a kitchen with a woman aged 25 to 35 in a flowered housedress pouring tea at a small table. A white Saratov refrigerator stands against the wall with a transistor radio on top, a calendar with a red 1976 hangs by the window, and a jar of jam and a loaf of dark bread sit on an oilcloth with a blue pattern. Eye-level view, soft window light, faded colors.

A color photograph from the 1980s in Tashkent, Soviet Union, of an Aeroflot Tu-154 on the apron with АЭРОФЛОТ along the white fuselage and a blue cheat line below the windows. Passengers aged 20 to 60 in light summer clothing walk toward the boarding stairs with bags, and a yellow GAZ-53 fuel truck stands under the wing. Eye-level view, hard afternoon sunlight, sharp, clean.
