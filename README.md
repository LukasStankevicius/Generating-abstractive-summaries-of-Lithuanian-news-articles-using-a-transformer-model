# Generating abstractive summaries of Lithuanian news articles using a transformer model
This repository contains the code used in this work.

# Simple usage
```python
from transformers import pipeline
name= "LukasStankevicius/t5-base-lithuanian-news-summaries-175"
my_pipeline = pipeline(task="text2text-generation", model=name, framework="pt")
```
Given the following article body from [15min](https://www.15min.lt/24sek/naujiena/lietuva/tarp-penkiu-rezultatyviausiu-tsrs-rinktines-visu-laiku-zaideju-trys-lietuviai-875-1380030):
```
text = """
Latvijos krepšinio legenda Valdis Valteris pirmadienį socialiniame tinkle pasidalino statistika, kurios viršūnėje yra Arvydas Sabonis.
1982 metais TSRS rinktinėje debiutavęs 222 cm ūgio vidurio puolėjas su raudona apranga sužaidė 52 rungtynes, per kurias rinko po 15,6 taško. Tai pats aukščiausias rezultatyvumo vidurkis tarp visų sovietų komandai atstovavusių žaidėjų, skaičiuojant tuos, kurie sužaidė ne mažiau nei 50 rungtynių. Antras šioje rikiuotėje kitas buvęs Kauno „Žalgirio“ krepšininkas Rimas Kurtinaitis. Jis debiutavo TSRS rinktinėje vėliau nei Sabas, – 1984 metais, bet irgi sužaidė 52 mačus. R.Kurtinaitis pelnė po 15 taškų. 25-ių rezultatyviausių žaidėjų sąrašu pasidalinęs latvis V.Valteris, pelnęs po 13,8 taško, yra trečias.
Ketvirtas yra iš Kazachstano kilęs Valerijus Tichonenka, pelnęs po 13,7 taško per 79 rungtynes. Rezultatyviausią visų laikų TSRS rinktinės penketą uždaro Modestas Paulauskas. Lietuvos krepšinio legenda pelnė po 13,6 taško per 84 mačus.
Dešimtuke taip pat yra Oleksandras Volkovas (po 13,5 taško), Sergejus Belovas (12,7), Anatolijus Myškinas (po 12,3), Vladimiras Tkačenka (11,7) ir Aleksandras Salnikovas (11,4). Dvyliktas šiame sąraše yra Valdemaras Chomičius, vidutiniškai rinkęs po 10 taškų, o keturioliktas dar vienas buvęs žalgirietis Sergejus Jovaiša (po 9,8 taško). Šarūno Marčiulionio rezultatyvumo vidurkis turėjo būti aukštesnis, bet jis sužaidė mažiau nei 50 rungtynių. Kaip žinia, Lietuvai išsilaisvinus ir atkūrus Nepriklausomybę, visi minėti mūsų šalies krepšininkai, išskyrus karjerą jau baigusį M.Paulauską, užsivilko žalią aprangą ir atstovavo savo tėvynei.
A.Sabonis pagal rezultatyvumo vidurkį yra pirmas – jis Lietuvos rinktinei pelnė po 20 taškų. Antras pagal taškų vidurkį yra Artūras Karnišovas, rinkęs po 18,2 taško ir pelnęs iš viso daugiausiai taškų atstovaujant Lietuvos rinktinei (1453).
Tarp žaidėjų, kurie sužaidė bent po 50 oficialių rungtynių Lietuvos rinktinėje, trečią vietą užima Ramūnas Šiškauskas (po 12,9), ketvirtąją Linas Kleiza (po 12,7 taško), o penktas – Saulius Štombergas (po 11,1 taško). Daugiausiai rungtynių Lietuvos rinktinėje sužaidęs ir daugiausiai olimpinių medalių (3) su ja laimėjęs Gintaras Einikis rinko po 9,6 taško, o pirmajame trejete pagal rungtynių skaičių ir pelnytus taškus esantis Šarūnas Jasikevičius pelnė po 9,9 taško.
"""
```
The summary can be obtained by:
```python
text = ' '.join(text.strip().split())
my_pipeline(text)[0]["generated_text"]
```
Output from above would be:
Lietuvos krepšinio federacijos (LKF) prezidento Arvydo Sabonio rezultatyvumo vidurkis yra aukščiausias tarp visų Sovietų Sąjungos rinktinėje atstovavusių žaidėjų, skaičiuojant tuos, kurie sužaidė bent po 50 oficialių rungtynių.

# Advanced usage
```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
name= "LukasStankevicius/t5-base-lithuanian-news-summaries-175"
tokenizer = T5Tokenizer.from_pretrained(name)
model = T5ForConditionalGeneration.from_pretrained(name)
def decode(x):
    return tokenizer.decode(x, skip_special_tokens=True)
```
Given the following article body from [15min](https://www.15min.lt/24sek/naujiena/lietuva/tarp-penkiu-rezultatyviausiu-tsrs-rinktines-visu-laiku-zaideju-trys-lietuviai-875-1380030):
```python
text = """
Latvijos krepšinio legenda Valdis Valteris pirmadienį socialiniame tinkle pasidalino statistika, kurios viršūnėje yra Arvydas Sabonis.
1982 metais TSRS rinktinėje debiutavęs 222 cm ūgio vidurio puolėjas su raudona apranga sužaidė 52 rungtynes, per kurias rinko po 15,6 taško. Tai pats aukščiausias rezultatyvumo vidurkis tarp visų sovietų komandai atstovavusių žaidėjų, skaičiuojant tuos, kurie sužaidė ne mažiau nei 50 rungtynių. Antras šioje rikiuotėje kitas buvęs Kauno „Žalgirio“ krepšininkas Rimas Kurtinaitis. Jis debiutavo TSRS rinktinėje vėliau nei Sabas, – 1984 metais, bet irgi sužaidė 52 mačus. R.Kurtinaitis pelnė po 15 taškų. 25-ių rezultatyviausių žaidėjų sąrašu pasidalinęs latvis V.Valteris, pelnęs po 13,8 taško, yra trečias.
Ketvirtas yra iš Kazachstano kilęs Valerijus Tichonenka, pelnęs po 13,7 taško per 79 rungtynes. Rezultatyviausią visų laikų TSRS rinktinės penketą uždaro Modestas Paulauskas. Lietuvos krepšinio legenda pelnė po 13,6 taško per 84 mačus.
Dešimtuke taip pat yra Oleksandras Volkovas (po 13,5 taško), Sergejus Belovas (12,7), Anatolijus Myškinas (po 12,3), Vladimiras Tkačenka (11,7) ir Aleksandras Salnikovas (11,4). Dvyliktas šiame sąraše yra Valdemaras Chomičius, vidutiniškai rinkęs po 10 taškų, o keturioliktas dar vienas buvęs žalgirietis Sergejus Jovaiša (po 9,8 taško). Šarūno Marčiulionio rezultatyvumo vidurkis turėjo būti aukštesnis, bet jis sužaidė mažiau nei 50 rungtynių. Kaip žinia, Lietuvai išsilaisvinus ir atkūrus Nepriklausomybę, visi minėti mūsų šalies krepšininkai, išskyrus karjerą jau baigusį M.Paulauską, užsivilko žalią aprangą ir atstovavo savo tėvynei.
A.Sabonis pagal rezultatyvumo vidurkį yra pirmas – jis Lietuvos rinktinei pelnė po 20 taškų. Antras pagal taškų vidurkį yra Artūras Karnišovas, rinkęs po 18,2 taško ir pelnęs iš viso daugiausiai taškų atstovaujant Lietuvos rinktinei (1453).
Tarp žaidėjų, kurie sužaidė bent po 50 oficialių rungtynių Lietuvos rinktinėje, trečią vietą užima Ramūnas Šiškauskas (po 12,9), ketvirtąją Linas Kleiza (po 12,7 taško), o penktas – Saulius Štombergas (po 11,1 taško). Daugiausiai rungtynių Lietuvos rinktinėje sužaidęs ir daugiausiai olimpinių medalių (3) su ja laimėjęs Gintaras Einikis rinko po 9,6 taško, o pirmajame trejete pagal rungtynių skaičių ir pelnytus taškus esantis Šarūnas Jasikevičius pelnė po 9,9 taško.
"""
```
And generation parameters ([documentation](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate), [explanation](https://github.com/huggingface/blog/blob/master/notebooks/02_how_to_generate.ipynb)):
```python
g_kwargs = dict(max_length=512, num_beams=10, no_repeat_ngram_size=2, early_stopping=True)
```
The summary can be obtained by:
```python
text = ' '.join(text.strip().split())
input_dict = tokenizer(text,  padding=True, return_tensors="pt", return_attention_mask=True)
output = model.generate(**input_dict, **g_kwargs)
list(map(decode, output.tolist()))[0]
```
Output from above would be:
```python
"Lietuvos krepšinio federacijos (LKF) prezidento Arvydo Sabonio rezultatyvumo vidurkis yra aukščiausias tarp visų Sovietų Sąjungos rinktinėje atstovavusių žaidėjų, skaičiuojant tuos, kurie sužaidė bent po 50 oficialių rungtynių."
```
If you do a lot of compute you can take advantage of GPU. Before line with `model.generate` additionally insert these lines:
```python
input_dict = {key:value.to("cuda:0") for key, value in input_dict.items()}
model = model.to("cuda:0")
```
