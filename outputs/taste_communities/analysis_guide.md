# Taste communities (network analysis) — Analysis guide

This document lists what to check in the networks and reports **pre-computed** metrics to support your data story.

---

## 1. Data and outputs in this folder

- **Performer network**: nodes = contestants (performers); edge between two if they covered at least one same original artist; edge weight = number of shared originals. Files: `performer_shared_original_network.html`, `performer_shared_original_edges.csv`, `performer_shared_original_top_pairs.csv`.
- **Cooccurrence network**: nodes = original artists; edge between two if they co-occurred in at least one edition; edge weight = co-occurrence count. Files: `cooccurrence_network.html`, `cooccurrence_edges.csv`, `cooccurrence_top_pairs.csv`.
- Optional: node color = gender when `artist_gender_by_artist.csv` exists.

---

## 2. What to check in the networks (visual)

- **Performer network**: Do performers cluster by taste? Are there clear communities? When gender is on: do clusters look more male/female/mixed? Who are the "bridge" performers (high betweenness)?
- **Cooccurrence network**: Which original artists sit in the centre? Do artists from the same era/genre cluster?

---

## 3. Computed metrics — Performer network

### Top 20 performers by weighted degree (taste overlap)

1. **Michele Bravi** — 17
2. **Diodato** — 12
3. **Madame** — 12
4. **Rose Villain** — 12
5. **Gaia** — 11
6. **Annalisa** — 11
7. **Ghemon** — 11
8. **Patty Pravo** — 10
9. **Coma Cose** — 10
10. **Colapesce Dimartino** — 9
11. **Ermal Meta** — 9
12. **Pinguini Tattici Nucleari** — 9
13. **Malika Ayane** — 9
14. **Irene Grandi** — 8
15. **Francesca Michielin** — 8
16. **Valerio Scanu** — 8
17. **Giusy Ferreri** — 8
18. **Negramaro** — 8
19. **Bugo** — 8
20. **Tredici Pietro** — 8

---

### Top 15 bridge performers (betweenness centrality)

1. **Michele Bravi** — 0.2017
2. **Malika Ayane** — 0.1557
3. **Gaia** — 0.1482
4. **Fulminacci** — 0.1259
5. **Colapesce Dimartino** — 0.1179
6. **Levante** — 0.1113
7. **Gianluca Grignani** — 0.1097
8. **Gianni Morandi** — 0.1097
9. **Annalisa** — 0.1049
10. **Irama** — 0.0923
11. **Tredici Pietro** — 0.0923
12. **Pinguini Tattici Nucleari** — 0.0670
13. **Coma Cose** — 0.0587
14. **[Elodie, Achille Lauro]** — 0.0439
15. **Ghemon** — 0.0382

---

### Top 15 performer pairs by shared originals

1. **Patty Pravo** and **Irene Grandi** — 2 shared original artist(s)
2. **Diodato** and **Madame** — 2 shared original artist(s)
3. **Leo Gassmann** and **Alessio Bernabei** — 2 shared original artist(s)
4. **Raf** and **Nina Zilli** — 1 shared original artist(s)
5. **Dargen D'Amico** and **Patty Pravo** — 1 shared original artist(s)
6. **Dargen D'Amico** and **Irene Grandi** — 1 shared original artist(s)
7. **Patty Pravo** and **Annalisa** — 1 shared original artist(s)
8. **Patty Pravo** and **Lara Fabian** — 1 shared original artist(s)
9. **Patty Pravo** and **Michele Bravi** — 1 shared original artist(s)
10. **Patty Pravo** and **Francesco Renga** — 1 shared original artist(s)
11. **Patty Pravo** and **Gaia** — 1 shared original artist(s)
12. **Patty Pravo** and **[Mahmood, Blanco]** — 1 shared original artist(s)
13. **Patty Pravo** and **Joan Thiele** — 1 shared original artist(s)
14. **Irene Grandi** and **Annalisa** — 1 shared original artist(s)
15. **Irene Grandi** and **Lara Fabian** — 1 shared original artist(s)

---

## 4. Community detection (performer network)

Found **20** communities (greedy modularity).

### Community 1 (size: 27)

**Members:** Aiello, Bugo, Chiara, Colla Zio, Coma Cose, Francesca Michielin, Fulminacci, Ghemon, Giusy Ferreri, Junior Cally, Levante, Lodovica Comello, Malika Ayane, Negramaro, Nek, Pinguini Tattici Nucleari, Ricchi e Poveri, Riki, Rkomi, Rosa Chemical, Rose Villain, Samurai Jay, Shari, Valerio Scanu, [Ditonellapiaga, Rettore]....

**Gender mix:** F: 8; M: 16; Mixed: 3.

**Top 10 original artists covered by this community:**
- Lucio Battisti (8)
- Vasco Rossi (6)
- Gianna Nannini (6)
- Zucchero Fornaciari (5)
- Caterina Caselli (4)
- Mina (4)
- Ricchi e Poveri (3)
- Daniele Silvestri (3)
- Nilla Pizzi (2)
- Rino Gaetano (2)

---

### Community 2 (size: 16)

**Members:** Achille Lauro, Alessio Bernabei, Bianca Atzei, Chiello, Elodie, Gianluca Grignani, Giordana Angi, Irama, Leo Gassmann, Loredana Bertè, Luchè, Noemi, Ultimo, [Elodie, Achille Lauro], [Highsnob, Hu], [Shablo con Guè Joshua, Tormento].

**Gender mix:** F: 5; M: 9; Mixed: 2.

**Top 10 original artists covered by this community:**
- Riccardo Cocciante (6)
- Luigi Tenco (5)
- Loredana Bertè (4)
- Eros Ramazzotti (4)
- Edoardo Bennato (3)
- Neffa (3)
- Gianluca Grignani (3)
- Mia Martini (2)
- Sottotono (2)
- Francesco Guccini (1)

---

### Community 3 (size: 13)

**Members:** Al Bano, Bresh, Clementino, Colapesce Dimartino, Diodato, Francesco Gabbani, Ghali, Giovanni Truppi, Madame, Marcella Bella, Moreno, Nayt, Olly.

**Gender mix:** F: 2; M: 11.

**Top 10 original artists covered by this community:**
- Adriano Celentano (7)
- Fabrizio De André (7)
- Toto Cutugno (2)
- Ghali (2)
- Franco Battiato (1)
- Lorella Cuccarini (1)
- Tricarico (1)

---

### Community 4 (size: 13)

**Members:** Anna Tatangelo, Bluvertigo, Brunori Sas, Ermal Meta, Gianni Morandi, Irene Fornaciari, Lucio Corsi, Mahmood, Random, Stadio, Tommaso Paradiso, Tosca, Tredici Pietro.

**Gender mix:** F: 3; M: 10.

**Top 10 original artists covered by this community:**
- Lucio Dalla (7)
- Domenico Modugno (4)
- Gianni Morandi (4)
- Jovanotti (3)
- Gigliola Cinquetti (1)
- Jvke (1)

---

### Community 5 (size: 13)

**Members:** Annalisa, Ariete, Dargen D'Amico, Francesco Renga, Gaia, Irene Grandi, Joan Thiele, Lara Fabian, Marco Masini, Michele Bravi, Patty Pravo, Simone Cristicchi, [Mahmood, Blanco].

**Gender mix:** F: 7; M: 6.

**Top 10 original artists covered by this community:**
- Ornella Vanoni (7)
- Patty Pravo (3)
- Franco Battiato (3)
- Gino Paoli (3)
- Matia Bazar (2)
- Francesco Nuti (1)
- Gianna Nannini (1)
- Giorgio Faletti (1)
- Luigi Tenco (1)
- Lucio Battisti (1)

---

### Community 6 (size: 5)

**Members:** Arisa, Massimo Ranieri, Neffa, Rocco Hunt, [Giovanni Caccamo, Deborah Iurato].

**Gender mix:** F: 1; M: 3; Mixed: 1.

**Top 10 original artists covered by this community:**
- Pino Daniele (5)
- Renato Carosone (2)
- Rita Pavone (1)
- Fiorella Mannoia (1)

---

### Community 7 (size: 3)

**Members:** Dear Jack, Orietta Berti, [Bugo, Morgan].

**Gender mix:** F: 1; M: 2.

**Top 10 original artists covered by this community:**
- Sergio Endrigo (3)
- Quartetto Cetra (1)

---

### Community 8 (size: 3)

**Members:** Fabrizio Moro, Fiorella Mannoia, Lorenzo Fragola.

**Gender mix:** F: 1; M: 2.

**Top 10 original artists covered by this community:**
- Francesco De Gregori (3)
- Ron (1)
- Pooh (1)
- Mannoia (1)
- Gabbani (1)

---

### Community 9 (size: 3)

**Members:** Anna Oxa, Le Vibrazioni, Paola Turci.

**Gender mix:** F: 2; M: 1.

**Top 10 original artists covered by this community:**
- Anna Oxa (3)
- Paul McCartney (1)

---

### Community 10 (size: 3)

**Members:** Eddie Brock, Il Tre, Maninni.

**Gender mix:** M: 3.

**Top 10 original artists covered by this community:**
- Fabrizio Moro (5)
- Ermal Meta (1)

---

### Community 11 (size: 3)

**Members:** Enrico Nigiotti, Willie Peyote, [Noemi, Tony Effe].

**Gender mix:** M: 2; Mixed: 1.

**Top 10 original artists covered by this community:**
- Samuele Bersani (2)
- Franco Califano (2)
- Simone Cristicchi (1)

---

### Community 12 (size: 2)

**Members:** Nina Zilli, Raf.

**Gender mix:** F: 1; M: 1.

**Top 10 original artists covered by this community:**
- Massimo Ranieri (2)
- Nik Kershaw (1)

---

### Community 13 (size: 2)

**Members:** J-Ax, [Biggio, Mandelli].

**Gender mix:** M: 2.

**Top 10 original artists covered by this community:**
- Cochi e Renato (2)

---

### Community 14 (size: 2)

**Members:** Giorgia, Rancore.

**Gender mix:** F: 1; M: 1.

**Top 10 original artists covered by this community:**
- Elisa (2)
- Giorgia (1)
- Adele (1)

---

### Community 15 (size: 2)

**Members:** Fasma, Lazza.

**Gender mix:** M: 2.

**Top 10 original artists covered by this community:**
- Nesli (2)

---

### Community 16 (size: 2)

**Members:** La Rappresentante di Lista, La Sad.

**Gender mix:** M: 1; Mixed: 1.

**Top 10 original artists covered by this community:**
- Donatella Rettore (2)
- The Ronettes (1)

---

### Community 17 (size: 2)

**Members:** Gianmaria, Lo Stato Sociale.

**Gender mix:** M: 2.

**Top 10 original artists covered by this community:**
- Afterhours (2)

---

### Community 18 (size: 2)

**Members:** Sal Da Vinci, Will.

**Gender mix:** M: 2.

**Top 10 original artists covered by this community:**
- Michele Zarrillo (2)

---

### Community 19 (size: 2)

**Members:** Modà, Renga Nek.

**Gender mix:** M: 1; Unknown: 1.

**Top 10 original artists covered by this community:**
- Francesco Renga (3)
- Nek (2)
- Le Vibrazioni (1)

---

### Community 20 (size: 2)

**Members:** Ana Mena, [Maria Antonietta, Colombre].

**Gender mix:** F: 1; Mixed: 1.

**Top 10 original artists covered by this community:**
- Jimmy Fontana (2)
- Alan Sorrenti (1)
- Julio Iglesias (1)

---


## 5. Computed metrics — Cooccurrence network

### Top 15 original-artist pairs by co-occurrence weight

1. **Ornella Vanoni** and **Adriano Celentano** — 4
2. **Ornella Vanoni** and **Lucio Dalla** — 4
3. **Adriano Celentano** and **Franco Battiato** — 4
4. **Gianna Nannini** and **Fabrizio De André** — 4
5. **Pino Daniele** and **Lucio Battisti** — 4
6. **Riccardo Cocciante** and **Fabrizio De André** — 4
7. **Riccardo Cocciante** and **Lucio Dalla** — 4
8. **Fabrizio De André** and **Lucio Battisti** — 4
9. **Fabrizio De André** and **Lucio Dalla** — 4
10. **Lucio Battisti** and **Lucio Dalla** — 4
11. **Sergio Endrigo** and **Ornella Vanoni** — 3
12. **Sergio Endrigo** and **Caterina Caselli** — 3
13. **Sergio Endrigo** and **Adriano Celentano** — 3
14. **Ornella Vanoni** and **Luigi Tenco** — 3
15. **Ornella Vanoni** and **Caterina Caselli** — 3

### Top 15 original artists by weighted degree (co-occurrence)

1. **Lucio Dalla** — 163
2. **Fabrizio De André** — 159
3. **Adriano Celentano** — 142
4. **Luigi Tenco** — 135
5. **Lucio Battisti** — 133
6. **Ornella Vanoni** — 129
7. **Riccardo Cocciante** — 118
8. **Gianna Nannini** — 107
9. **Caterina Caselli** — 101
10. **Vasco Rossi** — 99
11. **Pino Daniele** — 99
12. **Zucchero Fornaciari** — 99
13. **Franco Battiato** — 97
14. **Ricchi e Poveri** — 90
15. **Daniele Silvestri** — 86

---

## 6. Story angles you can build

- **Taste communities**: Use the communities above; name them by top covered originals or genre; describe size and gender mix.
- **Bridge performers**: Use the betweenness list; say who connects different taste clusters.
- **Gender and taste**: Compare gender mix across communities; note homophily or mixing.
- **Canon and co-occurrence**: Use central artists and top pairs to describe the "Sanremo cover canon" and typical pairings.
- **Strongest ties**: Use the top performer pairs to highlight similar repertoires or key editions.
