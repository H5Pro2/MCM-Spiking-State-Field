# Umsetzungsplan: MCM-Spiking-State-Field

## 1. Zieldefinition

Ziel ist die Umsetzung eines **spikenden inneren Zustandsmodells**, das die folgenden Funktionsketten realisiert:

- Wahrnehmung
- neuronale Verarbeitung
- kontinuierlicher MCM-Feldzustand
- Clustering
- Grübeln / Denken
- eigene Kontextbildung
- Kontextlernen
- Kontextwiedergabe als Datenausgang
- Reflexion
- Rückführung
- innere Wahrnehmung
- Selbstregulierung

Das Ziel ist **kein komplettes Brain-Scale-System**, sondern eine **saubere, aufbaubare Kernarchitektur**.

## 2. Entwurfsprinzipien

1. **Kontinuierlicher Zustandsraum**
   - keine harten Maürn im Feld
   - keine ontologisch getrennten Klassen
   - Interpretationszonen nur als spätere Auswertung

2. **Spikende Verarbeitung**
   - interne Repräsentationen werden über spikende Populationen getragen
   - der innere Zustand bleibt rekurrent aktiv

3. **Trennung von Feldkern und Interpretation**
   - Feldkern: `X`, `x(t)`, `rho(x,t)`
   - Interpretation: `Phi(x)` oder lesbare Clusterlabels

4. **Stimulus -> interner Zustand -> weiterer Prozess**
   - kein direktes Reiz-Reaktions-System

5. **Regulation als Kernfunktion**
   - das System muss überauslenkung, Schleifenbildung und Instabilität aktiv beeinflussen koennen

## 3. Formale Grundlage

### 3.1 Kernraum
`X = [-3, +3]`

### 3.2 Einzelzustand
`x(t) in X`

### 3.3 Feldzustand
`rho(x,t) >= 0`

Normierung:
`integral_{-3}^{+3} rho(x,t) dx = 1`

### 3.4 Rückführung / Drift
`v(x) = -k x`, mit `k > 0`

### 3.5 Potenzial
`V(x) = 0.5 * a * x^2`, mit `a > 0`

### 3.6 Psychologische Zustandsdynamik
`dx/dt = -k x + I(t) + eta(t)`

### 3.7 Feldgleichung
`partial rho / partial t = -partial_x (v(x) rho(x,t)) + D partial_x^2 rho(x,t)`

### 3.8 Spannung
`S(x) = |x|`

Optional:
`S(x) = |x|^alpha`, mit `alpha > 1`

### 3.9 Feldgroessen
Mittelwert:
`mu_x(t) = integral_{-3}^{+3} x * rho(x,t) dx`

Varianz:
`Var_x(t) = integral_{-3}^{+3} (x - mu_x(t))^2 * rho(x,t) dx`

### 3.10 Symbolische Leseschicht
`Phi: X -> A`

Wichtig:
`Phi` ist **nur Readout**, nicht die eigentliche Dynamik.

## 4. Gesamtarchitektur

### 4.1 Modul U - Wahrnehmung
**Aufgabe**
- externe Reize in interne Eingangsvektoren übersetzen

**Eingang**
- sensorische Daten
- Ereignisse
- Kontext von aussen

**Ausgang**
- `u(t)`

**Beispielkanäle**
- valence
- novelty
- relevance
- uncertainty
- social salience

---

### 4.2 Modul N - Spikende neuronale Verarbeitung
**Aufgabe**
- Reize in spikende Aktivitätsmuster übersetzen
- rekurrente Aktivität aufrechterhalten

**Zustand**
- Spike-Muster
- firing rates
- Ensemble-Zustände

**Form**
- Nengo-Ensembles
- rekurrente Verbindungen
- später optional Subnetze für verschiedene Funktionen

**Ausgang**
- `n(t)`

---

### 4.3 Modul X - MCM-Einzelzustand
**Aufgabe**
- aus `n(t)` einen kontinuierlichen inneren Feldwert ableiten

**Zustand**
- `x(t)`

**Update**
`dx/dt = -k_reg(t) * x + W_in * n(t) + r_mem(t) + eta(t)`

**Bedeutung**
- `-k_reg(t) * x` = adaptive Rückführung
- `W_in * n(t)` = Wahrnehmung wirkt auf Feld
- `r_mem(t)` = Replay / Denken
- `eta(t)` = Fluktuation

---

### 4.4 Modul RHO - Feldrekonstruktion
**Aufgabe**
- aus den neuronalen Aktivitäten eine kontinuierliche Dichte über dem Raum rekonstruieren

**Zustand**
- `rho(x,t)`

**Praktische Umsetzung**
- jedem Ensemble / jeder Unterpopulation wird ein bevorzugter Bereich `p_i` im Intervall `[-3, +3]` zugeordnet
- aus Aktivität `a_i(t)` wird Dichte rekonstruiert:
  `rho(x,t) = sum_i a_i(t) * K(x - p_i)`

**Nutzen**
- Mehrfachaktivität sichtbar
- Cluster / Peaks sichtbar
- Feldunruhe berechenbar

---

### 4.5 Modul C - Clustering
**Aufgabe**
- wiederkehrende Muster in `(x, rho, Var_x, Verlauf)` erkennen

**Eingang**
- Zeitfenster von `x(t)`
- Zeitfenster von `rho(x,t)`
- Feldgroessen

**Ausgang**
- Clusterobjekte `C_j`

**Clusterstruktur**
- `mu_j` = Schwerpunkt
- `Sigma_j` = Streuung
- `strength_j` = Auftretenshäufigkeit / Gewicht
- `age_j`
- `stability_j`

**Methoden**
- zürst offline: DBSCAN / HDBSCAN / Peak-Matching
- später online: streaming clustering

---

### 4.6 Modul M - Cluster-Gedächtnis
**Aufgabe**
- häufige oder relevante Cluster speichern
- spätere Wiederaktivierung ermoeglichen

**Zustand**
- Gedächtnisbank `Mem = {C_1, ..., C_k}`

**Funktionen**
- Stärkung häufiger Muster
- Vergessen irrelevanter Muster
- Replay-Ausloeser

---

### 4.7 Modul G - Grübeln / Denken / Replay
**Aufgabe**
- interne Aktivität ohne neün Aussenreiz weiterlaufen lassen

**Eingang**
- Cluster-Gedächtnis
- aktüller Kontext
- aktülle Feldlage

**Ausgang**
- `r_mem(t)`

**Replay-Regel**
`r_mem(t) = sum_j gate_j(t) * K(x - mu_j)`

**Interpretation**
- Denken = geordnete interne Wiederaufnahme
- Grübeln = selbstverstärkende Replay-Schleife mit schwacher Aufloesung

---

### 4.8 Modul K - Kontextbildung
**Aufgabe**
- aus Verlauf, Feldlage, Clustern und Reizgeschichte einen eigenen internen Kontext erzeugen

**Zustand**
- `c(t)`

**Update**
`c(t+1) = lambda_c * c(t) + f(x(t), rho(x,t), C_t, u(t), q(t))`

**Kontext enthält**
- letzte Feldlage
- Richtungsveränderung
- dominante Cluster
- Rückkehr-/Drift-Tendenz
- Reizhistorie
- Stabilitätsgrad

---

### 4.9 Modul L - Kontextlernen
**Aufgabe**
- lernen, welche Konfigurationen zu welchen Feldmustern und Regulationsbedarfen führen

**Lernobjekte**
- Clusterübergänge
- Kontext -> Cluster
- Kontext -> Regulationserfolg
- Kontext -> Replay-Risiko

**Minimalstart**
- übergangsmatrizen
- cluster conditionals
- Erfolgszähler für Regler

**später**
- lernende Verbindungen in Nengo
- plastische Gewichte für Kontextkopplung

---

### 4.10 Modul F - Reflexion
**Aufgabe**
- aktüllen Zustand mit Vorzustand, Verlauf und bekannten Mustern vergleichen

**Zustand**
- `rfl(t)`

**Funktion**
`rfl(t) = h(x(t), dx/dt, c(t), q(t-1), match(C_t))`

**Leistung**
- Schleifen erkennen
- Drift erkennen
- Rückkehr erkennen
- "dieses Muster kenne ich" abbilden

---

### 4.11 Modul Q - Innere Wahrnehmung / Self-State
**Aufgabe**
- interne Lage des Systems aus Feld- und Verlaufsdaten ableiten

**Zustand**
- `q(t)`

**Minimalvektor**
`q(t) = [x, |x|, dx/dt, Var_x, cluster_stability, center_distance]`

**optionale lesbare Labels**
- stable
- active
- excited
- stressed

Wichtig:
Labels sind nur lesbar, der eigentliche Zustand bleibt kontinuierlich.

---

### 4.12 Modul REG - Selbstregulierung
**Aufgabe**
- Feld- und Replay-Dynamik adaptiv modulieren

**Reglerzustand**
- `g(t)`

**Regelbare Groessen**
- Rückführungsstärke `k_reg(t)`
- Replay-Gain
- Input-Gain
- Noise-Level

**Beispiele**
- überauslenkung -> `k_reg` erhoehen
- Erstarrung -> Input-Gain leicht anheben
- Gründel-Schleife -> Replay-Gain senken
- geordnete Exploration -> Rückführung etwas lockern

---

### 4.13 Modul Y - Datenausgang
**Aufgabe**
- internen Zustand sichtbar machen

**Ausgabe**
- Spike-Aktivität
- `x(t)`
- `rho(x,t)`
- `mu_x(t)`
- `Var_x(t)`
- Spannung `S(x)`
- dominanter Cluster
- Kontext `c(t)`
- Self-State `q(t)`
- Reflexion `rfl(t)`
- Reglerzustand `g(t)`

**Wichtig**
- Datenausgang ist zunächst Beobachtung, nicht Verhalten.

## 5. Signalfluss

`u(t) -> n(t) -> x(t)`
`n(t) -> rho(x,t)`
`x(t), rho(x,t) -> C_t`
`C_t -> Mem`
`Mem, c(t) -> r_mem(t)`
`x(t), rho(x,t), C_t, u(t) -> c(t+1)`
`x(t), c(t), C_t -> rfl(t)`
`x(t), rho(x,t), rfl(t) -> q(t)`
`q(t), c(t), rfl(t) -> g(t)`
`g(t) -> k_reg(t), replay_gain, input_gain, noise_gain`
`x(t), rho(x,t), c(t), q(t), rfl(t), g(t) -> y(t)`

## 6. Umsetzungsphasen

### Phase A - Minimaler neuraler MCM-Core
**Ziel**
- Wahrnehmung
- spikende Aktivität
- kontinuierlicher Feldzustand
- einfache Rückführung
- Datenausgang

**Lieferobjekte**
- Nengo-Modell mit Input-Node, Ensemble, rekurrenter Rückführung
- Probe für Spikes, Rates, `x(t)`

**Abnahmekriterium**
- Reize verschieben `x(t)`
- ohne Reiz kehrt `x(t)` in Richtung `0` zurück

---

### Phase B - Feldbeobachtung
**Ziel**
- `rho(x,t)` rekonstruieren
- Mittelwert / Varianz / Peaks ableiten

**Abnahmekriterium**
- mehr als ein Aktivitätszentrum im Feld erkennbar
- Varianz steigt bei Instabilität sichtbar an

---

### Phase C - Clustering und Gedächtnis
**Ziel**
- wiederkehrende Feldmuster automatisch sammeln
- Replay vorbereiten

**Abnahmekriterium**
- stabile Cluster über mehrere Runs reproduzierbar
- Replay kann bekannte Muster reaktivieren

---

### Phase D - Kontextbildung und Kontextlernen
**Ziel**
- internen Kontextvektor erzeugen
- übergänge und Regelungsbedarf lernen

**Abnahmekriterium**
- gleicher Stimulus führt bei verschiedenem Kontext zu verschiedenen Feldverläufen

---

### Phase E - Reflexion und Self-State
**Ziel**
- System kann eigenen Zustand als Meta-Zustand ausdrücken

**Abnahmekriterium**
- Schleifen, Drift, Rückkehr, Instabilität werden intern kenntlich

---

### Phase F - Selbstregulierung
**Ziel**
- adaptive Regelung statt nur passiver Dynamik

**Abnahmekriterium**
- extreme Auslenkungen nehmen ab
- geordnete Zwischenzustände nehmen zu
- Replay-Schleifen werden kontrollierbar

## 7. Technologievorschlag

### Kernstack
- Python
- Nengo
- NengoSPA für spätere Symbol-/Kontextkomponenten
- NumPy
- SciPy
- scikit-learn oder hdbscan für Clustering
- pandas für Auswertung
- matplotlib für Feld-/Aktivitätsplots

### Empfohlene Repo-Bausteine
```text
src/
  core/
    perception.py
    neural_core.py
    mcm_state.py
    field_density.py
    clustering.py
    memory.py
    replay.py
    context.py
    reflection.py
    self_state.py
    regulation.py
    output.py

  experiments/
    exp_phase_a.py
    exp_phase_b.py
    exp_phase_c.py

  viz/
    plot_spikes.py
    plot_field.py
    plot_clusters.py

tests/
  test_mcm_state.py
  test_regulation.py
  test_context.py

docs/
  README.md
  UMSETZUNGSPLAN.md
```

## 8. Messgroessen und Evaluation

### Kernmetriken
- Rückkehrzeit zum Zentrum
- Maximalabweichung
- mittlere Feldvarianz
- Anzahl stabiler Cluster
- Replay-Intensität
- Schleifenlänge
- Regulationswirksamkeit
- Kontextsensitivität
- Selbstzustandskonsistenz

### Beispiel-Fragen
- Wird Wahrnehmung als innere Lage gehalten?
- Bleiben Wiederkehrmuster als Cluster erhalten?
- Kann das System ohne neün Stimulus intern weiterarbeiten?
- ändert Regulation die Dynamik messbar?
- Liefert der Datenausgang einen sinnvollen Zustandsreport?

## 9. Risiken und offene Punkte

### Technische Risiken
- zu schwache Kopplung: Feld bleibt trivial
- zu starke Kopplung: Feld kippt in Daürauslenkung
- Replay destabilisiert Grunddynamik
- Clustering wird zu verrauscht
- Kontext explodiert dimensional

### Konzeptionelle Risiken
- MCM ist hypothetisch und nicht empirisch validiert
- die psychologischen Zonen dürfen nicht mit echter Neuroanatomie verwechselt werden
- "Reflexion" bleibt hier technische Meta-Verarbeitung, kein Nachweis von Bewusstsein

## 10. Harte Entscheidungsregeln für die Umsetzung

1. Der Feldraum bleibt **kontinuierlich**.
2. Zonen bleiben **Readout**, nicht Strukturgrenzen.
3. Verhalten ist **sekundär**; primär ist innerer Zustandsreport.
4. Erst **Phase A-F** stabilisieren, dann erst komplexe Agentik.
5. Kein Funktionswildwuchs vor sauberem Kern.

## 11. Erwartetes Ergebnis

Wenn dieser Plan sauber umgesetzt ist, entsteht:

- kein klassisches Reiz-Reaktions-System
- kein reines Symbolsystem
- kein vollständiger Gehirnnachbau

sondern ein:

**spikendes inneres Zustandsmodell mit eigener Feldlage, Musterbildung, Replay, Kontext, Reflexion und Selbstregulierung**

## 12. Abschlussformel des Projekts

**Spaun-Technik liefert die neuronale Aktivität.  
MCM liefert den inneren Zustandsraum.  
Die Kombination liefert ein System, das Wahrnehmung nicht nur verarbeitet, sondern als eigene Lage hält, weiterbearbeitet, wiederaufnimmt und reguliert.**
