# Umsetzungsplan: MCM-Spiking-State-Field

## 1. Zieldefinition

Ziel ist die Umsetzung eines **spikenden inneren Zustandsmodells**, das die folgenden Funktionsketten realisiert:

- Wahrnehmung
- neuronale Verarbeitung
- kontinuierlicher MCM-Feldzustand
- Clustering
- Grübeln / Denken / Replay
- eigene Kontextbildung
- Kontextlernen
- Reflexion
- innere Wahrnehmung / Self-State
- Meta-Regulation
- Selbstregulierung
- Datenausgang als interner Zustandsreport

Das Ziel ist **kein komplettes Brain-Scale-System**, sondern eine **saubere, aufbaubare Kernarchitektur**.

## 2. Entwurfsprinzipien

1. **Kontinuierlicher Zustandsraum**
   - keine harten Mauern im Feld
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
   - das System muss Überauslenkung, Schleifenbildung und Instabilität aktiv beeinflussen können

6. **Meta-Regulation als eigene Ordnungsebene**
   - Metaregulatoren beschreiben nicht die direkte Feldlage, sondern wie das System seine Lage verarbeitet
   - sie modulieren Schwellen, Rückführung, Offenhaltung, Schutzweite und Variabilitätsbalance

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

### 3.9 Feldgrößen
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
- Kontext von außen

**Ausgang**
- `u(t)`

**Beispielkanäle**
- valence
- novelty
- relevance
- uncertainty
- social_salience

### 4.2 Modul N - Spikende neuronale Verarbeitung
**Aufgabe**
- Reize in spikende Aktivitätsmuster übersetzen
- rekurrente Aktivität aufrechterhalten

**Zustand**
- Spike-Muster
- firing rates
- Ensemble-Zustände

**Ausgang**
- `n(t)`

### 4.3 Modul X - MCM-Einzelzustand
**Aufgabe**
- aus `n(t)` einen kontinuierlichen inneren Feldwert ableiten

**Zustand**
- `x(t)`

**Update**
`dx/dt = -k_reg(t) * x + W_in * n(t) + r_mem(t) + eta(t)`

### 4.4 Modul RHO - Feldrekonstruktion
**Aufgabe**
- aus den neuronalen Aktivitäten eine kontinuierliche Dichte über dem Raum rekonstruieren

**Zustand**
- `rho(x,t)`

**Praktische Umsetzung**
- jedem Ensemble / jeder Unterpopulation wird ein bevorzugter Bereich `p_i` im Intervall `[-3, +3]` zugeordnet
- aus Aktivität `a_i(t)` wird Dichte rekonstruiert:
  `rho(x,t) = sum_i a_i(t) * K(x - p_i)`

### 4.5 Modul C - Clustering
**Aufgabe**
- wiederkehrende Muster in `(x, rho, Var_x, Verlauf)` erkennen

**Ausgang**
- Clusterobjekte `C_j`

**Clusterstruktur**
- `mu_j` = Schwerpunkt
- `Sigma_j` = Streuung
- `strength_j` = Auftretenshäufigkeit / Gewicht
- `age_j`
- `stability_j`

### 4.6 Modul M - Cluster-Gedächtnis
**Aufgabe**
- häufige oder relevante Cluster speichern
- spätere Wiederaktivierung ermöglichen

### 4.7 Modul G - Grübeln / Denken / Replay
**Aufgabe**
- interne Aktivität ohne neuen Außenreiz weiterlaufen lassen

**Ausgang**
- `r_mem(t)`

**Replay-Regel**
`r_mem(t) = sum_j gate_j(t) * K(x - mu_j)`

### 4.8 Modul K - Kontextbildung
**Aufgabe**
- aus Verlauf, Feldlage, Clustern und Reizgeschichte einen eigenen internen Kontext erzeugen

**Zustand**
- `c(t)`

### 4.9 Modul L - Kontextlernen
**Aufgabe**
- lernen, welche Konfigurationen zu welchen Feldmustern und Regulationsbedarfen führen

### 4.10 Modul F - Reflexion
**Aufgabe**
- aktuellen Zustand mit Vorzustand, Verlauf und bekannten Mustern vergleichen

**Zustand**
- `rfl(t)`

### 4.11 Modul Q - Innere Wahrnehmung / Self-State
**Aufgabe**
- interne Lage des Systems aus Feld- und Verlaufsdaten ableiten

**Zustand**
- `q(t)`

**Minimalvektor**
`q(t) = [x, |x|, dx/dt, Var_x, cluster_stability, center_distance]`

**Optionale lesbare Labels**
- stable
- active
- excited
- stressed
- diffuse

### 4.12 Modul META - Meta-Regulation
**Aufgabe**
- Regler zweiter Ordnung aus Verlauf, Self-State, Reflexion und Kontext ableiten
- bestimmen, wie stark das System Spannung verarbeitet, begrenzt, offenhält oder zurückführt

**Zustand**
- `m(t)`

### 4.13 Modul REG - Selbstregulierung
**Aufgabe**
- Feld- und Replay-Dynamik adaptiv modulieren

**Regelbare Größen**
- Rückführungsstärke `k_reg(t)`
- Replay-Gain
- Input-Gain
- Noise-Level
- Schutzschwellen
- Felddämpfung / Offenhaltung

### 4.14 Modul Y - Datenausgang
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
- Meta-Regulationszustand `m(t)`
- Reglerzustand `g(t)`

## 5. Signalfluss

`u(t) -> n(t) -> x(t)`  
`n(t) -> rho(x,t)`  
`x(t), rho(x,t) -> C_t`  
`C_t -> Mem`  
`Mem, c(t) -> r_mem(t)`  
`x(t), rho(x,t), C_t, u(t) -> c(t+1)`  
`x(t), c(t), C_t -> rfl(t)`  
`x(t), rho(x,t), rfl(t) -> q(t)`  
`q(t), c(t), rfl(t), history -> m(t)`  
`m(t), q(t), c(t), rfl(t) -> g(t)`  
`g(t) -> k_reg(t), replay_gain, input_gain, noise_gain, protection_width`  
`x(t), rho(x,t), c(t), q(t), rfl(t), m(t), g(t) -> y(t)`

## 6. Umsetzungsphasen

### Phase A - Minimaler neuraler MCM-Core
**Ziel**
- Wahrnehmung
- spikende Aktivität
- kontinuierlicher Feldzustand
- einfache Rückführung
- Datenausgang

**Abnahmekriterium**
- Reize verschieben `x(t)`
- ohne Reiz kehrt `x(t)` in Richtung `0` zurück

### Phase B - Feldbeobachtung
**Ziel**
- `rho(x,t)` rekonstruieren
- Mittelwert / Varianz / Peaks ableiten

**Abnahmekriterium**
- Feldunruhe und Aktivitätszentren sind messbar
- Varianz steigt bei Instabilität sichtbar an

### Phase C - Clustering und Gedächtnis
**Ziel**
- wiederkehrende Feldmuster automatisch sammeln
- Replay vorbereiten

**Abnahmekriterium**
- stabile Cluster werden reproduzierbar erkannt
- Replay kann bekannte Muster reaktivieren

### Phase D - Kontextbildung und Kontextlernen
**Ziel**
- internen Kontextvektor erzeugen
- Übergänge und Regelungsbedarf lernen

**Abnahmekriterium**
- gleicher Stimulus führt bei verschiedenem Kontext zu verschiedenen Feldverläufen

### Phase E - Reflexion und Self-State
**Ziel**
- System kann eigenen Zustand als Meta-Zustand ausdrücken

**Abnahmekriterium**
- Schleifen, Drift, Rückkehr und Instabilität werden intern kenntlich

### Phase F - Meta-Regulation und Selbstregulierung
**Ziel**
- Metaregulatoren zweiter Ordnung ableiten
- adaptive Regelung statt nur passiver Dynamik

**Abnahmekriterium**
- Meta-Regulatoren modulieren Reglerparameter nachvollziehbar
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

### Aktuelle Projektstruktur

```text
/mnt/data/
│── main.py
│── start.py
│── README.md
│── UMSETZUNGSPLAN.md
├── src/
│   ├── __init__.py
│   ├── debug_reader.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── clustering.py
│   │   ├── context.py
│   │   ├── field_density.py
│   │   ├── mcm_state.py
│   │   ├── memory.py
│   │   ├── meta_regulation.py
│   │   ├── neural_core.py
│   │   ├── output.py
│   │   ├── perception.py
│   │   ├── reflection.py
│   │   ├── regulation.py
│   │   ├── replay.py
│   │   └── self_state.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── exp_phase_a.py
│   │   ├── exp_phase_b.py
│   │   ├── exp_phase_c.py
│   │   ├── exp_phase_d.py
│   │   ├── exp_phase_e.py
│   │   └── exp_phase_f.py
│   └── viz/
│       ├── __init__.py
│       ├── plot_field.py
│       ├── plot_clusters.py
│       └── plot_spikes.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_clustering.py
    ├── test_context.py
    ├── test_experiments_pipeline.py
    ├── test_field_density.py
    ├── test_mcm_state.py
    ├── test_memory.py
    ├── test_meta_regulation.py
    ├── test_neural_core.py
    ├── test_output.py
    ├── test_perception.py
    ├── test_reflection.py
    ├── test_regulation.py
    ├── test_replay.py
    └── test_self_state.py
```

## 8. Messgrößen und Evaluation

### Kernmetriken
- Rückkehrzeit zum Zentrum
- Maximalabweichung
- mittlere Feldvarianz
- Anzahl stabiler Cluster
- Replay-Intensität
- Schleifenlänge
- Regulationswirksamkeit
- Metaregulationskonsistenz
- Kontextsitivität
- Selbstzustandskonsistenz

### Beispiel-Fragen
- Wird Wahrnehmung als innere Lage gehalten?
- Bleiben Wiederkehrmuster als Cluster erhalten?
- Kann das System ohne neuen Stimulus intern weiterarbeiten?
- Ändert Regulation die Dynamik messbar?
- Liefert der Datenausgang einen sinnvollen Zustandsreport?

## 9. Risiken und offene Punkte

### Technische Risiken
- zu schwache Kopplung: Feld bleibt trivial
- zu starke Kopplung: Feld kippt in Dauerauslenkung
- Replay destabilisiert Grunddynamik
- Clustering wird zu verrauscht
- Kontext explodiert dimensional

### Konzeptionelle Risiken
- MCM ist hypothetisch und nicht empirisch validiert
- die psychologischen Zonen dürfen nicht mit echter Neuroanatomie verwechselt werden
- Reflexion bleibt hier technische Meta-Verarbeitung, kein Nachweis von Bewusstsein

## 10. Harte Entscheidungsregeln für die Umsetzung

1. Der Feldraum bleibt **kontinuierlich**.
2. Zonen bleiben **Readout**, nicht Strukturgrenzen.
3. Metaregulatoren bleiben **zweite Ordnung** und werden nicht mit Primärzuständen vermischt.
4. Verhalten ist **sekundär**; primär ist innerer Zustandsreport.
5. Erst **Phase A-F** stabilisieren, dann erst komplexe Agentik.
6. Kein Funktionswildwuchs vor sauberem Kern.

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
