# --------------------------------------------------
# Einfacher Plan vom Projekt
# --------------------------------------------------
Reize aufnehmen
Das System bekommt Eingaben (z. B. valence, novelty, relevance).

In neuronale Aktivität umwandeln
Die Eingaben werden als spikende/neuronale Aktivität verarbeitet.

Inneren Zustand bilden (x + Feld rho)
Daraus entsteht ein kontinuierlicher innerer Zustandswert im Bereich −3,+3 plus Felddichte (wie „innere Lagekarte“).

Muster erkennen (Clustering) + merken (Memory)
Wiederkehrende Zustände werden als Muster gespeichert.

Replay / internes Weiterdenken
Das System kann bekannte Muster intern wieder aktivieren (ohne neuen Außenreiz).

Kontext + Selbstbeobachtung + Reflexion
Es bewertet seine eigene Lage: stabil/instabil, Drift, Schleifen usw.

Meta-Regulation + Selbstregulierung
Ein „Regler über dem Regler“ passt Rückführung/Replay an, damit das System nicht kippt oder festhängt.

Datenausgabe
Am Ende gibt es einen internen Zustandsreport aus (nicht primär Verhalten nach außen).

# --------------------------------------------------
# Aktuelle Gesamtstruktur
# --------------------------------------------------

/mnt/data/
│── main.py
│── start.py
│── Wichtig - Strukturaufbau.txt
│── README.md
├── src/
│   ├── __init__.py
│   │── debug_reader.py
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

# --------------------------------------------------
# Zusammenfassung
# --------------------------------------------------

1) Was das Projekt sein will (Zielbild)
Das Repo beschreibt eine Architektur, die spikende neuronale Verarbeitung (Spaun/Nengo-Idee) mit einem kontinuierlichen MCM-Zustandsraum kombiniert — inklusive Kontext, Reflexion, Meta-Regulation und Selbstregulierung als Gesamtziel. 

Die geplanten Umsetzungsphasen sind A–F:
A (Core), B (Feld), C (Clustering/Memory/Replay), D (Kontext), E (Reflexion/Self-State), F (Meta-Regulation/Selbstregulierung). 

2) Was ist aktuell fertig / vorhanden
✅ Kernpipeline A–C ist implementiert
Wahrnehmung/Neural-Core/MCM-State/Felddichte/Clustering/Memory/Replay sind als Module vorhanden. 

Experimente für Phase A, B, C sind vorhanden. 

✅ D–F Grundbausteine sind jetzt als Module vorhanden
context.py, reflection.py, self_state.py, meta_regulation.py, regulation.py sind im Core enthalten. 

README/Architekturtext enthält Meta-Regulation inzwischen explizit. 

✅ Tests laufen
Test-Suite läuft grün: 19 passed.

Plan-seitig vorgesehene Dateien test_context.py und test_regulation.py sind vorhanden. 

3) Was noch fehlt (für „wirklich Phase D–F abgeschlossen“)
Auch wenn die Module da sind, fehlen typischerweise noch diese Punkte für „fertig“ im Sinne der Abnahmekriterien:

Stärkere Integration in den Laufzeitfluss
D/E/F-Module sind vorhanden, aber müssen als zusammenhängender Regelkreis durchgängig im Experiment-Loop wirken (mit klarer Wirkung auf Dynamik). 

Abnahmekriterien nachweisbar machen

Kontextsensitivität (gleicher Stimulus, anderer Kontext → anderer Verlauf)

Schleifen-/Drift-/Rückkehr-Signale robust

Meta-Regulatoren steuern Reglerparameter nachvollziehbar und verbessern Stabilität
Das steht als Ziel explizit im Plan, braucht aber i. d. R. zusätzliche Auswerteläufe/Reports. 

Evaluation-Metriken systematisch ausrollen
Für D–F sollten Kernmetriken (Replay-Intensität, Schleifenlänge, Regulationswirksamkeit, Metaregulationskonsistenz etc.) als feste Auswertung implementiert werden. 

4) Realistische Einordnung (Stand jetzt)
Technisch: solides Prototyp-Fundament, gute Modulabdeckung, tests grün.

Reifegrad: „funktionierender Forschungsprototyp“, aber noch nicht „voll validierte D–F-Regelarchitektur“ nach Plan-Abnahme.

5) Nächster sinnvoller Schritt (konkret)
Wenn du willst, mache ich als Nächstes direkt eine Mini-Roadmap mit 3 Commits:

D/E/F-Module in exp_phase_c verbindlich verdrahten,

Abnahmekriterien als Assertions/Metriken implementieren,

Evaluationsreport (JSON/CSV + kurze Interpretation) hinzufügen.

Checks, die ich dafür ausgeführt habe

✅ pytest -q (19 passed)

✅ nl -ba README.md | sed -n '1,220p'

✅ nl -ba docs/UMSETZUNGSPLAN.md | sed -n '414,540p'