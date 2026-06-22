Projekt magisterski: analiza danych sportowych na podstawie StatsBomb Open Data z Mundialu 2018.

## Struktura projektu

```text
MGR_MK/
├─ notebooks/          # notebooki eksploracyjne
├─ src/mgr_mk/         # funkcje używane w analizie
├─ tests/              # proste testy kontrolne
├─ outputs/            # wygenerowane wykresy i tabele
└─ open-data-master/   # dane StatsBomb Open Data
```

## Najważniejsze moduły

- `src/mgr_mk/data_loader.py` - ładowanie meczów, eventów i składów.
- `src/mgr_mk/features.py` - cechy zawodników per 90 minut.
- `src/mgr_mk/datasets.py` - dataset zawodnik-mecz dla całego turnieju.
- `src/mgr_mk/momentum.py` - autorski wskaźnik match momentum.
- `src/mgr_mk/plots.py` - wykresy momentum i skumulowanego xG.

## Momentum

Momentum jest autorskim wskaźnikiem opartym na wagach zdarzeń StatsBomb:

- gol ze strzału: `15.0`,
- samobój dla drużyny: `5.0`,
- zwykły strzał: `1.0 + 6.0 * xG`,
- podanie w final third: `0.35`,
- prowadzenie piłki w final third: `0.25`,
- udany drybling: `0.25`,
- odzysk piłki: `0.15`.

Wykres słupkowy pokazuje różnicę momentum w oknach 5-minutowych, a wykres liniowy pokazuje skumulowane momentum obu drużyn.

## Uruchomienie

```bash
pip install -r requirements.txt
```

Głównym miejscem pracy jest notebook:

```text
notebooks/00_glowny_notebook.ipynb
```

Ten notebook sam dodaje `src/` do ścieżki Pythona, więc można go uruchamiać bez instalowania projektu jako pakietu. Pozostałe notebooki w katalogu `notebooks/` są starszą/eksploracyjną wersją analizy.

Dataset dla wszystkich meczów i zawodników generuje notebook:

```text
notebooks/01_dataset_wszyscy_zawodnicy.ipynb
```

Wynikowy plik CSV jest zapisywany do `outputs/tables/player_match_dataset.csv`.
