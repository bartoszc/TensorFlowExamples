# TensorFlow Examples

## Opis projektu

Ten projekt zawiera przykłady zastosowania biblioteki TensorFlow do pracy z sieciami konwolucyjnymi (CNN). W szczególności korzystamy z modelu MobileNetV2 do przewidywania etykiet obrazów.

## Wymagania

Projekt wymaga zainstalowania następujących bibliotek:
- numpy
- tensorflow

Można je zainstalować za pomocą polecenia:

```python
pip install numpy tensorflow
```

## Struktura katalogu

Projekt zawiera jeden główny folder:
- `CNN`: zawiera skrypt Python `predict_image.py`, który ładuje obraz, przetwarza go i korzysta z modelu MobileNetV2 do przewidywania etykiety obrazu.

## Przykład użycia

Aby przewidzieć etykietę obrazu, wykonaj następujące kroki:

1. Umieść obraz, który chcesz przetestować, w katalogu projektu (lub zmodyfikuj ścieżkę w skrypcie).
2. Uruchom skrypt `predict_image.py` z katalogu `CNN`.

Przykład komendy:

```python
python CNN/predict_image.py
```

Wyniki zostaną wyświetlone w terminalu, a skrypt pokaże trzy najbardziej prawdopodobne etykiety dla danego obrazu wraz z odpowiadającymi im prawdopodobieństwami.

## Przykład wyniku

Po uruchomieniu skryptu z obrazem `lion.jpg`, możesz otrzymać wynik podobny do poniższego:

1: Label: lion, Score: 0.93
2: Label: horse, Score: 0.05
3: Label: bird, Score: 0.02


## Jak to działa

Skrypt ładuje obraz, przekształca go do odpowiedniej formy i rozmiaru, następnie korzysta z wstępnie wytrenowanego modelu MobileNetV2 z wagami dla ImageNet, aby przewidzieć etykietę obrazu. Dekodowanie przewidywań odbywa się przez funkcję `decode_predictions`, która zwraca nazwy klas i prawdopodobieństwa.

