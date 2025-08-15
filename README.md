# Müsli Test | Clustering aus Aggregierten Mermalen

Aggregierter Eingangsvektor von [Funkez123](https://github.com/Funkez123/grain-segmentation)

Das Repo enthält mehrere Python scripts um einen Datensatz zuclusterm und weitere Daten den Clustern zuzuordnen.
Implementiert ist eine [SOM](https://de.wikipedia.org/wiki/Selbstorganisierende_Karte), welche daten den jeweiligen Kohonen zuordnen kann und dann an weitere Clusteringalgorithmen weitergegeben werden kann.
Desweiteren sind [K-means](https://de.wikipedia.org/wiki/K-Means-Algorithmus), [GMM](https://en.wikipedia.org/wiki/Mixture_model), [spektrales Clustering](https://de.wikipedia.org/wiki/Spektrales_Clustering), [ART](https://en.wikipedia.org/wiki/Adaptive_resonance_theory) und [NG](https://de.wikipedia.org/wiki/Neural_Gas)
als Clusteringalgorithmen implementiert.

In testdata.py soll aggregated_features_per_image.csv gesplittet werden in Trainingsdaten(scaled_data) und Testdaten(test_data).
in Som.py wird dann mit den Trainingsdaten eine Kohonenkarte trainiert und die Gewichte als neu Trainingsdaten gespeichert(som_scaled_data), während von den Trainingsdaten die BMU ermittelt wird und von diesem Kohon die Gewichte gespeichert werden(som_test_data).
In den anderen Dateien wird nur der jeweilige Algorthimus angewendent einmal mit den "normalen" Trainingsdaten und einmal mit den SOM Daten. Die entstehenden Clusterzuordnungen werden in clusterlabels.csv mit einem entsprechendem label gespeichert.

Manchmal habe ich noch ein paar plot funktionen rein gemacht, die sind aber nicht so wichtig.
Number of Clusters = 5, wird schon passen.

Bei Fragen einfach direkt schreiben.
