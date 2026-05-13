"""
Métricas de clasificación.

Este módulo implementa métricas fundamentales para evaluar
modelos de clasificación.

Las funciones aquí definidas operan sobre ArrayLike
y no dependen de librerías externas de machine learning.

Incluye:
- Accuracy
- Precision
- Recall

:authors: Tomás Macrade
:date: 28/02/2026
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from whiteboxml.utils import (
    _compute_metric_components,
    _validacion_average,
    _validacion_inputs,
)


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Cálculo del accuracy.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :return: accuracy
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)
    return float(np.mean(vector_true == vector_pred))


def precision(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "binary",
    pos_label: Any = 1,
) -> float | np.ndarray:
    """
    Cálculo de la precision.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param average: define el tipo de average en
    clasificación multiclase ("binary","micro", "macro", "weighted", None)
    :param pos_label: valor a considerar como positivo
    en el caso de targets binarios. Ignorado si average != "binary".
    :return: score de precision o array con la precision por clase
    en caso de average = None
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    average = _validacion_average(average)
    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    classes = np.unique(vector_true)

    if average == "binary":
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fp = np.sum((vector_pred == pos_label) & (vector_true != pos_label))
        return float(tp / (tp + fp)) if tp + fp > 0 else 0.0

    if average == "micro":
        tp = np.sum(vector_true == vector_pred)
        total_muestras = vector_true.size
        return float(tp / total_muestras) if total_muestras > 0 else 0.0

    if average == "macro":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FP"]
        )
        tp, fp = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision = np.nan_to_num(tp / (tp + fp))
        return float(np.mean(per_class_precision))

    if average == "weighted":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FP"]
        )
        tp, fp = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_precision = np.nan_to_num(tp / (tp + fp))

        supports = np.array([np.sum(vector_true == c) for c in classes])
        total_support = np.sum(supports)
        weights = supports / total_support

        return float(np.sum(per_class_precision * weights))

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FP"]
    )
    tp, fp = components[:, 0], components[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_precision = np.nan_to_num(tp / (tp + fp))
    return per_class_precision


def recall(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "micro",
    pos_label: Any = 1,
) -> float | np.ndarray:
    """
    Cálculo del recall.

    :param y_true: targets reales
    :param y_pred: targets predichos
    :param average: define el tipo de average en
    clasificación multiclase ("binary","micro", "macro", "weighted", None)
    :param pos_label: valor a considerar como positivo en el caso de targets binarios
    :return: score de recall o array con la recall por clase en caso de average = None
    :authors: Tomás Macrade
    :date: 28/02/2026
    """

    average = _validacion_average(average)
    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    classes = np.unique(vector_true)

    if average == "binary":
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))
        fn = np.sum((vector_pred != pos_label) & (vector_true == pos_label))
        return float(tp / (tp + fn)) if tp + fn > 0 else 0.0

    if average == "micro":
        tp = np.sum(vector_true == vector_pred)
        total_muestras = vector_true.size
        return float(tp / total_muestras) if total_muestras > 0 else 0.0

    if average == "macro":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FN"]
        )
        tp, fn = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall = np.nan_to_num(tp / (tp + fn))
        return float(np.mean(per_class_recall))

    if average == "weighted":
        components = _compute_metric_components(
            vector_true, vector_pred, classes, ["TP", "FN"]
        )
        tp, fn = components[:, 0], components[:, 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            per_class_recall = np.nan_to_num(tp / (tp + fn))

        supports = np.array([np.sum(vector_true == c) for c in classes])
        total_support = np.sum(supports)
        weights = supports / total_support

        return float(np.sum(per_class_recall * weights))

    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FN"]
    )
    tp, fn = components[:, 0], components[:, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.nan_to_num(tp / (tp + fn))
    return per_class_recall


def f1_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    average: str | None = "binary",
    pos_label: Any = 1,
) -> float | np.ndarray:
    """
    Cálculo del F1 Score.
    El F1 Score es la media entre precision y recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    :param y_true: valores reales (ground truth)
    :param y_pred: valores predichos por el modelo
    :param average: tipo de promedio en multiclase
    :param pos_label: clase positiva (solo para binary)
    :return: valor de F1 o vector por clase
    :authors: Cecilia Gómez. Grupo 6
    :date: 24/04/2026
    """

    # Validamos el parámetro average (lo normaliza y revisa que sea válido)
    average = _validacion_average(average)

    # Convertimos los inputs a numpy arrays y validamos dimensiones
    vector_true, vector_pred = _validacion_inputs(y_true, y_pred)

    # Obtenemos las clases únicas presentes en los datos
    classes = np.unique(vector_true)

    # CASO BINARIO

    if average == "binary":

        # True Positives: predije positivo y era positivo
        tp = np.sum((vector_pred == pos_label) & (vector_true == pos_label))

        # False Positives: predije positivo pero era negativo
        fp = np.sum((vector_pred == pos_label) & (vector_true != pos_label))

        # False Negatives: predije negativo pero era positivo
        fn = np.sum((vector_pred != pos_label) & (vector_true == pos_label))

        # Precision = TP / (TP + FP)
        # Evitamos división por 0
        precision_score = float(tp / (tp + fp)) if tp + fp > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall_score = float(tp / (tp + fn)) if tp + fn > 0 else 0.0

        # F1 combina ambas
        denominator = precision_score + recall_score

        return (
            2 * precision_score * recall_score / denominator if denominator > 0 else 0.0
        )

    # MICRO (global)

    if average == "micro":
        # En clasificación multiclase:
        # micro-F1 = accuracy
        return accuracy(vector_true, vector_pred)

    # MULTICLASE (macro, weighted, None)

    # Obtenemos TP, FP y FN
    components = _compute_metric_components(
        vector_true, vector_pred, classes, ["TP", "FP", "FN"]
    )

    # Separar cada componente
    tp, fp, fn = components[:, 0], components[:, 1], components[:, 2]

    # Calculamos precision y recall por clase
    with np.errstate(divide="ignore", invalid="ignore"):

        # Precision por clase
        per_class_precision = np.nan_to_num(tp / (tp + fp))

        # Recall por clase
        per_class_recall = np.nan_to_num(tp / (tp + fn))

        # F1 por clase
        per_class_f1 = np.nan_to_num(
            2
            * per_class_precision
            * per_class_recall
            / (per_class_precision + per_class_recall)
        )

    # MACRO

    if average == "macro":
        # Promedio simple entre clases
        return float(np.mean(per_class_f1))

    # WEIGHTED

    if average == "weighted":

        # Cantidad de ejemplos por clase (soporte)
        supports = np.array([np.sum(vector_true == c) for c in classes])

        # Total de ejemplos
        total_support = np.sum(supports)

        # Pesos relativos de cada clase
        weights = supports / total_support

        # Promedio ponderado
        return float(np.sum(per_class_f1 * weights))

    # NONE → devuelve por clase

    return per_class_f1
