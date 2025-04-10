import math
from typing import Dict

import pandas as pd


def calculate_scores(
    found_words: Dict[str, int], weights_file: str
) -> Dict[str, int]:
    """
    Рассчитывает баллы для найденных слов на основе их весов из Excel файла,
    группируя их по категориям.

    :param found_words: Словарь с найденными словами и количеством их появлений
    :param weights_file: Путь к Excel файлу с весами
    :return: Словарь с подсчитанными баллами для каждой категории
    """
    df = pd.read_excel(weights_file)
    category_scores = {}
    for word, count in found_words.items():
        word_data = df[df["word"] == word]
        if not word_data.empty:
            category = word_data["category"].iloc[0]
            weight = word_data["weight"].iloc[0]
            score = weight * weight * count
            if category in category_scores:
                category_scores[category] += score
            else:
                category_scores[category] = score

    return dict(
        sorted(category_scores.items(), key=lambda item: item[1], reverse=True)
    )


def apply_softmax(scores: Dict[str, int]) -> Dict[str, float]:
    """
    Применяет функцию softmax к скорам категорий для получения вероятностей.

    :param scores: Словарь с категориями и их скорами
    :return: Словарь с категориями и их вероятностями, отсортированный по убыванию вероятностей
    """
    if not scores:
        return {}

    exp_scores = [math.exp(score) for score in scores.values()]
    sum_exp = sum(exp_scores)
    probabilities = {
        category: round(math.exp(score) / sum_exp, 4)
        for category, score in scores.items()
    }

    return dict(
        sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    )


if __name__ == "__main__":
    found_words = {"карта": 1, "мир": 1, "дебетовый карта": 2}

    scores = calculate_scores(
        found_words, "../output/weights_with_category.xlsx"
    )
    print(scores)

    probabilities = apply_softmax(scores)
    print(probabilities)
