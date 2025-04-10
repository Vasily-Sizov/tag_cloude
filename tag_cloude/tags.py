import logging
import re
import time
from typing import Dict, List, Tuple
from tqdm import tqdm
import ahocorasick
import pandas as pd
from pymorphy2 import MorphAnalyzer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.getLogger("pymorphy2").setLevel(logging.WARNING)


class KeywordLoader:
    def __init__(self, keywords_file: str, morph: MorphAnalyzer):
        """
        Инициализирует загрузчик ключевых слов.

        :param keywords_file: Путь к файлу с ключевыми словами.
        :param morph: Экземпляр MorphAnalyzer для лемматизации.
        """
        self.keywords_file = keywords_file
        self.morph = morph

    def get_words_from_txt(self) -> List[str]:
        """
        Загружает ключевые слова из текстового файла.

        :return: Список лемматизированных ключевых слов
        :raises FileNotFoundError: Если файл не найден
        """
        logger.info(f"Загрузка ключевых слов из файла: {self.keywords_file}")
        try:
            with open(self.keywords_file, "r", encoding="utf-8") as f:
                keywords = [line.strip().lower() for line in f if line.strip()]
                if not keywords:
                    logger.warning(f"Файл {self.keywords_file} пуст")
                    return []

                lemmatized_keywords = [
                    self.morph.parse(word)[0].normal_form
                    for word in keywords
                    if word
                ]
                logger.info(
                    f"Загружено и лемматизировано {len(lemmatized_keywords)} ключевых слов."
                )
                return lemmatized_keywords
        except FileNotFoundError:
            logger.error(f"Файл {self.keywords_file} не найден.")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке ключевых слов: {e}")
            raise

    def get_words_from_excel(self, column: str) -> List[str]:
        """
        Загружает ключевые слова из Excel файла.

        :param column: Название колонки с ключевыми словами
        :return: Список ключевых слов
        :raises FileNotFoundError: Если файл не найден
        :raises ValueError: Если колонка не найдена
        """
        logger.info(f"Загрузка ключевых слов из файла: {self.keywords_file}")
        try:
            df = pd.read_excel(self.keywords_file)

            if column not in df.columns:
                error_msg = f"Колонка '{column}' не найдена в файле"
                logger.error(error_msg)
                raise ValueError(error_msg)
            words = df[column].dropna().astype(str).tolist()

            if not words:
                logger.warning(f"В колонке '{column}' нет данных")
                return []
            return words

        except FileNotFoundError:
            logger.error(f"Файл {self.keywords_file} не найден.")
            raise
        except Exception as e:
            logger.error(f"Ошибка при загрузке ключевых слов: {e}")
            raise

    def fix_keywords(self, keywords: List[str]) -> List[str]:
        """
        Нормализует и лемматизирует список ключевых слов.

        :param keywords: Список ключевых слов для обработки
        :return: Список лемматизированных ключевых слов
        :raises ValueError: Если список ключевых слов пуст
        """
        if not keywords:
            logger.warning("Получен пустой список ключевых слов")
            return []
        cleaned_keywords = [
            word.strip().lower() for word in keywords if word and word.strip()
        ]
        if not cleaned_keywords:
            logger.warning("После очистки список ключевых слов пуст")
            return []
        lemmatized_keywords = [
            self.morph.parse(word)[0].normal_form for word in cleaned_keywords
        ]
        logger.info(f"Обработано {len(lemmatized_keywords)} ключевых слов")
        return lemmatized_keywords


class DialogProcessor:
    def __init__(self, keywords: List[str], morph: MorphAnalyzer):
        """
        Инициализирует процессор диалогов.

        :param keywords: Список ключевых слов.
        :param morph: Экземпляр MorphAnalyzer для лемматизации.
        """
        self.automaton = self.build_automaton(keywords)
        self.morph = morph

    def build_automaton(self, keywords: List[str]) -> ahocorasick.Automaton:
        """
        Создает автомат для поиска ключевых слов.

        :param keywords: Список ключевых слов.
        :return: Автомат для поиска ключевых слов.
        """
        logger.info("Создание автомата для поиска ключевых слов.")
        automaton = ahocorasick.Automaton()
        for word in keywords:
            automaton.add_word(word, word)
        automaton.make_automaton()
        logger.info("Автомат успешно создан.")
        return automaton

    def process_single_dialog(self, dialog: str) -> Dict[str, Dict[str, int]]:
        """
        Обрабатывает один диалог и возвращает результаты подсчета.

        :param dialog: Строка, представляющая диалог.
        :return: Словарь с подсчетами для клиента, оператора и пересечением.
        """
        logger.info("Обработка одного диалога.")
        client_texts, operator_texts = self.extract_texts(dialog)
        client_counts, operator_counts = self.count_words(
            client_texts, operator_texts
        )
        intersection_counts = self.get_intersection(
            client_counts, operator_counts
        )

        return {
            "client_counts": client_counts,
            "operator_counts": operator_counts,
            "intersection_counts": intersection_counts,
        }

    def process_dialogs(
        self, dialogs: List[str]
    ) -> List[Dict[str, Dict[str, int]]]:
        """
        Обрабатывает список диалогов и возвращает результаты подсчета.

        :param dialogs: Список строк, представляющих диалоги.
        :return: Список словарей с подсчетами для каждого диалога.
        """
        results = []
        logger.info("Начало подсчета вхождений в диалогах.")
        total_time = 0

        for dialog in tqdm(dialogs):
            start_time = time.time()
            result = self.process_single_dialog(dialog)
            results.append(result)
            end_time = time.time()
            total_time += end_time - start_time

        average_time = total_time / len(dialogs) if dialogs else 0
        logger.info(
            f"Среднее время обработки диалога: {average_time:.6f} секунд"
        )

        return results

    def extract_texts(self, dialog: str) -> Tuple[List[str], List[str]]:
        """
        Извлекает тексты клиента и оператора из диалога.

        :param dialog: Строка, представляющая диалог.
        :return: Кортеж из двух списков: текстов клиента и оператора.
        """
        parts = dialog.split("; ")
        client_texts = []
        operator_texts = []

        for part in parts:
            if "'client':" in part:
                client_text = part.replace("'client': ", "").strip().lower()
                client_text = re.sub(r"[^а-яА-Яa-zA-Z\s]", "", client_text)
                client_texts.append(client_text)
            elif "'operator':" in part:
                operator_text = (
                    part.replace("'operator': ", "").strip().lower()
                )
                operator_text = re.sub(r"[^а-яА-Яa-zA-Z\s]", "", operator_text)
                operator_texts.append(operator_text)

        return client_texts, operator_texts

    def count_words(
        self, client_texts: List[str], operator_texts: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Подсчитывает вхождения слов в текстах клиента и оператора.

        :param client_texts: Список текстов клиента.
        :param operator_texts: Список текстов оператора.
        :return: Кортеж из двух словарей: для клиента и оператора, отсортированных по значениям.
        """
        client_counts = {}
        operator_counts = {}

        for client_text in client_texts:
            lemmatized_words = [
                self.morph.parse(word)[0].normal_form
                for word in re.findall(r"\b\w+\b", client_text)
            ]
            lemmatized_text = " ".join(lemmatized_words)
            for _, word in self.automaton.iter(lemmatized_text):
                if word in client_counts:
                    client_counts[word] += 1
                else:
                    client_counts[word] = 1

        for operator_text in operator_texts:
            lemmatized_words = [
                self.morph.parse(word)[0].normal_form
                for word in re.findall(r"\b\w+\b", operator_text)
            ]
            lemmatized_text = " ".join(lemmatized_words)
            for _, word in self.automaton.iter(lemmatized_text):
                if word in operator_counts:
                    operator_counts[word] += 1
                else:
                    operator_counts[word] = 1

        sorted_client_counts = dict(
            sorted(
                client_counts.items(), key=lambda item: item[1], reverse=True
            )
        )
        sorted_operator_counts = dict(
            sorted(
                operator_counts.items(), key=lambda item: item[1], reverse=True
            )
        )

        return sorted_client_counts, sorted_operator_counts

    def get_intersection(
        self, client_counts: Dict[str, int], operator_counts: Dict[str, int]
    ) -> Dict[str, int]:
        """
        Получает пересечение словарей с подсчетами.

        :param client_counts: Счетчик вхождений слов клиента.
        :param operator_counts: Счетчик вхождений слов оператора.
        :return: Словарь с пересечением слов и их минимальными вхождениями,
            отсортированный по количеству.
        """
        intersection_counts = {}
        for word in client_counts.keys():
            if word in operator_counts:
                intersection_counts[word] = min(
                    client_counts[word], operator_counts[word]
                )

        return dict(
            sorted(
                intersection_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

    def save_results_to_excel(
        self, results: List[Dict[str, Dict[str, int]]], file_path: str
    ) -> None:
        """
        Сохраняет результаты в файл Excel.

        :param results: Список результатов подсчета.
        :param file_path: Путь к файлу для сохранения результатов.
        """
        logger.info(f"Сохранение результатов в файл: {file_path}")

        data = []
        for i, res in enumerate(results):
            data.append(
                {
                    "dialog_id": i + 1,
                    "client_counts": res["client_counts"],
                    "operator_counts": res["operator_counts"],
                    "intersection_counts": res["intersection_counts"],
                }
            )

        with pd.ExcelWriter(file_path) as writer:
            df = pd.DataFrame(data)
            df.to_excel(writer, index=False)
        logger.info("Результаты успешно сохранены.")


class ExactWordDialogProcessor(DialogProcessor):
    def count_words(
        self, client_texts: List[str], operator_texts: List[str]
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        """
        Подсчитывает точные вхождения слов и фраз в текстах клиента и оператора,
        используя автомат Ахо-Корасик с проверкой границ слов.

        :param client_texts: Список текстов клиента.
        :param operator_texts: Список текстов оператора.
        :return: Кортеж из двух словарей: для клиента и оператора, отсортированных по значениям.
        """
        client_counts = {}
        operator_counts = {}

        # Обработка текстов клиента
        for client_text in client_texts:
            # Лемматизируем текст
            words = re.findall(r"\b\w+\b", client_text.lower())
            lemmatized_words = [
                self.morph.parse(word)[0].normal_form for word in words
            ]
            lemmatized_text = " ".join(lemmatized_words)

            # Используем автомат для поиска
            for end_pos, found_word in self.automaton.iter(lemmatized_text):
                start_pos = end_pos - len(found_word) + 1
                # Проверяем границы слова
                is_word_boundary = (
                    start_pos == 0 or lemmatized_text[start_pos - 1] == " "
                ) and (
                    end_pos + 1 == len(lemmatized_text)
                    or lemmatized_text[end_pos + 1] == " "
                )
                if is_word_boundary:
                    client_counts[found_word] = (
                        client_counts.get(found_word, 0) + 1
                    )

        # Обработка текстов оператора
        for operator_text in operator_texts:
            words = re.findall(r"\b\w+\b", operator_text.lower())
            lemmatized_words = [
                self.morph.parse(word)[0].normal_form for word in words
            ]
            lemmatized_text = " ".join(lemmatized_words)

            for end_pos, found_word in self.automaton.iter(lemmatized_text):
                start_pos = end_pos - len(found_word) + 1
                is_word_boundary = (
                    start_pos == 0 or lemmatized_text[start_pos - 1] == " "
                ) and (
                    end_pos + 1 == len(lemmatized_text)
                    or lemmatized_text[end_pos + 1] == " "
                )
                if is_word_boundary:
                    operator_counts[found_word] = (
                        operator_counts.get(found_word, 0) + 1
                    )

        # Сортируем результаты
        sorted_client_counts = dict(
            sorted(
                client_counts.items(), key=lambda item: item[1], reverse=True
            )
        )
        sorted_operator_counts = dict(
            sorted(
                operator_counts.items(), key=lambda item: item[1], reverse=True
            )
        )

        return sorted_client_counts, sorted_operator_counts


if __name__ == "__main__":
    dialogs = [
        "'client': мирный карта карта кредит дебетовая карта; 'operator': у нас есть карты премиум и карты стандарт",
        "'client': доставка есть?; 'operator': да, доставка бесплатная. доставкой занимается компания 'СДЭК'. доставками. диставка; 'client': хочу оплатить картой карта",
    ]

    morph_analyzer = MorphAnalyzer()
    keywords_loader = KeywordLoader("../output/weights.xlsx", morph_analyzer)
    keywords = keywords_loader.get_words_from_excel(column="word")
    keywords = keywords_loader.fix_keywords(keywords)

    processor = DialogProcessor(keywords, morph_analyzer)
    results = processor.process_dialogs(dialogs)
    processor.save_results_to_excel(results, "../output/results.xlsx")

    processor2 = ExactWordDialogProcessor(keywords, morph_analyzer)
    results2 = processor2.process_dialogs(dialogs)
    processor2.save_results_to_excel(
        results2, "../output/results_exact_words.xlsx"
    )
