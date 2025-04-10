import logging
import re
from typing import Dict, List, Tuple, Union

import pandas as pd
from pymorphy2 import MorphAnalyzer

# Настройка логирования
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FileReadError(Exception):
    """Исключение при ошибке чтения файла."""

    pass


class FileWriteError(Exception):
    """Исключение при ошибке записи файла."""

    pass


class KeywordsProcessor:
    def __init__(self, morph: MorphAnalyzer):
        """
        Инициализирует процессор ключевых слов.

        :param morph: Морфологический анализатор
        """
        self.morph = morph
        logger.info("Инициализирован процессор ключевых слов")

    def read_keywords(self, input_file: str) -> List[str]:
        """
        Читает ключевые слова из файла.

        :param input_file: Путь к файлу с ключевыми словами
        :return: Список ключевых слов
        :raises FileReadError: Если возникла ошибка при чтении файла
        """
        logger.info(f"Чтение ключевых слов из файла: {input_file}")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                keywords = [line.strip().lower() for line in f if line.strip()]
            logger.info(f"Прочитано {len(keywords)} ключевых слов")
            return keywords
        except FileNotFoundError:
            error_msg = f"Файл не найден: {input_file}"
            logger.error(error_msg)
            raise FileReadError(error_msg)
        except Exception as e:
            error_msg = f"Ошибка при чтении файла {input_file}: {str(e)}"
            logger.error(error_msg)
            raise FileReadError(error_msg)

    def process_keyword(self, keyword: str) -> Tuple[str, int]:
        """
        Обрабатывает одно ключевое слово.

        :param keyword: Ключевое слово или фраза
        :return: Кортеж из лемматизированной фразы и её веса
        """
        words = re.findall(r"\b\w+\b", keyword)
        lemmatized_words = [
            self.morph.parse(word)[0].normal_form for word in words
        ]
        result = " ".join(lemmatized_words), len(words)
        logger.debug(f"Обработано ключевое слово: {keyword} -> {result}")
        return result

    def create_keywords_data(
        self, keywords: List[str]
    ) -> List[Dict[str, Union[str, int]]]:
        """
        Создает список обработанных ключевых слов.

        :param keywords: Список ключевых слов
        :return: Список словарей с обработанными данными
        """
        logger.info("Начало обработки ключевых слов")
        processed_words = []
        seen_words = set()

        for keyword in keywords:
            lemmatized_phrase, weight = self.process_keyword(keyword)
            if lemmatized_phrase not in seen_words:
                seen_words.add(lemmatized_phrase)
                processed_words.append(
                    {
                        "category": "",
                        "weight": weight,
                        "word": lemmatized_phrase,
                    }
                )

        logger.info(f"Обработано {len(processed_words)} уникальных слов")
        return processed_words

    def save_to_excel(
        self, data: List[Dict[str, Union[str, int]]], output_file: str
    ) -> None:
        """
        Сохраняет данные в Excel файл.

        :param data: Список словарей с данными
        :param output_file: Путь к выходному Excel файлу
        :raises FileWriteError: Если возникла ошибка при записи файла
        """
        logger.info(f"Сохранение данных в файл: {output_file}")
        try:
            df = pd.DataFrame(data)
            df = df.sort_values("weight", ascending=False)
            with pd.ExcelWriter(output_file) as writer:
                df.to_excel(writer, index=False)
            logger.info("Данные успешно сохранены")
        except Exception as e:
            error_msg = f"Ошибка при сохранении файла {output_file}: {str(e)}"
            logger.error(error_msg)
            raise FileWriteError(error_msg)

    def process_keywords_file(self, input_file: str, output_file: str) -> None:
        """
        Обрабатывает файл с ключевыми словами и создает Excel файл с весами.

        :param input_file: Путь к файлу с ключевыми словами
        :param output_file: Путь к выходному Excel файлу
        """
        keywords = self.read_keywords(input_file)
        processed_data = self.create_keywords_data(keywords)
        self.save_to_excel(processed_data, output_file)


if __name__ == "__main__":
    morph_analyzer = MorphAnalyzer()
    processor = KeywordsProcessor(morph_analyzer)
    processor.process_keywords_file(
        "../input/keywords.txt", "../output/weights.xlsx"
    )
