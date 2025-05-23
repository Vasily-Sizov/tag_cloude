## Tag_cloude

Проект предназначен для выделения тегов из текстов и формирования скоров для принятия решений.

## Запуск проекта (mac)

Необходимо последоватено выполнить следующие команды:
- `pyenv local 3.9`
- `pyenv exec python3 -m venv venv`
- `source venv/bin/activate`
- `python3 -m pip install --upgrade pip`
- `pip install -r requirements.txt`

___

## Шаг 0. Промптинг с LLM.

Чтобы получить список банковский терминов (и другие сущности), мы использовали следующий промпт:



___

## Шаг 1. Формирование словаря (`keywords.py`)

Сначала необходимо подготовить файл в `.txt` формате со всеми терминами, которые вы хотите искать.
Для примера подготовленный файл находится в `./input/keywords.txt`.

Вход:
>карта
>доставка
>мир
>карта
>банковская карта
>лучшая банковская карта
>дебетовая карта
>карты
>карту
>картой

Выход: 

| category | weight | word |
|----------|---------|------|
| _ | 3 | хороший банковский карта |
| _ | 2 | банковский карта |
| _ | 2 | дебетовый карта |
| _ | 1 | карта |
| _ | 1 | доставка |
| _ | 1 | мир |

- Вес на данный момент расчитывается как количество слов во фразе. Далее они будут учитываться в скоре как квадрат веса. Но вы можете задать и свой вес.
- Слова из исходного файла уже лемматизированы и дедуплицированы.

___

## Шаг 2. Поиск слов в диалогах (`tags.py`)

- `DialogProcessor` - основной класс. Метод `count_words` ищет и считает любые подстроки.
- `ExactWordDialogProcessor(DialogProcessor)` - наследник. Метод `count_words` ищет точные совпадения.

Выход:

| dialog_id | client_counts | operator_counts | intersection_counts |
|-----------|---------------|-----------------|-------------------|
| 1 | {'карта': 3, 'дебетовый карта': 1} | {'карта': 2} | {'карта': 2} |

**Основная идея**: если и клиент и оператор одновременно употребляли какие-то банковские термины, то с большой вероятностью тема обращения связана именно с этими словами. Это столбец `intersection_counts`.

 ---

## Шаг 3. Разметка

Необходимо присвоить одну категорию каждому слову или фразе.
Разметка ручная:(

| category | weight | word |
|----------|---------|------|
| карта | 3 | хороший банковский карта |
| карта | 2 | банковский карта |
| карта | 2 | дебетовый карта |
| карта | 1 | карта |
| доставка | 1 | доставка |
| другое | 1 | мир |

## Шаг 4. Формирование скора (`score.py`)

На данном этапе все реализовано сейчас топорно:

Вход: 
```python
{"карта": 1, "мир": 1, "дебетовый карта": 2}
```

Выход:
```python
{'карта': 9, 'другое': 1}
```

**Идея**: если над этими выходами применить Softmax, то вообще получим подобие вероятностей:

```python
{'карта': 0.9997, 'другое': 0.0003}
```

## Шаг 5. Принятие решения о продукте (не реализовано)

Предполагается реализовать бизнес-правила, которые позволят принимать решение о классе продукта в зависимости от итоговых скоров.
Данный функционал необходимо тестировать на реальных продуктах.

**Важно отметить**, что качество очень сильно зависит от формирования словаря (шаг 1).