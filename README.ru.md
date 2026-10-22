# Reversible Fractional Attention (RFA)

> Внимание как **линейная динамическая система** с **O(1)-состоянием** и
> **строго обратимым** шагом по времени. Research-grade PyTorch-имплементация,
> собранная из исходного `TRSS.ipynb` proof-of-concept.

[![CI](https://img.shields.io/badge/CI-pytest%20%2B%20ruff-blue)](.github/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Built with uv](https://img.shields.io/badge/built%20with-uv-purple)](https://docs.astral.sh/uv/)

Документация на других языках: [**English**](README.md)

---

## Кратко

Обычное softmax-внимание требует **O(L²)** памяти по длине последовательности.
Линейное внимание убирает softmax, но всё равно хранит все пары key/value.
**RFA** идёт дальше: рекуррентная схема

```
N_t = γ · N_{t-1} + k_t ⊗ v_t       (числитель, d×d матрица)
Z_t = γ · Z_{t-1} + k_t             (знаменатель, d-вектор)
o_t = W_o · (q_t^T N_t) / (q_t^T Z_t + ε)
```

держит **константное** состояние `(N, Z)`, размер которого не зависит от длины
контекста. Так как обновление — линейная комбинация предыдущего состояния и
свежей **дельты**, оно **точно обращается**: по `(N_t, Z_t)` и дельте
`(delta_N, delta_Z)` можно восстановить `(N_{t-1}, Z_{t-1})` без повторного
прогона:

```
N_{t-1} = (N_t - delta_N) / γ
Z_{t-1} = (Z_t - delta_Z) / γ
```

Это даёт дешёвый **откат по времени** при потоковом инференсе: нужно хранить
только дельты, причём их можно агрессивно квантовать (INT8 и даже Rank-1),
не нарушая обратимость.

## Ключевые свойства

| Свойство | Реализация | Тест |
|---|---|---|
| Состояние O(1) (не зависит от `seq_len`) | `ReversibleFractionalAttention.forward_sequence` | `test_state_does_not_grow_with_sequence_length` |
| Строгая обратимость (FP32) | `step_backward_reconstruct` | `test_multi_step_reversibility_real` |
| INT8-стриминг (~4× сжатие) | `QuantizedStreamDelta` | `test_decompress_roundtrip_is_close` |
| **Rank-1 стриминг** (до **128×** при `d=64`) | `Rank1Stream` | `test_rank1_rewind_reproduces_state_within_tolerance` |
| Параллельная форма для обучения (O(L²)) | `ParallelReversibleAttention` | `test_parallel_matches_step_wise_up_to_tolerance` |
| Комплексно-значный селективный вариант | `ComplexReversibleAttention` | `test_single_step_reversibility_complex` |

## Установка

```bash
git clone https://github.com/your-org/reversible-fractional-attention.git
cd reversible-fractional-attention
uv sync --extra dev         # или: pip install -e ".[dev]"
pytest                      # ~5 секунд на CPU
```

## Быстрый старт

```python
import torch
from rfa import ReversibleFractionalAttention

model = ReversibleFractionalAttention(d_model=64, decay_rate=0.98)
state = model.initial_state(batch_size=1)

for x_t in stream_of_tokens:
    step = model.step_forward(x_t, state)
    state = step.state
    # `step.delta` можно квантовать и стримить на диск / в сеть,
    # а потом использовать для точного обращения шага.
```

Откат на один шаг через квантованную дельту:

```python
from rfa import QuantizedStreamDelta

q_n, s_n = QuantizedStreamDelta.compress(step.delta[0])
q_z, s_z = QuantizedStreamDelta.compress(step.delta[1])
# ... сохранить (q_n, s_n, q_z, s_z) куда-нибудь ...
rec_n = QuantizedStreamDelta.decompress(q_n, s_n)
rec_z = QuantizedStreamDelta.decompress(q_z, s_z)
n_prev, z_prev = model.step_backward_reconstruct(
    n_curr, z_curr, (rec_n, rec_z)
)
```

## Примеры

Запускаются через `python examples/<file>.py`:

| Файл | Что показывает |
|---|---|
| `01_step_forward.py` | Константное O(1) состояние на 100 токенах |
| `02_reversibility_check.py` | Точная ошибка реконструкции ≈ `1e-6` |
| `03_int8_streaming.py` | INT8-стриминг + стресс-тест на 30 шагов |
| `04_rank1_compression.py` | Сжатие в 128× через Rank-1 стриминг |
| `05_parallel_training.py` | Обучение ассоциативной recall-задачи end-to-end |
| `06_complex_attention.py` | Комплексно-значный селективный вариант |

## Как это работает

### Пошаговая форма (инференс, O(1) память)

На каждом шаге модель получает токен `x_t`, считает проекции query/key/value
через `Linear` и обновляет два аккумулятора:

```
q_t = ELU(W_q x_t) + 1     # строго положительное ядро
k_t = ELU(W_k x_t) + 1
v_t = W_v x_t

N_t = γ N_{t-1} + k_t ⊗ v_t      # [d, d] внешнее произведение, с затуханием
Z_t = γ Z_{t-1} + k_t             # [d] вектор, с затуханием

o_t = W_o · (q_t^T N_t) / (q_t^T Z_t + ε)
```

`ELU + 1` используется как строго положительное ядро: держит знаменатель
вдали от нуля без явного `+ eps` и численно стабилен для больших по модулю
отрицательных пре-активаций (в отличие от `exp`).

### Параллельная форма (обучение, O(L²) память)

Для обучения материализуется каузальная матрица затухания
`Decay[t, i] = γ^{t-i}` при `t ≥ i`, и все позиции считаются одним батч-матмюлом.
Это **математически эквивалентно** пошаговой форме (тест
`test_parallel_matches_step_wise_up_to_tolerance` проверяет относительную
ошибку < `1e-4`). Коэффициент `γ` параметризован как
`sigmoid(decay_param * 10 + 2)`, чтобы оставаться в `(0, 1)` и обучаться
end-to-end.

### Rank-1 стриминг

Дельта числителя `delta_N = k ⊗ v` — **ранг-1** матрица. Вместо стриминга
полной `d × d` FP32-матрицы мы стримим только два `d`-вектора `k` и `v`
(каждый квантованный в INT8), а внешнее произведение восстанавливается на лету
при откате:

| `d_model` | FP32 матрица | Rank-1 INT8 | Сжатие |
|---:|---:|---:|---:|
| 32 | 4 КБ | 64 Б | 64× |
| 64 | 16 КБ | 128 Б | **128×** |
| 128 | 64 КБ | 256 Б | 256× |

Ошибка отката после 30 шагов остаётся ниже 1.5%
(см. `examples/04_rank1_compression.py`).

### Комплексный вариант

Комплексная версия `ComplexReversibleAttention` добавляет **обучаемый
поканальный фазовый поворот** `exp(iθ)`:

```
N_t = γ · e^{iθ} ⊙ N_{t-1} + k_t ⊗ v_t
```

Это делает рекурренцию **селективной по содержанию**, а не только по времени:
два токена с одним `k`, но в разные фазы будут по-разному интерферировать в
комплексной плоскости — модель может «поворачивать прочь» устаревшие
ассоциации.

## Структура репозитория

```
reversible-fractional-attention/
├── src/rfa/
│   ├── attention.py          # 3 слоя внимания
│   ├── streaming.py          # INT8 + Rank-1 стриминг
│   └── data.py               # TinyShakespeare лоадер
├── tests/                    # pytest-сьют (6 файлов)
├── examples/                 # 6 runnable-демо
├── notebooks/                # Очищенный .ipynb
├── .github/workflows/ci.yml  # Python 3.10 / 3.11 / 3.12
├── pyproject.toml            # uv + hatchling + ruff
└── README.md / README.ru.md
```

## Статус и ограничения

Это **research preview**, собранный из Colab-ноутбука. Это не drop-in замена
`nn.MultiheadAttention`:

- Пока нет multi-head варианта (головы свернуты в `d_model`).
- Знаменатель `q_t^T Z_t + ε` строго не нормализован; на очень длинных
  последовательностях с малым `γ` может дрейфовать амплитуда выхода.
- INT8-квантование использует per-tensor scale; per-channel был бы точнее
  для выбросов.
- Комплексный вариант не бенчмаркался на реальной LM-задаче.

Контрибьюции приветствуются — см. раздел [Контрибьюция](#контрибьюция) ниже.

## Контрибьюция

1. `uv sync --extra dev`
2. `pre-commit install`
3. Добавьте тест в `tests/`, который падает без вашего изменения.
4. `pytest` и `ruff check .` должны быть зелёными.
5. Откройте PR — CI гоняет Python 3.10/3.11/3.12.

## Цитирование

```bibtex
@misc{rfa2025,
  title  = {Reversible Fractional Attention},
  author = {RFA Contributors},
  year   = {2025},
  note   = {Research preview, Apache-2.0}
}
```

## Лицензия

Apache License 2.0 — см. [LICENSE](LICENSE).
