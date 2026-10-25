from typing import Dict, List, Tuple
from core.types import PathInfo

# Тип для таблицы: (prev_patch_id, cur_patch_id) -> список путей
PairKey = Tuple[int, int]
PathTable = Dict[PairKey, List[PathInfo]]


class PairHashMap:
    """Простая хеш-таблица для хранения пар патчей и путей"""

    def __init__(self):
        self._table: PathTable = {}

    def add_path(self, prev_id: int, cur_id: int, distance: float, energy: float):
        """Добавить путь для пары (prev, cur)"""
        key = (prev_id, cur_id)
        if key not in self._table:
            self._table[key] = []
        self._table[key].append(PathInfo(distance=distance, energy=energy))

    def get_paths(self, prev_id: int, cur_id: int) -> List[PathInfo]:
        """Получить все пути для пары (prev, cur)"""
        key = (prev_id, cur_id)
        return self._table.get(key, [])

    def has_pair(self, prev_id: int, cur_id: int) -> bool:
        """Проверить, есть ли пара в таблице"""
        return (prev_id, cur_id) in self._table

    def get_stats(self) -> Dict:
        """Статистика таблицы"""
        total_paths = sum(len(paths) for paths in self._table.values())
        return {
            'unique_pairs': len(self._table),
            'total_paths': total_paths,
            'avg_paths_per_pair': total_paths / max(1, len(self._table))
        }