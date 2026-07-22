"""
temporal_tracker.py
--------------------
Armazena o historico de medicoes de vegetacao por camera/ponto fixo, e
compara ao longo do tempo para detectar crescimento (ex: vegetacao invadindo
o acostamento).

Usa SQLite: nao precisa de servidor de banco de dados, funciona bem tanto
embarcado (Raspberry Pi / Jetson) quanto em um servidor central. Se no futuro
o volume de cameras/dados crescer muito, a mesma logica migra facilmente para
Postgres/TimescaleDB.
"""

from __future__ import annotations
import sqlite3
import datetime as dt
from dataclasses import dataclass
from typing import Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS vegetation_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    coverage_pct REAL NOT NULL,
    area_px INTEGER NOT NULL,
    mask_path TEXT,
    image_path TEXT
);
CREATE INDEX IF NOT EXISTS idx_camera_time
    ON vegetation_records (camera_id, timestamp);
"""


@dataclass
class GrowthReport:
    camera_id: str
    current_coverage_pct: float
    previous_coverage_pct: Optional[float]
    delta_pct: Optional[float]
    days_between: Optional[float]
    rate_pct_per_day: Optional[float]
    alert: bool
    alert_reason: Optional[str] = None


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def record_measurement(db_path: str, camera_id: str, coverage_pct: float,
                        area_px: int, mask_path: str = None,
                        image_path: str = None,
                        timestamp: dt.datetime = None) -> None:
    timestamp = timestamp or dt.datetime.utcnow()
    conn = sqlite3.connect(db_path)
    conn.execute(
        """INSERT INTO vegetation_records
           (camera_id, timestamp, coverage_pct, area_px, mask_path, image_path)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (camera_id, timestamp.isoformat(), coverage_pct, area_px, mask_path, image_path),
    )
    conn.commit()
    conn.close()


def get_history(db_path: str, camera_id: str):
    conn = sqlite3.connect(db_path)
    cur = conn.execute(
        """SELECT timestamp, coverage_pct, area_px FROM vegetation_records
           WHERE camera_id = ? ORDER BY timestamp ASC""",
        (camera_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def compute_growth(db_path: str, camera_id: str,
                    alert_rate_pct_per_day: float = 0.5,
                    alert_absolute_pct: float = 30.0) -> GrowthReport:
    """Compara a ultima medicao com a anterior e sinaliza alerta se:
      - a taxa de crescimento (pontos percentuais de cobertura por dia)
        ultrapassar `alert_rate_pct_per_day`, OU
      - a cobertura absoluta atual ultrapassar `alert_absolute_pct`
        (ex: vegetacao ja cobre 30% do campo de visao da camera).

    Esses limiares (thresholds) sao um ponto de ajuste fino: o ideal e
    calibrar com base em fotos reais e no criterio de manutencao da rodovia
    (ex: distancia da vegetacao ate a pista, altura estimada, etc.).
    """
    history = get_history(db_path, camera_id)
    if not history:
        raise ValueError(f"Nenhum historico encontrado para camera_id={camera_id}")

    current = history[-1]
    current_ts = dt.datetime.fromisoformat(current[0])
    current_pct = current[1]

    if len(history) < 2:
        alert = current_pct >= alert_absolute_pct
        return GrowthReport(
            camera_id=camera_id,
            current_coverage_pct=current_pct,
            previous_coverage_pct=None,
            delta_pct=None,
            days_between=None,
            rate_pct_per_day=None,
            alert=alert,
            alert_reason="cobertura absoluta acima do limiar" if alert else None,
        )

    previous = history[-2]
    previous_ts = dt.datetime.fromisoformat(previous[0])
    previous_pct = previous[1]

    days_between = max((current_ts - previous_ts).total_seconds() / 86400.0, 1e-6)
    delta_pct = current_pct - previous_pct
    rate = delta_pct / days_between

    alert = False
    reason = None
    if rate >= alert_rate_pct_per_day:
        alert = True
        reason = f"taxa de crescimento ({rate:.2f} pp/dia) acima do limiar ({alert_rate_pct_per_day} pp/dia)"
    elif current_pct >= alert_absolute_pct:
        alert = True
        reason = f"cobertura absoluta ({current_pct:.1f}%) acima do limiar ({alert_absolute_pct}%)"

    return GrowthReport(
        camera_id=camera_id,
        current_coverage_pct=current_pct,
        previous_coverage_pct=previous_pct,
        delta_pct=delta_pct,
        days_between=days_between,
        rate_pct_per_day=rate,
        alert=alert,
        alert_reason=reason,
    )