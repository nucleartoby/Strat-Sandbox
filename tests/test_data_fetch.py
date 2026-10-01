"""
Dukascopy ingestion: the decoder, the scaling trap, and the column-order trap.

Nothing here touches the network. The fetch path is exercised against cached
``.bi5`` payloads and hand-built CSVs, so the suite stays runnable offline and
does not hammer a free service.
"""
import lzma
import struct
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

from data_fetch import _decode_hour, _frame, point_value, to_engine_csv, to_pipeline_csv
from data_ingestion import load_duka_csv, load_tick_data


def make_bi5(records) -> bytes:
    """Build a .bi5 payload: LZMA-compressed 20-byte big-endian records.

    Layout is ``!IIIff``: ms-since-hour, ask, bid (both integer points), then
    ask and bid volume as floats in millions.
    """
    raw = b"".join(struct.pack("!IIIff", *r) for r in records)
    return lzma.compress(raw, format=lzma.FORMAT_ALONE)


def test_decode_hour_reconstructs_prices_and_times():
    # EUR/USD: 110412 points at 1e-5 = 1.10412
    payload = make_bi5([
        (0,     110415, 110412, 1.5, 2.0),
        (1_500, 110418, 110415, 0.5, 0.75),
    ])
    rows = _decode_hour(payload, date(2024, 1, 3), hour=10, scale=1e-5)
    assert len(rows) == 2

    ts, bid, ask, bid_sz, ask_sz = rows[0]
    assert ts == datetime(2024, 1, 3, 10, 0, 0)
    assert bid == pytest.approx(1.10412)
    assert ask == pytest.approx(1.10415)
    assert ask > bid, "ask/bid swapped: duka's record order is ask first"
    assert bid_sz == pytest.approx(2.0e6)   # volumes are in millions
    assert ask_sz == pytest.approx(1.5e6)

    # The hour offset is applied explicitly, not inferred from minute rollovers.
    assert rows[1][0] == datetime(2024, 1, 3, 10, 0, 1, 500_000)


def test_decode_hour_applies_the_hour_offset():
    """duka infers the hour by watching the minute counter roll over, which
    loses track across an hour with no ticks. Each hour is decoded on its own
    here, so a quiet hour cannot shift the ones after it."""
    payload = make_bi5([(0, 110415, 110412, 1.0, 1.0)])
    for hour in (0, 7, 23):
        rows = _decode_hour(payload, date(2024, 1, 3), hour=hour, scale=1e-5)
        assert rows[0][0].hour == hour


def test_empty_payload_is_not_an_error():
    """Dukascopy serves empty files for hours with no quotes (holidays)."""
    assert _decode_hour(b"", date(2024, 1, 3), 3, 1e-5) == []


def test_jpy_pairs_use_a_different_point_value():
    """duka hardcodes 1e-5 for everything, so USD/JPY comes back 100x small.

    Nothing downstream can detect that on its own -- the series is
    self-consistent, just wrong by two orders of magnitude.
    """
    assert point_value("EURUSD") == 1e-5
    assert point_value("GBPUSD") == 1e-5
    assert point_value("USDJPY") == 1e-3
    assert point_value("EURJPY") == 1e-3

    # 147123 points is 147.123 for a JPY cross, not 1.47123.
    payload = make_bi5([(0, 147125, 147123, 1.0, 1.0)])
    rows = _decode_hour(payload, date(2024, 1, 3), 10, point_value("USDJPY"))
    assert rows[0][1] == pytest.approx(147.123)

    wrong = _decode_hour(payload, date(2024, 1, 3), 10, 1e-5)
    assert wrong[0][1] == pytest.approx(1.47123), "this is the duka behaviour we avoid"


def test_frame_has_the_columns_the_pipeline_expects():
    rows = [(datetime(2024, 1, 3, 10, 0, i), 1.1041, 1.1043, 2e6, 1.5e6)
            for i in range(10)]
    df = _frame(rows)
    assert {"timestamp", "bid", "ask", "mid", "spread", "volume",
            "bid_size", "ask_size"} <= set(df.columns)
    assert df.attrs["volume_is_proxy"] is False
    assert (df["spread"] > 0).all()
    assert df["timestamp"].dt.tz is not None, "Dukascopy times are GMT; tz must be set"
    assert df["volume"].iloc[0] == pytest.approx(3.5e6)


def test_empty_frame_keeps_its_schema():
    df = _frame([])
    assert len(df) == 0
    assert {"timestamp", "bid", "ask", "mid", "spread", "volume"} <= set(df.columns)


def test_duka_csv_loader_handles_ask_before_bid(tmp_path):
    """duka writes time,ask,bid -- the reverse of the web export.

    Read with the web-export assumption, every spread is negative, the cleaner
    discards them all as crossed, and you get an empty frame with no error.
    """
    path = tmp_path / "EURUSD-2024_01_03-2024_01_03.csv"
    path.write_text(
        "2024-01-03 10:00:00.000000,1.10415,1.10412,1.5,2.0\n"
        "2024-01-03 10:00:01.500000,1.10418,1.10415,0.5,0.75\n"
    )
    df = load_duka_csv(str(path))
    assert len(df) == 2
    assert (df["ask"] > df["bid"]).all()
    assert df["bid"].iloc[0] == pytest.approx(1.10412)


def test_duka_csv_loader_handles_an_optional_header(tmp_path):
    path = tmp_path / "with_header.csv"
    path.write_text(
        "time,ask,bid,ask_volume,bid_volume\n"
        "2024-01-03 10:00:00.000000,1.10415,1.10412,1.5,2.0\n"
    )
    df = load_duka_csv(str(path))
    assert len(df) == 1
    assert df["ask"].iloc[0] == pytest.approx(1.10415)
    # Sniffed by header, so it must also be picked up by provider="auto".
    assert len(load_tick_data(str(path), provider="auto")) == 1


def test_duka_csv_loader_rejects_swapped_columns(tmp_path):
    path = tmp_path / "swapped.csv"
    path.write_text(
        "time,ask,bid,ask_volume,bid_volume\n"
        "2024-01-03 10:00:00.000000,1.10412,1.10415,1.5,2.0\n"
        "2024-01-03 10:00:01.000000,1.10413,1.10416,1.5,2.0\n"
    )
    with pytest.raises(ValueError, match="crossed"):
        load_duka_csv(str(path))


def test_duka_csv_loader_warns_about_jpy_scaling(tmp_path):
    path = tmp_path / "usdjpy.csv"
    path.write_text(
        "time,ask,bid,ask_volume,bid_volume\n"
        "2024-01-03 10:00:00.000000,0.00147125,0.00147123,1.5,2.0\n"
    )
    with pytest.warns(RuntimeWarning, match="100x too small"):
        load_duka_csv(str(path))


def test_csv_writers_round_trip(tmp_path):
    rows = [(datetime(2024, 1, 3, 10, 0, i), 1.1041, 1.1043, 2e6, 1.5e6)
            for i in range(5)]
    df = _frame(rows)

    pipeline_path = to_pipeline_csv(df, tmp_path / "ticks.csv")
    back = load_tick_data(str(pipeline_path), provider="generic")
    assert len(back) == len(df)
    np.testing.assert_allclose(back["bid"], df["bid"])

    engine_path = to_engine_csv(df, tmp_path / "engine.csv")
    header = engine_path.read_text().splitlines()[0]
    # The realtime binary requires these four by name.
    assert header.startswith("ts_ns,bid,ask,volume")
    first = engine_path.read_text().splitlines()[1].split(",")
    assert int(first[0]) == pd.Timestamp(df["timestamp"].iloc[0]).value


def test_loaders_tolerate_whole_second_timestamps(tmp_path):
    """Real tick data mixes whole seconds with sub-second precision.

    pandas infers one format from the first row and raises on the first row
    that differs -- which, on a real day, is dozens of rows in and looks like
    corrupt data rather than a parsing default.
    """
    path = tmp_path / "mixed_precision.csv"
    path.write_text(
        "timestamp,bid,ask,volume\n"
        "2024-01-03 00:00:01.255000+00:00,1.09416,1.09420,6300000\n"
        "2024-01-03 00:03:12+00:00,1.09417,1.09421,4500000\n"          # whole second
        "2024-01-03 00:03:12.485000+00:00,1.09418,1.09422,2700000\n"
    )
    df = load_tick_data(str(path), provider="generic")
    assert len(df) == 3, "a whole-second row must not be dropped or raise"
    assert df["timestamp"].is_monotonic_increasing
    assert df["timestamp"].dt.tz is not None


def test_blotter_tolerates_whole_second_timestamps(tmp_path):
    from data_ingestion import load_trade_blotter
    path = tmp_path / "fills.csv"
    path.write_text(
        "timestamp,side,price,size\n"
        "2024-01-03 09:15:03.221000+00:00,buy,1.09412,1000000\n"
        "2024-01-03 09:15:04+00:00,sell,1.09410,1000000\n"
    )
    df = load_trade_blotter(str(path))
    assert len(df) == 2
    assert list(df["side"]) == ["buy", "sell"]
