"""Unit tests for prepare.py's decode_station and build_nights."""

from __future__ import annotations

import math

import pandas as pd
import pytest

import prepare

FULL_HEADER = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION",
               "TMP", "DEW", "SLP", "WND", "GA1"]


def write_csv(path, rows, columns=FULL_HEADER):
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def isd_row(date, temp="+0050,1", dew="+0010,1", slp="10230,1",
            wnd="180,1,N,0015,1", ga1="02,1", **over):
    row = {"STATION": "12105499999", "DATE": date, "LATITUDE": "51.7",
           "LONGITUDE": "19.4", "ELEVATION": "180", "TMP": temp, "DEW": dew,
           "SLP": slp, "WND": wnd, "GA1": ga1}
    row.update(over)
    return row


class TestDecodeStation:
    def test_shifts_utc_to_local_standard_time(self, tmp_path):
        path = write_csv(tmp_path / "isd_pl_tomaszow_12105499999_2021.csv",
                         [isd_row("2021-04-01T17:00:00")])
        out = prepare.decode_station(path)
        assert out.loc[0, "lst"] == pd.Timestamp("2021-04-01 18:00:00")
        assert out["lst"].dt.tz is None

    def test_uk_station_keeps_utc_clock(self, tmp_path):
        path = write_csv(tmp_path / "isd_uk_cranwell_03379099999_2021.csv",
                         [isd_row("2021-04-01T17:00:00")])
        out = prepare.decode_station(path)
        assert out.loc[0, "lst"] == pd.Timestamp("2021-04-01 17:00:00")

    def test_decodes_all_measurement_columns(self, tmp_path):
        path = write_csv(tmp_path / "isd_pl_tomaszow_12105499999_2021.csv",
                         [isd_row("2021-04-01T17:00:00")])
        row = prepare.decode_station(path).iloc[0]
        assert row["station"] == "pl_tomaszow"
        assert (row["lat"], row["lon"], row["elev"]) == (51.7, 19.4, 180.0)
        assert row["temp_c"] == pytest.approx(5.0)
        assert row["dewpoint_c"] == pytest.approx(1.0)
        assert row["slp_hpa"] == pytest.approx(1023.0)
        assert row["wind_ms"] == pytest.approx(1.5)
        assert row["cloud_oktas"] == 2.0

    def test_missing_ga1_column_becomes_nan(self, tmp_path):
        cols = [c for c in FULL_HEADER if c != "GA1"]
        rows = [{k: v for k, v in isd_row("2021-04-01T17:00:00").items() if k != "GA1"}]
        path = write_csv(tmp_path / "isd_pl_tomaszow_12105499999_2021.csv", rows, cols)
        out = prepare.decode_station(path)
        assert "cloud_oktas" in out.columns
        assert math.isnan(out.loc[0, "cloud_oktas"])
        assert len(out) == 1

    def test_extra_columns_are_ignored(self, tmp_path):
        row = isd_row("2021-04-01T17:00:00")
        row["AA1"] = "01,0000,9,1"
        path = write_csv(tmp_path / "isd_pl_tomaszow_12105499999_2021.csv",
                         [row], FULL_HEADER + ["AA1"])
        out = prepare.decode_station(path)
        assert "AA1" not in out.columns
        assert len(out) == 1

    def test_unparseable_dates_are_dropped(self, tmp_path):
        path = write_csv(tmp_path / "isd_pl_tomaszow_12105499999_2021.csv",
                         [isd_row("not-a-date"), isd_row("2021-04-01T17:00:00")])
        out = prepare.decode_station(path)
        assert len(out) == 1

    def test_station_column_is_populated_for_every_row(self, tmp_path):
        path = write_csv(tmp_path / "isd_pl_leczyca_12105399999_2021.csv",
                         [isd_row("2021-04-01T17:00:00"),
                          isd_row("2021-04-01T18:00:00")])
        out = prepare.decode_station(path)
        assert list(out["station"]) == ["pl_leczyca"] * 2

    def test_unknown_station_prefix_raises(self, tmp_path):
        path = write_csv(tmp_path / "isd_de_berlin_10382099999_2021.csv",
                         [isd_row("2021-04-01T17:00:00")])
        with pytest.raises(ValueError, match="no LST offset"):
            prepare.decode_station(path)


def obs_frame(rows, station="pl_tomaszow"):
    """Build an LST observation frame like decode_station returns."""
    frame = pd.DataFrame(rows)
    frame["lst"] = pd.to_datetime(frame["lst"])
    frame["station"] = station
    for col, default in [("lat", 51.7), ("lon", 19.4), ("elev", 180.0),
                         ("dewpoint_c", 1.0), ("slp_hpa", 1020.0),
                         ("wind_ms", 1.0), ("cloud_oktas", 2.0)]:
        if col not in frame.columns:
            frame[col] = default
    return frame


def good_night(day="2021-04-01", temp=5.0, night_temps=(3.0, 1.0, -1.0), **kw):
    """A minimal night that passes every quality filter."""
    rows = [
        {"lst": f"{day} 12:00", "temp_c": temp + 4},
        {"lst": f"{day} 15:00", "temp_c": temp + 2},
        {"lst": f"{day} 18:00", "temp_c": temp},
    ]
    hours = ["21:00", "01:00", "05:00"]
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    for hour, value in zip(hours, night_temps):
        stamp = f"{day} {hour}" if hour == "21:00" else f"{next_day} {hour}"
        rows.append({"lst": stamp, "temp_c": value})
    return obs_frame(rows, **kw)


class TestBuildNights:
    def test_builds_one_row_per_night(self):
        nights, rejected = prepare.build_nights(good_night())
        assert len(nights) == 1
        # The post-midnight observations make the following date a candidate
        # evening too; it has no 18:00 snapshot and is rejected as such.
        assert "no_evening_snapshot" in rejected
        row = nights.iloc[0]
        assert row["station"] == "pl_tomaszow"
        assert row["date"] == "2021-04-01"
        assert row["month"] == 4
        assert row["doy"] == 91

    def test_features_come_from_the_1800_snapshot(self):
        nights, _ = prepare.build_nights(good_night(temp=5.0))
        row = nights.iloc[0]
        assert row["temp_c"] == pytest.approx(5.0)
        assert row["dewpoint_c"] == pytest.approx(1.0)
        assert row["dewpoint_depression_c"] == pytest.approx(4.0)

    def test_label_is_the_overnight_minimum(self):
        nights, _ = prepare.build_nights(good_night(night_temps=(3.0, -2.5, 0.5)))
        assert nights.iloc[0]["tmin_overnight_c"] == pytest.approx(-2.5)

    def test_label_ignores_daytime_and_gap_observations(self):
        obs = good_night(night_temps=(3.0, 1.0, 2.0))
        # A far colder reading in the excluded 18:00-20:00 gap must not become
        # the label, and neither must a cold post-08:00 morning reading.
        obs = pd.concat([obs, obs_frame([
            {"lst": "2021-04-01 19:00", "temp_c": -20.0},
            {"lst": "2021-04-02 09:00", "temp_c": -30.0},
        ])], ignore_index=True)
        nights, _ = prepare.build_nights(obs)
        assert nights.iloc[0]["tmin_overnight_c"] == pytest.approx(1.0)

    def test_trends_are_now_minus_past(self):
        obs = good_night(temp=5.0)
        obs = pd.concat([obs, obs_frame([
            {"lst": "2021-03-31 18:00", "temp_c": 9.0, "slp_hpa": 1010.0},
        ])], ignore_index=True)
        # 15:00 row (3 h before the cutoff) carries temp 7.0 and slp 1020.
        row = prepare.build_nights(obs)[0].iloc[0]
        assert row["temp_change_3h"] == pytest.approx(-2.0)
        assert row["temp_change_24h"] == pytest.approx(-4.0)
        assert row["slp_tendency_3h"] == pytest.approx(0.0)

    def test_trend_is_nan_without_a_past_observation(self):
        obs = obs_frame([
            {"lst": "2021-04-01 18:00", "temp_c": 5.0},
            {"lst": "2021-04-01 21:00", "temp_c": 3.0},
            {"lst": "2021-04-02 01:00", "temp_c": 1.0},
            {"lst": "2021-04-02 05:00", "temp_c": -1.0},
        ])
        row = prepare.build_nights(obs)[0].iloc[0]
        assert math.isnan(row["temp_change_3h"])
        assert math.isnan(row["temp_change_24h"])

    def test_radiative_potential_from_cloud_and_wind(self):
        obs = good_night()
        obs.loc[obs["lst"] == pd.Timestamp("2021-04-01 18:00"),
                ["cloud_oktas", "wind_ms"]] = [0.0, 1.0]
        row = prepare.build_nights(obs)[0].iloc[0]
        assert row["radiative_potential"] == pytest.approx(0.5)

    def test_radiative_potential_uses_defaults_when_cloud_and_wind_missing(self):
        obs = good_night()
        obs.loc[obs["lst"] == pd.Timestamp("2021-04-01 18:00"),
                ["cloud_oktas", "wind_ms"]] = [math.nan, math.nan]
        row = prepare.build_nights(obs)[0].iloc[0]
        # (1 - 4/8) / (1 + 3) with the documented mid-cloud / light-wind defaults
        assert row["radiative_potential"] == pytest.approx(0.125)
        assert math.isnan(row["cloud_oktas"])
        assert math.isnan(row["wind_ms"])

    def test_night_outside_the_risk_window_is_rejected(self):
        nights, rejected = prepare.build_nights(good_night(day="2021-07-01"))
        assert nights.empty
        assert rejected["out_of_window"] >= 1
        assert "no_evening_snapshot" not in rejected

    def test_missing_evening_snapshot_is_rejected(self):
        obs = good_night()
        obs = obs[obs["lst"] != pd.Timestamp("2021-04-01 18:00")]
        nights, rejected = prepare.build_nights(obs)
        assert nights.empty
        assert rejected["no_evening_snapshot"] >= 1

    def test_missing_evening_dewpoint_is_rejected(self):
        obs = good_night()
        obs.loc[obs["lst"] == pd.Timestamp("2021-04-01 18:00"), "dewpoint_c"] = math.nan
        nights, rejected = prepare.build_nights(obs)
        assert nights.empty
        assert rejected["no_evening_snapshot"] >= 1

    def test_too_few_night_observations_is_rejected(self):
        nights, rejected = prepare.build_nights(
            good_night(night_temps=(3.0, math.nan, -1.0)))
        assert nights.empty
        assert rejected["no_late_obs"] == 1

    def test_night_without_a_near_dawn_observation_is_rejected(self):
        obs = obs_frame([
            {"lst": "2021-04-01 18:00", "temp_c": 5.0},
            {"lst": "2021-04-01 21:00", "temp_c": 3.0},
            {"lst": "2021-04-01 22:00", "temp_c": 2.0},
            {"lst": "2021-04-01 23:00", "temp_c": 1.0},
        ])
        nights, rejected = prepare.build_nights(obs)
        assert nights.empty
        assert rejected["no_late_obs"] == 1

    def test_multiple_stations_are_grouped_independently(self):
        obs = pd.concat([good_night(station="pl_tomaszow"),
                         good_night(station="uk_cranwell", night_temps=(4.0, 2.0, 2.5))],
                        ignore_index=True)
        nights, _ = prepare.build_nights(obs)
        assert sorted(nights["station"]) == ["pl_tomaszow", "uk_cranwell"]
        by_station = nights.set_index("station")["tmin_overnight_c"]
        assert by_station["uk_cranwell"] == pytest.approx(2.0)

    def test_empty_input(self):
        empty = obs_frame([{"lst": "2021-04-01 18:00", "temp_c": 5.0}]).iloc[0:0]
        nights, rejected = prepare.build_nights(empty)
        assert nights.empty
        assert rejected == {}


def station_year_csv(path, day="2021-04-01"):
    """A raw CSV whose LST rows form one complete, passing night."""
    next_day = (pd.Timestamp(day) + pd.Timedelta(days=1)).date().isoformat()
    stamps = [f"{day}T11:00:00", f"{day}T14:00:00", f"{day}T17:00:00",
              f"{day}T20:00:00", f"{next_day}T00:00:00", f"{next_day}T04:00:00"]
    temps = ["+0090,1", "+0070,1", "+0050,1", "+0030,1", "+0010,1", "-0010,1"]
    return write_csv(path, [isd_row(s, temp=t) for s, t in zip(stamps, temps)])


class TestMain:
    def test_writes_nights_csv(self, tmp_path, monkeypatch, capsys):
        raw, out = tmp_path / "data_raw", tmp_path / "data"
        raw.mkdir()
        station_year_csv(raw / "isd_pl_tomaszow_12105499999_2021.csv")
        monkeypatch.setattr(prepare, "RAW_DIR", raw)
        monkeypatch.setattr(prepare, "OUT_DIR", out)
        prepare.main()
        nights = pd.read_csv(out / "nights.csv")
        assert len(nights) == 1
        assert nights.loc[0, "tmin_overnight_c"] == pytest.approx(-1.0)
        assert "frost-risk windows only" in capsys.readouterr().out

    def test_missing_raw_directory_is_a_clear_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(prepare, "RAW_DIR", tmp_path / "absent")
        with pytest.raises(FileNotFoundError, match="fetch_data.py"):
            prepare.main()

    def test_no_usable_nights_exits_non_zero(self, tmp_path, monkeypatch):
        raw, out = tmp_path / "data_raw", tmp_path / "data"
        raw.mkdir()
        # A summer station-year: every candidate night is out of the risk window.
        station_year_csv(raw / "isd_pl_tomaszow_12105499999_2021.csv", day="2021-07-01")
        monkeypatch.setattr(prepare, "RAW_DIR", raw)
        monkeypatch.setattr(prepare, "OUT_DIR", out)
        with pytest.raises(SystemExit, match="no usable nights"):
            prepare.main()
        assert not (out / "nights.csv").exists()
