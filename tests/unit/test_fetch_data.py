"""Unit tests for fetch_data.py's download validation and CLI plumbing.

No test here touches the network: a fake session object stands in for
``requests.Session``.
"""

from __future__ import annotations

import threading

import pytest
import requests

import fetch_data
from frostlib import isd, net

CSV_BODY = '"STATION","DATE","TMP"\n"12105499999","2021-04-01T17:00:00","+0050,1"\n'


class FakeResponse:
    def __init__(self, text=CSV_BODY, status=200):
        self.text = text
        self.status_code = status

    @property
    def content(self):
        return self.text.encode("utf-8")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    def __init__(self, response=None):
        self.response = response if response is not None else FakeResponse()
        self.calls = []

    def get(self, url, headers=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "timeout": timeout})
        return self.response


class TestFetchStationYear:
    def test_writes_the_csv_and_summarises_it(self, tmp_path):
        out = tmp_path / "nested" / "isd_pl_tomaszow_12105499999_2021.csv"
        summary = fetch_data.fetch_station_year("12105499999", 2021, out, FakeSession())
        assert out.read_text() == CSV_BODY
        assert summary["rows"] == 1
        assert summary["output"] == str(out)
        assert summary["size_mb"] == pytest.approx(
            round(len(CSV_BODY) / 1_000_000, 2))

    def test_requests_the_expected_url_and_headers(self, tmp_path):
        session = FakeSession()
        fetch_data.fetch_station_year("12105499999", 2021,
                                      tmp_path / "out.csv", session)
        call = session.calls[0]
        assert call["url"] == f"{net.GLOBAL_HOURLY_BASE}/2021/12105499999.csv"
        assert call["headers"]["User-Agent"] == net.USER_AGENT
        assert call["timeout"] == net.TIMEOUT_SECONDS

    def test_accepts_an_unquoted_station_header(self, tmp_path):
        out = tmp_path / "out.csv"
        body = "STATION,DATE\n12105499999,2021-04-01T17:00:00\n"
        fetch_data.fetch_station_year("1", 2021, out, FakeSession(FakeResponse(body)))
        assert out.exists()

    def test_tolerates_a_utf8_bom(self, tmp_path):
        out = tmp_path / "out.csv"
        fetch_data.fetch_station_year("1", 2021, out,
                                      FakeSession(FakeResponse("\ufeff" + CSV_BODY)))
        assert out.exists()

    def test_http_error_propagates_and_writes_nothing(self, tmp_path):
        out = tmp_path / "out.csv"
        with pytest.raises(requests.HTTPError):
            fetch_data.fetch_station_year("1", 2021, out,
                                          FakeSession(FakeResponse(status=404)))
        assert not out.exists()

    @pytest.mark.parametrize("body", [
        "<html><body>Error: no STATION data</body></html>",
        "",
        '"DATE","TMP"\n"2021-04-01T17:00:00","+0050,1"\n',
    ])
    def test_non_isd_body_is_rejected_without_writing(self, tmp_path, body):
        out = tmp_path / "out.csv"
        with pytest.raises(ValueError, match="unexpected response"):
            fetch_data.fetch_station_year("1", 2021, out,
                                          FakeSession(FakeResponse(body)))
        assert not out.exists()

    def test_existing_file_is_left_intact_when_validation_fails(self, tmp_path):
        out = tmp_path / "out.csv"
        out.write_text("previous good data\n")
        with pytest.raises(ValueError):
            fetch_data.fetch_station_year("1", 2021, out,
                                          FakeSession(FakeResponse("<html>")))
        assert out.read_text() == "previous good data\n"


class TestSession:
    def test_reuses_one_session_per_thread(self):
        assert fetch_data._session() is fetch_data._session()

    def test_each_thread_gets_its_own_session(self):
        seen = []

        def record():
            seen.append(fetch_data._session())

        threads = [threading.Thread(target=record) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert seen[0] is not seen[1]


class TestStationConfiguration:
    def test_defaults_merge_both_regions(self):
        assert fetch_data.DEFAULT_STATIONS == {
            **fetch_data.POLAND_STATIONS, **fetch_data.UK_STATIONS}

    def test_every_default_name_carries_an_lst_prefix(self):
        for name in fetch_data.DEFAULT_STATIONS:
            assert isd.lst_offset(name) in (0, 1)

    def test_station_ids_are_unique(self):
        ids = list(fetch_data.DEFAULT_STATIONS.values())
        assert len(set(ids)) == len(ids)


class TestMainCli:
    def run_main(self, monkeypatch, tmp_path, argv, fetcher=None):
        calls = []

        def fake_fetch(station_id, year, out_path, session):
            calls.append((station_id, year, out_path))
            if fetcher is not None:
                return fetcher(station_id, year, out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(CSV_BODY)
            return {"rows": 1, "size_mb": 0.01, "output": str(out_path)}

        monkeypatch.setattr(fetch_data, "RAW_DIR", tmp_path)
        monkeypatch.setattr(fetch_data, "fetch_station_year", fake_fetch)
        monkeypatch.setattr("sys.argv", ["fetch_data.py", *argv])
        fetch_data.main()
        return calls

    def test_fetches_one_file_per_station_year(self, monkeypatch, tmp_path):
        calls = self.run_main(monkeypatch, tmp_path,
                             ["--stations", "pl_x=111", "uk_y=222",
                              "--years", "2020", "2021"])
        assert len(calls) == 4
        assert (tmp_path / "isd_pl_x_111_2020.csv").exists()

    def test_existing_files_are_skipped(self, monkeypatch, tmp_path):
        (tmp_path / "isd_pl_x_111_2020.csv").write_text(CSV_BODY)
        calls = self.run_main(monkeypatch, tmp_path,
                             ["--stations", "pl_x=111", "--years", "2020"])
        assert calls == []

    def test_force_redownloads_existing_files(self, monkeypatch, tmp_path):
        (tmp_path / "isd_pl_x_111_2020.csv").write_text("stale\n")
        calls = self.run_main(monkeypatch, tmp_path,
                             ["--stations", "pl_x=111", "--years", "2020", "--force"])
        assert len(calls) == 1
        assert (tmp_path / "isd_pl_x_111_2020.csv").read_text() == CSV_BODY

    def test_bare_station_id_is_rejected(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="name=id"):
            self.run_main(monkeypatch, tmp_path, ["--stations", "12105499999"])

    def test_default_stations_and_years_are_used(self, monkeypatch, tmp_path):
        calls = self.run_main(monkeypatch, tmp_path, [])
        assert len(calls) == (len(fetch_data.DEFAULT_STATIONS)
                              * len(fetch_data.DEFAULT_YEARS))

    def test_one_failure_does_not_abort_the_batch_but_exits_non_zero(
            self, monkeypatch, tmp_path):
        def flaky(station_id, year, out_path):
            if year == 2020:
                raise requests.HTTPError("503")
            out_path.write_text(CSV_BODY)
            return {"rows": 1, "size_mb": 0.01, "output": str(out_path)}

        with pytest.raises(SystemExit) as exc:
            self.run_main(monkeypatch, tmp_path,
                          ["--stations", "pl_x=111", "--years", "2020", "2021"],
                          fetcher=flaky)
        assert exc.value.code == 1
        assert (tmp_path / "isd_pl_x_111_2021.csv").exists()
