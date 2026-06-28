"""Contract tests for every registered scraper.

Two layers:
  * Interface conformance, parametrized over every discovered scraper.
  * Fixture-based smoke tests that push small recorded responses through a fake
    HttpClient and assert at least one well-formed RawJob comes out. API scrapers
    plus the two straightforward HTML scrapers are covered this way; the id-walking
    iCIMS scraper and the browser-driven JS scraper get interface-only checks (no
    network / no browser in unit tests).

No live network is used here.
"""
from __future__ import annotations

import inspect
from urllib.parse import urlparse

import pytest

from scrapers import registry
from scrapers.base import BaseScraper, RawJob

EXPECTED_KEYS = {
    "greenhouse_api", "lever_api", "smartrecruiters", "workday",
    "icims", "avature", "static_html", "javascript_rendered", "afs", "elastic",
}


# --------------------------------------------------------------------------- #
# Fake HTTP client
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, text="", json_data=None, status_code=200, url="", cookies=None):
        self._text = text
        self._json = json_data
        self.status_code = status_code
        self.url = url
        self.cookies = cookies or {}

    @property
    def text(self):
        return self._text

    def json(self):
        if self._json is None:
            raise ValueError("no json body")
        return self._json


class FakeHttpClient:
    """Routes requests to canned values via (predicate, value) lists."""

    def __init__(self, json_routes=None, text_routes=None, resp_routes=None):
        self.json_routes = json_routes or []
        self.text_routes = text_routes or []
        self.resp_routes = resp_routes or []

    @staticmethod
    def _match(routes, url, default=None):
        for pred, val in routes:
            if pred(url):
                return val
        return default

    def get_json(self, url, **kw):
        return self._match(self.json_routes, url)

    def post_json(self, url, **kw):
        return self._match(self.json_routes, url)

    def get_text(self, url, **kw):
        return self._match(self.text_routes, url, default="")

    def get(self, url, **kw):
        return self._match(self.resp_routes, url)

    def post(self, url, **kw):
        return self._match(self.resp_routes, url)

    def close(self):
        pass


def assert_valid_job(job, expected_key, require_id=True):
    assert isinstance(job, RawJob)
    assert job.scraper_key == expected_key
    assert job.source_url, "source_url must be non-empty"
    assert job.raw_text and job.raw_text.strip(), "raw_text must be non-empty"
    assert job.title, "title must be non-empty"
    if require_id:
        assert job.job_id, "job_id must be non-empty"


# --------------------------------------------------------------------------- #
# Interface conformance (all scrapers)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", registry.all_classes(), ids=lambda c: c.SCRAPER_KEY)
def test_interface_contract(cls):
    assert issubclass(cls, BaseScraper)
    assert isinstance(cls.SCRAPER_KEY, str) and cls.SCRAPER_KEY, "SCRAPER_KEY required"
    assert isinstance(cls.SITE_HINTS, list)
    assert isinstance(cls.PRIORITY, int)
    assert inspect.isgeneratorfunction(cls.scrape), "scrape must be a generator"
    # instantiable with an injected http client
    inst = cls(http=FakeHttpClient())
    assert inst.SCRAPER_KEY == cls.SCRAPER_KEY
    # matches() is a usable classmethod
    assert isinstance(cls.matches("https://example.com"), bool)


def test_all_expected_scrapers_registered():
    keys = set(registry.all_keys())
    assert EXPECTED_KEYS <= keys, f"missing: {EXPECTED_KEYS - keys}"
    assert "template" not in keys and "" not in keys


def test_static_html_never_auto_matches():
    cls = registry.get_class("static_html")
    assert cls.SITE_HINTS == []
    assert cls.matches("https://anything.com/jobs/") is False


# --------------------------------------------------------------------------- #
# Fixture smoke tests
# --------------------------------------------------------------------------- #
def test_greenhouse_smoke():
    from scrapers.greenhouse_scraper import GreenhouseScraper

    fixture = {"jobs": [{
        "id": 123,
        "title": "Software Engineer",
        "updated_at": "2026-01-01T00:00:00-00:00",
        "location": {"name": "Remote"},
        "offices": [{"name": "Remote - US"}, {"name": "Remote - EU"}],
        "absolute_url": "https://boards.greenhouse.io/acme/jobs/123",
        "content": "&lt;p&gt;Build &lt;b&gt;things&lt;/b&gt;. Requires 5+ years.&lt;/p&gt;",
    }]}
    http = FakeHttpClient(json_routes=[(lambda u: "boards-api.greenhouse.io" in u, fixture)])
    jobs = list(GreenhouseScraper(http=http).scrape("https://boards.greenhouse.io/acme", "Acme"))
    assert len(jobs) == 1
    assert_valid_job(jobs[0], "greenhouse_api")
    assert "Build things" in jobs[0].raw_text  # HTML unescaped + stripped
    assert jobs[0].locations_all == ["Remote - US", "Remote - EU"]


def test_lever_smoke():
    from scrapers.lever_scraper import LeverScraper

    fixture = [{
        "id": "abc-1",
        "text": "Backend Engineer",
        "categories": {"location": "NYC", "commitment": "Full-time", "allLocations": ["NYC", "SF"]},
        "descriptionPlain": "We build APIs.",
        "lists": [{"text": "Requirements", "content": "&lt;ul&gt;&lt;li&gt;Python&lt;/li&gt;&lt;/ul&gt;"}],
        "additionalPlain": "Nice to have: Go.",
        "hostedUrl": "https://jobs.lever.co/acme/abc-1",
        "workplaceType": "remote",
        "createdAt": 1700000000000,
    }]
    http = FakeHttpClient(json_routes=[(lambda u: "api.lever.co" in u, fixture)])
    jobs = list(LeverScraper(http=http).scrape("https://jobs.lever.co/acme", "Acme"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "lever_api")
    assert "Python" in job.raw_text
    assert job.locations_all == ["NYC", "SF"]


def test_smartrecruiters_smoke():
    from scrapers.smartrecruiters_scraper import SmartRecruitersScraper

    sr_list = {"totalFound": 1, "content": [{
        "id": "P1",
        "name": "Data Scientist",
        "location": {"city": "Austin", "region": "TX", "country": "us", "remote": True},
        "typeOfEmployment": {"label": "Full-time"},
        "releasedDate": "2026-01-01T00:00:00.000Z",
        "company": {"name": "Acme"},
    }]}
    sr_detail = {"jobAd": {"sections": {
        "jobDescription": {"text": "&lt;p&gt;Analyze data.&lt;/p&gt;"},
        "qualifications": {"text": "&lt;p&gt;PhD preferred. 3+ years.&lt;/p&gt;"},
    }}}
    http = FakeHttpClient(json_routes=[
        (lambda u: "/postings/P1" in u, sr_detail),
        (lambda u: "api.smartrecruiters.com" in u, sr_list),
    ])
    jobs = list(SmartRecruitersScraper(http=http).scrape("https://jobs.smartrecruiters.com/Acme", "Acme"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "smartrecruiters")
    assert "Analyze data" in job.raw_text and "PhD preferred" in job.raw_text


def test_workday_smoke():
    from scrapers.workday_scraper import WorkdayScraper

    wd_list = {"total": 1, "jobPostings": [{
        "title": "ML Engineer",
        "externalPath": "/job/Austin/ML-Engineer_JR100",
        "locationsText": "Austin, TX",
        "postedOn": "Posted Today",
    }]}
    wd_detail = {"jobPostingInfo": {
        "id": "internalhash",
        "title": "ML Engineer",
        "jobRequisitionId": "JR100",
        "jobDescription": "&lt;p&gt;Train models.&#xa;5+ years.&lt;/p&gt;",
        "location": "Austin, TX",
        "timeType": "Full time",
        "startDate": "2026-01-01",
        "externalUrl": "https://acme.wd1.myworkdayjobs.com/en-US/careers/job/Austin/ML-Engineer_JR100",
    }}
    http = FakeHttpClient(json_routes=[
        (lambda u: u.endswith("/jobs"), wd_list),
        (lambda u: "/job/" in u, wd_detail),
    ])
    jobs = list(WorkdayScraper(http=http).scrape(
        "https://acme.wd1.myworkdayjobs.com/en-US/careers", "Acme"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "workday")
    assert job.job_id == "JR100"
    assert "Train models" in job.raw_text
    assert "&#xa;" not in job.raw_text  # entity decoded


def test_avature_smoke():
    from scrapers.avature_scraper import AvatureScraper

    listing = '<html><body><a href="/Acme/JobDetail?jobId=55">Engineer</a></body></html>'
    detail = "<html><head><title>Engineer</title></head><body><h1>Engineer</h1><p>Do things. 4+ years.</p></body></html>"
    http = FakeHttpClient(text_routes=[
        (lambda u: "JobDetail" in u, detail),
        (lambda u: True, listing),
    ])
    jobs = list(AvatureScraper(http=http).scrape("https://careers.acme.net/jobs/search", "Acme"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "avature")
    assert job.job_id == "55"
    assert "Do things" in job.raw_text


def test_afs_smoke():
    from scrapers.afs_scraper import AFSScraper

    page = (
        '<html><script>$A.initConfig({"fwuid":"FW123",'
        '"loaded":{"APPLICATION@markup://siteforce:communityApp":"TOK456"}})'
        "</script></html>"
    )
    aura_resp = {"actions": [{"state": "SUCCESS", "returnValue": {"returnValue": {
        "isError": False, "statusCode": 200, "totalResults": 1, "responseLst": [{
            "id": 999, "jobId": "7828", "title": "Software Engineer ",
            "location": {"name": "Washington, DC"},
            "offices": [{"name": "Washington, DC"}, {"name": "Remote - US"}],
            "jobUrl": "https://boards.greenhouse.io/accenturefederalservices/jobs/999?gh_jid=999#app",
            "jobContent": "<p>Build <b>things</b>. 2+ years.</p>",
            "clearance": [
                {"name": "Clearance Level", "value": "Secret", "value_type": "single_select"},
                {"name": "Job Category", "value": ["Cyber"], "value_type": "multi_select"},
            ],
        }],
    }}}]}
    http = FakeHttpClient(
        text_routes=[(lambda u: "search-jobs" in u, page)],
        json_routes=[(lambda u: "sfsites/aura" in u, aura_resp)],
    )
    jobs = list(AFSScraper(http=http).scrape(
        "https://smartafs.my.site.com/careers/s/search-jobs", "Accenture Federal Services"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "afs")
    assert job.job_id == "999"
    assert job.title == "Software Engineer"
    assert job.location == "Washington, DC"
    assert job.locations_all == ["Washington, DC", "Remote - US"]
    assert "Build things" in job.raw_text          # HTML unescaped + stripped
    assert "Clearance Level: Secret" in job.raw_text  # custom fields appended
    assert "Job Category: Cyber" in job.raw_text


def test_afs_skips_seen_url():
    from scrapers.afs_scraper import AFSScraper

    page = (
        '<html><script>x={"fwuid":"FW",'
        '"loaded":{"APPLICATION@markup://siteforce:communityApp":"TOK"}}</script></html>'
    )
    aura_resp = {"actions": [{"state": "SUCCESS", "returnValue": {"returnValue": {
        "isError": False, "totalResults": 1, "responseLst": [{
            "id": 999, "title": "Software Engineer", "location": {"name": "DC"},
            "jobUrl": "https://boards.greenhouse.io/accenturefederalservices/jobs/999?gh_jid=999#app",
            "jobContent": "SHOULD NOT MATTER",
        }],
    }}}]}
    http = FakeHttpClient(
        text_routes=[(lambda u: "search-jobs" in u, page)],
        json_routes=[(lambda u: "sfsites/aura" in u, aura_resp)],
    )
    scraper = AFSScraper(http=http)
    scraper.seen_urls = {
        "https://boards.greenhouse.io/accenturefederalservices/jobs/999?gh_jid=999#app"
    }
    jobs = list(scraper.scrape(
        "https://smartafs.my.site.com/careers/s/search-jobs", "AFS"))
    assert len(jobs) == 1
    assert jobs[0].already_seen is True
    assert jobs[0].raw_text == ""
    assert jobs[0].job_id == "999"
    assert jobs[0].title == "Software Engineer"


def _elastic_fixtures(content="Build distributed systems. 8+ years."):
    page = FakeResponse(text="<html></html>", cookies={"XSRF-TOKEN": "tok%3D%3D"})
    api_resp = {
        "meta": {"page": {"current": 1, "total_pages": 1, "total_results": 1}},
        "results": [{
            "title": {"raw": "Principal Software Engineer "},
            "content": {"raw": content},
            "location": {"raw": "United States"},
            "hybrid_locations": {"raw": "San Francisco, CA; Remote - US"},
            "subdivision": {"raw": "Engineering"},
            "category": {"raw": "Engineering"},
            "job_type": {"raw": "Remote"},
            "req_id": {"raw": "8000999"},
            "url": {"raw": "engineering/united-states/principal-software-engineer/8000999"},
            "created_at": {"raw": "2026-06-24T15:05:55+00:00"},
        }],
    }
    http = FakeHttpClient(
        resp_routes=[(lambda u: "/jobs/" in u, page)],
        json_routes=[(lambda u: "/api/appSearch" in u, api_resp)],
    )
    return http


def test_elastic_smoke():
    from scrapers.elastic_scraper import ElasticScraper

    http = _elastic_fixtures()
    jobs = list(ElasticScraper(http=http).scrape(
        "https://jobs.elastic.co/jobs/department/engineering"
        "?filters%5B0%5D%5Bfield%5D=location"
        "&filters%5B0%5D%5Bvalues%5D%5B0%5D=United%20States"
        "&filters%5B0%5D%5Btype%5D=any", "Elastic"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "elastic")
    assert job.job_id == "8000999"
    assert job.title == "Principal Software Engineer"  # trimmed
    assert job.source_url == (
        "https://jobs.elastic.co/jobs/engineering/united-states/"
        "principal-software-engineer/8000999")
    assert job.location == "United States"
    assert job.locations_all == ["United States", "San Francisco, CA", "Remote - US"]
    assert job.posted_date == "2026-06-24"
    assert job.platform == "greenhouse"
    assert "Build distributed systems" in job.raw_text
    assert "Department: Engineering" in job.raw_text  # facet fields appended


def test_elastic_filter_parsing():
    from scrapers.elastic_scraper import ElasticScraper

    p = urlparse(
        "https://jobs.elastic.co/jobs/department/engineering"
        "?filters%5B0%5D%5Bfield%5D=location"
        "&filters%5B0%5D%5Bvalues%5D%5B0%5D=United%20States"
        "&filters%5B0%5D%5Btype%5D=any")
    filters = ElasticScraper._parse_filters(p)
    # department -> subdivision value filter (path), plus the location query filter
    assert {"any": [{"subdivision": "Engineering"}]} in filters
    assert {"any": [{"location": "United States"}]} in filters


def test_elastic_no_token_yields_nothing():
    from scrapers.elastic_scraper import ElasticScraper

    page = FakeResponse(text="<html></html>", cookies={})  # no XSRF-TOKEN
    http = FakeHttpClient(resp_routes=[(lambda u: True, page)])
    jobs = list(ElasticScraper(http=http).scrape(
        "https://jobs.elastic.co/jobs/department/engineering", "Elastic"))
    assert jobs == []


def test_elastic_skips_seen_url():
    from scrapers.elastic_scraper import ElasticScraper

    http = _elastic_fixtures(content="SHOULD NOT MATTER")
    scraper = ElasticScraper(http=http)
    scraper.seen_urls = {
        "https://jobs.elastic.co/jobs/engineering/united-states/"
        "principal-software-engineer/8000999"
    }
    jobs = list(scraper.scrape(
        "https://jobs.elastic.co/jobs/department/engineering", "Elastic"))
    assert len(jobs) == 1
    assert jobs[0].already_seen is True
    assert jobs[0].raw_text == ""
    assert jobs[0].job_id == "8000999"
    assert jobs[0].title == "Principal Software Engineer"


def test_static_html_smoke():
    from scrapers.static_html_scraper import StaticHtmlScraper

    listing = '<html><body><a href="/jobs/eng-1">Engineer</a><a href="/about">x</a></body></html>'
    detail = "<html><head><title>Engineer</title></head><body><p>Write code. 2+ years.</p></body></html>"
    http = FakeHttpClient(text_routes=[
        (lambda u: u.endswith("/careers"), listing),
        (lambda u: "/jobs/" in u, detail),
    ])
    jobs = list(StaticHtmlScraper(http=http).scrape("https://acme.example/careers", "Acme"))
    assert len(jobs) == 1
    job = jobs[0]
    assert_valid_job(job, "static_html")
    assert "Write code" in job.raw_text


# --------------------------------------------------------------------------- #
# seen_urls skip: detail-fetch scrapers must NOT re-download known jobs
# --------------------------------------------------------------------------- #
def test_smartrecruiters_skips_seen_url():
    from scrapers.smartrecruiters_scraper import SmartRecruitersScraper

    sr_list = {"totalFound": 1, "content": [{
        "id": "P1", "name": "Data Scientist",
        "location": {"city": "Austin"}, "company": {"name": "Acme"},
    }]}
    sr_detail = {"jobAd": {"sections": {"jobDescription": {"text": "SHOULD NOT BE FETCHED"}}}}
    http = FakeHttpClient(json_routes=[
        (lambda u: "/postings/P1" in u, sr_detail),
        (lambda u: "api.smartrecruiters.com" in u, sr_list),
    ])
    scraper = SmartRecruitersScraper(http=http)
    scraper.seen_urls = {"https://jobs.smartrecruiters.com/Acme/P1"}
    jobs = list(scraper.scrape("https://jobs.smartrecruiters.com/Acme", "Acme"))
    assert len(jobs) == 1
    assert jobs[0].already_seen is True
    assert jobs[0].raw_text == ""       # detail page was NOT fetched
    assert jobs[0].job_id == "P1"


def test_avature_skips_seen_url():
    from scrapers.avature_scraper import AvatureScraper

    listing = '<html><body><a href="/Acme/JobDetail?jobId=55">Engineer</a></body></html>'
    detail = "<html><body><p>SHOULD NOT BE FETCHED</p></body></html>"
    http = FakeHttpClient(text_routes=[
        (lambda u: "JobDetail" in u, detail),
        (lambda u: True, listing),
    ])
    scraper = AvatureScraper(http=http)
    scraper.seen_urls = {"https://careers.acme.net/Acme/JobDetail?jobId=55"}
    jobs = list(scraper.scrape("https://careers.acme.net/jobs/search", "Acme"))
    assert len(jobs) == 1
    assert jobs[0].already_seen is True
    assert jobs[0].raw_text == ""
    assert jobs[0].job_id == "55"
    assert jobs[0].title == "Engineer"  # title came from the listing anchor, no fetch


def test_workday_skips_seen_url():
    from scrapers.workday_scraper import WorkdayScraper

    wd_list = {"total": 1, "jobPostings": [{
        "title": "ML Engineer",
        "externalPath": "/job/Austin/ML-Engineer_JR100",
        "locationsText": "Austin, TX",
    }]}
    wd_detail = {"jobPostingInfo": {"jobDescription": "SHOULD NOT BE FETCHED", "title": "ML Engineer"}}
    http = FakeHttpClient(json_routes=[
        (lambda u: u.endswith("/jobs"), wd_list),
        (lambda u: "/job/" in u, wd_detail),
    ])
    scraper = WorkdayScraper(http=http)
    scraper.seen_urls = {"https://acme.wd1.myworkdayjobs.com/careers/job/Austin/ML-Engineer_JR100"}
    jobs = list(scraper.scrape("https://acme.wd1.myworkdayjobs.com/en-US/careers", "Acme"))
    assert len(jobs) == 1
    assert jobs[0].already_seen is True
    assert jobs[0].raw_text == ""       # detail page was NOT fetched
    assert jobs[0].title == "ML Engineer"


def test_no_skip_when_seen_urls_empty():
    # default (empty seen_urls) => scrapers fetch details as before
    from scrapers.smartrecruiters_scraper import SmartRecruitersScraper

    sr_list = {"totalFound": 1, "content": [{"id": "P1", "name": "Data Scientist",
                                             "location": {"city": "Austin"}}]}
    sr_detail = {"jobAd": {"sections": {"jobDescription": {"text": "&lt;p&gt;Real text.&lt;/p&gt;"}}}}
    http = FakeHttpClient(json_routes=[
        (lambda u: "/postings/P1" in u, sr_detail),
        (lambda u: "api.smartrecruiters.com" in u, sr_list),
    ])
    jobs = list(SmartRecruitersScraper(http=http).scrape("https://jobs.smartrecruiters.com/Acme", "Acme"))
    assert len(jobs) == 1
    assert jobs[0].already_seen is False
    assert "Real text" in jobs[0].raw_text
