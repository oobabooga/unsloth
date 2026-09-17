# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the release walk-back is allowed to fetch before it is needed.

A plan costs two network reads, the release manifest and its checksum asset. Everything
past the first plan is consumed only when the plan before it is REJECTED: a bundle that
downloads and fails validation, or one built for a newer macOS than the host. macOS raises
the limit to 16 releases (#5883, #5896, #6494), so resolving the walk-back up front spent
30 requests on an ordinary update to throw them away.

Two guarantees here, and they are equally load-bearing:

  * the resolver fetches ONE release when the newest is usable, and
  * the walk-back still reaches older releases, in order, when it is not.

The macOS half of the same change: the newest release comes from the download host (the
CDN every other platform already uses), so an ordinary update makes no api.github.com call
at all, and the API listing is reached only when this host cannot use that release.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    # Same idiom as conftest.py: these test modules are not a package, so the directory
    # is what makes a sibling importable.
    sys.path.insert(0, str(TEST_DIR))

from _pr10648_helpers import llama_host, load_studio_module  # noqa: E402

# The install-tree writer the back-compat suite already maintains, reused rather than
# re-implemented so a change to what a real tree contains reaches this file too.
from test_keep_install_backcompat_9979 import MACOS, S12, build_install  # noqa: E402

ILP = load_studio_module("studio_install_llama_prebuilt_plan_laziness", "install_llama_prebuilt.py")

PUBLISHED_REPO = "unslothai/llama.cpp"
# Newest first, the order the resolver walks.
RELEASE_TAGS = [f"b{11000 - index}-mix-abcdef0" for index in range(20)]


def macos_host(**overrides):
    return llama_host(
        ILP.HostInfo,
        system = "Darwin",
        machine = "arm64",
        macos_version = overrides.pop("macos_version", (15, 5)),
        **overrides,
    )


def linux_host(**overrides):
    return llama_host(ILP.HostInfo, system = "Linux", machine = "x86_64", **overrides)


class ApiRateLimited(RuntimeError):
    """What fetch_json raises for a 403 from api.github.com: a bare RuntimeError."""


class FakeReleases:
    """Stand in for the two network reads a release plan costs, and count them.

    ``reject`` names the releases whose asset selection fails, which is what an unusable
    bundle looks like to the resolver: a too-new macOS build, or a release with no asset
    for this host.
    """

    def __init__(
        self,
        monkeypatch,
        *,
        reject: "set[str] | None" = None,
        cdn_tag: str | None = None,
        api_error: "BaseException | None" = None,
    ):
        self.resolved: list[str] = []
        self._api_error = api_error
        self.api_listing_reads = 0
        self.cdn_reads = 0
        self._reject = reject or set()
        self._cdn_tag = cdn_tag

        monkeypatch.setattr(ILP, "_download_host_resolve_enabled", lambda: cdn_tag is not None)
        monkeypatch.setattr(ILP, "_download_host_resolved_release", self._cdn_release)
        monkeypatch.setattr(ILP, "iter_published_release_bundles", self._api_bundles)
        monkeypatch.setattr(ILP, "validated_checksums_for_bundle", self._checksums)
        monkeypatch.setattr(ILP, "resolve_release_asset_choice", self._attempts)
        monkeypatch.setattr(
            ILP,
            "_linux_published_attempts",
            lambda host, bundle: self._attempts(host, bundle.upstream_tag, bundle, None),
        )
        monkeypatch.setattr(ILP, "apply_approved_hashes", lambda attempts, checksums: attempts)

    def _bundle(self, release_tag: str):
        return ILP.PublishedReleaseBundle(
            repo = PUBLISHED_REPO,
            release_tag = release_tag,
            upstream_tag = release_tag.split("-")[0],
            assets = {},
        )

    def _cdn_release(
        self,
        repo: str,
        published_release_tag: str = "",
    ):
        if self._cdn_tag is None:
            return None
        self.cdn_reads += 1
        bundle = self._bundle(self._cdn_tag)
        return ILP.ResolvedPublishedRelease(bundle = bundle, checksums = self._checksums(repo, bundle))

    def _api_bundles(self, repo: str):
        self.api_listing_reads += 1
        if self._api_error is not None:
            raise self._api_error
        for release_tag in RELEASE_TAGS:
            yield self._bundle(release_tag)

    def _checksums(self, repo: str, bundle):
        return ILP.ApprovedReleaseChecksums(
            repo = repo,
            release_tag = bundle.release_tag,
            upstream_tag = bundle.upstream_tag,
            source_commit = "deadbeef",
            artifacts = {},
        )

    def _attempts(self, host, resolved_tag, bundle, checksums):
        self.resolved.append(bundle.release_tag)
        if bundle.release_tag in self._reject:
            return []
        return [
            ILP.AssetChoice(
                repo = PUBLISHED_REPO,
                tag = bundle.release_tag,
                name = f"llama-{bundle.upstream_tag}-bin-macos-arm64.tar.gz",
                url = f"https://example.com/{bundle.upstream_tag}.tar.gz",
                source_label = "published",
                install_kind = "macos-arm64",
                expected_sha256 = "a" * 64,
            )
        ]


def resolve(host):
    return ILP.resolve_simple_install_release_plans("latest", host, PUBLISHED_REPO, "")


def test_a_usable_newest_release_resolves_exactly_one_plan(monkeypatch):
    releases = FakeReleases(monkeypatch)
    _tag, plans = resolve(macos_host())

    assert plans[0].release_tag == RELEASE_TAGS[0]
    assert releases.resolved == [RELEASE_TAGS[0]]


def test_the_walk_back_still_reaches_older_releases_in_order(monkeypatch):
    """Three unusable releases in a row, the shape #5896 was written for."""
    releases = FakeReleases(monkeypatch, reject = set(RELEASE_TAGS[:3]))
    _tag, plans = resolve(macos_host())

    assert plans[0].release_tag == RELEASE_TAGS[3]
    # The three rejected ones were read, and nothing past the first usable one.
    assert releases.resolved == RELEASE_TAGS[:4]


def test_asking_for_the_next_plan_resolves_one_more_and_no_further(monkeypatch):
    """What the installer does when a downloaded bundle fails validation."""
    releases = FakeReleases(monkeypatch)
    _tag, plans = resolve(macos_host())
    assert releases.resolved == [RELEASE_TAGS[0]]

    assert ILP._has_release_plan(plans, 1) is True
    assert plans[1].release_tag == RELEASE_TAGS[1]
    assert releases.resolved == RELEASE_TAGS[:2]


def test_the_walk_back_is_still_capped(monkeypatch):
    """Iterating everything stops at the macOS limit rather than the whole release list."""
    releases = FakeReleases(monkeypatch)
    _tag, plans = resolve(macos_host())

    assert len(list(plans)) == ILP.DEFAULT_MAX_MACOS_RELEASE_FALLBACKS
    assert releases.resolved == RELEASE_TAGS[: ILP.DEFAULT_MAX_MACOS_RELEASE_FALLBACKS]


def test_linux_still_takes_the_newest_release_from_the_download_host(monkeypatch):
    releases = FakeReleases(monkeypatch, cdn_tag = RELEASE_TAGS[0])
    _tag, plans = resolve(linux_host())

    assert len(list(plans)) == 1  # the CDN surfaces only the newest, as before
    assert releases.resolved == [RELEASE_TAGS[0]]
    assert releases.api_listing_reads == 0


def test_macos_takes_the_newest_release_from_the_download_host(monkeypatch):
    """The ordinary macOS update: CDN only, no api.github.com call."""
    releases = FakeReleases(monkeypatch, cdn_tag = RELEASE_TAGS[0])
    _tag, plans = resolve(macos_host())

    assert plans[0].release_tag == RELEASE_TAGS[0]
    assert releases.cdn_reads == 1
    assert releases.api_listing_reads == 0


def test_macos_falls_back_to_the_api_only_when_it_must_walk(monkeypatch):
    """The newest release is unusable here, so the walk-back needs the listing the CDN
    cannot give it -- and the release it already saw is not resolved twice."""
    releases = FakeReleases(monkeypatch, reject = {RELEASE_TAGS[0]}, cdn_tag = RELEASE_TAGS[0])
    _tag, plans = resolve(macos_host())

    assert plans[0].release_tag == RELEASE_TAGS[1]
    assert releases.cdn_reads == 1
    assert releases.api_listing_reads == 1
    assert releases.resolved == RELEASE_TAGS[:2]


def test_a_walk_that_finds_nothing_usable_still_fails(monkeypatch):
    FakeReleases(monkeypatch, reject = set(RELEASE_TAGS))

    with pytest.raises(ILP.PrebuiltFallback):
        resolve(macos_host())


def test_the_first_plan_is_resolved_before_the_resolver_returns(monkeypatch):
    """The caller turns a rate-limited listing into a source build, and can only do that
    while the resolver is on the stack, so the first fetch must not be deferred."""

    def _explode(repo: str):
        raise RuntimeError("GitHub API returned 403")

    monkeypatch.setattr(ILP, "_download_host_resolve_enabled", lambda: False)
    monkeypatch.setattr(ILP, "iter_published_release_bundles", _explode)

    with pytest.raises(RuntimeError, match = "403"):
        resolve(macos_host())


def test_a_403_on_the_deferred_lookup_reaches_the_installer_as_a_fallback(monkeypatch):
    """The newest release comes from the CDN, fails validation, and the walk-back the
    installer then asks for hits a rate-limited api.github.com.

    That second lookup happens inside the install loop, after the resolver returned, so
    without iter_release_plans / _has_release_plan reading it the raw RuntimeError would
    exit EXIT_ERROR -- no source build, and no keeping the install that is already there.
    """
    releases = FakeReleases(
        monkeypatch,
        cdn_tag = RELEASE_TAGS[0],
        api_error = ApiRateLimited("GitHub API returned 403 for .../releases?per_page=100"),
    )
    _tag, plans = resolve(macos_host())
    assert plans[0].release_tag == RELEASE_TAGS[0]
    assert releases.api_listing_reads == 0

    # Exactly what install_prebuilt does when a plan fails validation: ask whether there
    # is another one.
    with pytest.raises(ILP.PrebuiltFallback, match = "failed to inspect published releases"):
        ILP._has_release_plan(plans, 1, PUBLISHED_REPO)
    assert releases.api_listing_reads == 1


def test_the_install_loop_reads_a_deferred_403_the_same_way(monkeypatch):
    """The other half of the same call site: the loop's own iteration resolves plan 2.

    A separate plans object because a generator that has raised is finished -- asking the
    first one again would answer "no more plans" rather than re-raising.
    """
    FakeReleases(
        monkeypatch,
        cdn_tag = RELEASE_TAGS[0],
        api_error = ApiRateLimited("GitHub API returned 403 for .../releases?per_page=100"),
    )
    _tag, plans = resolve(macos_host())

    with pytest.raises(ILP.PrebuiltFallback, match = "failed to inspect published releases"):
        for _plan in ILP.iter_release_plans(plans, PUBLISHED_REPO):
            pass


def test_the_resolver_itself_still_fails_hard_on_a_403(monkeypatch):
    """--resolve-prebuilt caches a PrebuiltFallback as "no prebuilt" for 24h, so a
    transport failure must stay a hard error where that path can see it. Only the install
    path, which source builds instead of caching, reads it as a fallback."""
    FakeReleases(
        monkeypatch,
        api_error = ApiRateLimited("GitHub API returned 403 for .../releases?per_page=100"),
    )

    with pytest.raises(ApiRateLimited):
        resolve(macos_host())


def test_a_deferred_403_still_reaches_the_source_build_end_to_end(tmp_path, monkeypatch):
    """The whole path the reviewer's scenario walks: CDN plan, failed validation, 403.

    A deferred 403 has to land where a listing failure always landed: keep the install
    that is already there, or exit EXIT_FALLBACK so setup.sh builds from source. EXIT_ERROR,
    which a raw RuntimeError out of the deferred lookup produces, is neither -- the setup
    scripts do not source build on it.
    """
    plan = ILP.InstallReleasePlan(
        requested_tag = "latest",
        llama_tag = "b11000",
        release_tag = RELEASE_TAGS[0],
        attempts = [
            ILP.AssetChoice(
                repo = PUBLISHED_REPO,
                tag = RELEASE_TAGS[0],
                name = "llama-b11000-bin-macos-arm64.tar.gz",
                url = "https://example.com/llama-b11000-bin-macos-arm64.tar.gz",
                source_label = "published",
                install_kind = "macos-arm64",
                expected_sha256 = "a" * 64,
            )
        ],
        approved_checksums = ILP.ApprovedReleaseChecksums(
            repo = PUBLISHED_REPO,
            release_tag = RELEASE_TAGS[0],
            upstream_tag = "b11000",
            source_commit = "deadbeef",
            artifacts = {},
        ),
    )

    def _plans():
        # The CDN answers first; the walk-back behind it is api.github.com, rate limited.
        yield plan
        raise ApiRateLimited("GitHub API returned 403 for .../releases?per_page=100")

    monkeypatch.setattr(
        ILP,
        "_fork_manifest_release_plans",
        lambda *a, **k: ("latest", ILP.LazyReleasePlans(_plans())),
    )
    monkeypatch.setattr(ILP, "detect_host", lambda *a, **k: MACOS)
    monkeypatch.setattr(ILP, "collect_system_report", lambda *a, **k: "report")
    monkeypatch.setattr(ILP, "existing_install_matches_plan", lambda *a, **k: False)
    monkeypatch.setattr(ILP, "resolve_validation_model", lambda probe: probe)

    def _validation_fails(*args, **kwargs):
        raise ILP.PrebuiltFallback("staged bundle failed validation")

    monkeypatch.setattr(ILP, "validate_prebuilt_attempts", _validation_fails)

    # Runnable: the listing failure is transient, so the install already on disk is the
    # right answer. Returning is what main() turns into exit 0.
    install_dir = build_install(tmp_path, host = MACOS, marker = S12, payload_backend = "metal")
    ILP.install_prebuilt(install_dir, "latest", PUBLISHED_REPO, "")
    assert (install_dir / "llama-server").exists()

    # And with a tree that cannot run, the source-build fallback setup.sh branches on.
    broken = build_install(
        tmp_path / "broken", host = MACOS, marker = S12, payload_backend = "metal", runnable = False
    )
    with pytest.raises(SystemExit) as caught:
        ILP.install_prebuilt(broken, "latest", PUBLISHED_REPO, "")
    assert caught.value.code == ILP.EXIT_FALLBACK
