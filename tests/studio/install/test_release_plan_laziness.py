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
    ):
        self.resolved: list[str] = []
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
