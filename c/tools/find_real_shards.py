"""Locate which safetensors shards of a real (HF-hosted) colibri checkpoint hold the
embedding, lm_head, final norm, and the first `first_k_dense_replace` transformer
layers — the minimum needed to load and run a genuine (if artificially truncated,
dense-only) slice of a real model, without downloading the full multi-hundred-GB
checkpoint.

Works by HTTP-Range-fetching just the safetensors header (a few KB-hundred KB) of
every shard — never the multi-GB data section — so scanning the whole repo costs
tens of MB, not hundreds of GB. See scripts/verify_real_model.sh.
"""
import argparse
import json
import sys
import urllib.request


def api_get(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def range_get(url, start, end, timeout=60):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="e.g. jlnsrk/GLM-5.2-colibri-int4")
    ap.add_argument("--out", default="found_shards.json")
    args = ap.parse_args()

    base = f"https://huggingface.co/{args.repo}/resolve/main/"
    meta = api_get(f"https://huggingface.co/api/models/{args.repo}?blobs=true")
    shards = sorted(s["rfilename"] for s in meta["siblings"] if s["rfilename"].endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors files found in {args.repo}")
    print(f"{len(shards)} shards total", file=sys.stderr)

    cfg = api_get(base.replace("resolve/main/", "raw/main/config.json"))
    first_dense = cfg["first_k_dense_replace"]
    print(f"first_k_dense_replace={first_dense} -> need layers 0..{first_dense-1} (dense, no MoE)", file=sys.stderr)

    wanted = ["model.embed_tokens.", "lm_head.", "model.norm."]
    wanted += [f"model.layers.{i}." for i in range(first_dense)]

    found = {}
    for i, fn in enumerate(shards):
        url = base + fn
        first = range_get(url, 0, 15)
        hlen = int.from_bytes(first[:8], "little")
        hdr = range_get(url, 8, 8 + hlen - 1)
        obj = json.loads(hdr)
        matches = [k for k in obj if any(k.startswith(p) for p in wanted)]
        if matches:
            found[fn] = matches
            print(f"  {fn}: {len(matches)} matches", file=sys.stderr)
        if i % 20 == 0:
            print(f"...scanned {i}/{len(shards)}", file=sys.stderr)

    have = set()
    for names in found.values():
        for n in names:
            for p in wanted:
                if n.startswith(p):
                    have.add(p)
    missing = [p for p in wanted if p not in have]
    if missing:
        sys.exit(f"incomplete coverage, missing prefixes: {missing}")

    json.dump({"first_k_dense_replace": first_dense, "shards": sorted(found)}, open(args.out, "w"), indent=2)
    print("OK ->", args.out, file=sys.stderr)


if __name__ == "__main__":
    main()
