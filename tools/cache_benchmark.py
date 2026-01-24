import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer", choices=("clip", "sarclip"), required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--cache-dir", default="work_dirs/sanity/scorer_cache")
    parser.add_argument("--out-json", default="work_dirs/sanity/cache_benchmark.json")
    parser.add_argument("--clip-model", default="RN50")
    parser.add_argument("--sarclip-model", default="RN50")
    parser.add_argument("--sarclip-pretrained", default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    image_path = Path(args.image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    prompts = [args.prompt]

    from sfod.scorers import ClipScorer, SarclipScorer, DiskCache
    from sfod.scorers.clip_scorer import ClipScorerConfig
    from sfod.scorers.sarclip_scorer import SarclipScorerConfig

    if args.scorer == "clip":
        scorer = ClipScorer(ClipScorerConfig(model=args.clip_model))
    else:
        pretrained = args.sarclip_pretrained.strip() or None
        scorer = SarclipScorer(SarclipScorerConfig(model=args.sarclip_model, pretrained=pretrained))

    cache = DiskCache(repo_root / args.cache_dir)

    stat = image_path.stat()
    key_obj = {
        "scorer": scorer.signature(),
        "image": str(image_path),
        "image_mtime_ns": int(stat.st_mtime_ns),
        "image_size": int(stat.st_size),
        "prompts": prompts,
    }

    runs = []
    for i in range(args.runs):
        t0 = time.time()
        entry, hit = cache.get_or_compute(key_obj, lambda: scorer.score(image_path=str(image_path), prompts=prompts))
        dt = time.time() - t0
        scores = entry["value"]
        runs.append({"run": i + 1, "hit": hit, "sec": dt, "scores": scores})
        print(f"[cache_benchmark] run={i+1} hit={hit} sec={dt:.3f} scores={scores}")

    out_json = (repo_root / args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(
            {
                "scorer": args.scorer,
                "image": str(image_path),
                "prompts": prompts,
                "cache_dir": str((repo_root / args.cache_dir).resolve()),
                "runs": runs,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[cache_benchmark] wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

