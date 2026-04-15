import _env_setup  # noqa: F401 — sets T-drive caches, DLL paths, imports torch first

from ocr.pipeline import run_pipeline


def main():
    print("Tirumala Document Processing Pipeline")
    print("=" * 50)
    run_pipeline()


if __name__ == "__main__":
    main()

