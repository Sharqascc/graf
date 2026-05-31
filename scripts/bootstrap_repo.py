from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    print(f"GRAF repo root: {root}")

if __name__ == "__main__":
    main()
