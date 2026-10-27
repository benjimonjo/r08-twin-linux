#!/usr/bin/env python3
"""Start the R08 live twin:   python run.py"""
import sys, traceback


def _missing(exc):
    print(f"\nMissing package: {exc}\nInstall with:\n  pip install numpy opencv-python mss pillow pynput moderngl\n"
          "(moderngl = smooth GPU 3D view; optional.  pytesseract + Tesseract are only needed for the optional 'Step size' text.)\n")
    try:
        input("Press Enter to close...")
    except Exception:
        pass
    sys.exit(1)


try:
    from r08.ui import main
except ImportError as exc:
    _missing(exc)
except Exception:
    traceback.print_exc()
    try:
        input("\nPress Enter to close...")
    except Exception:
        pass
    sys.exit(1)

if __name__ == "__main__":
    main()
