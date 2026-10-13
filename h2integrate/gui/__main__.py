"""Launch the H2Integrate GUI: ``python -m h2integrate.gui [streamlit options]``."""

import sys
from pathlib import Path


def main():
    try:
        from streamlit.web import cli as stcli
    except ImportError:
        sys.exit("The H2Integrate GUI requires Streamlit: pip install 'h2integrate[gui]'")

    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
