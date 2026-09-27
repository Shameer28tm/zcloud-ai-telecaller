import os
import sys
import site

site_pkg = os.path.abspath(os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages"))
site.addsitedir(site_pkg)
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pytest

if __name__ == "__main__":
    args = sys.argv[1:] if len(sys.argv) > 1 else ["tests/test_sip_foundation.py", "-v"]
    sys.exit(pytest.main(args))
