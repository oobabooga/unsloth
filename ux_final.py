"""Final revision on hardware: the full A/B user steps, then the switch refusals and the run banner."""
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
import ux_ab
import ux_ab2

ux_ab.main()
ux_ab2.main()
