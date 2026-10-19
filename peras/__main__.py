"""Deprecated: use `python -m pythia`."""
import runpy

runpy.run_module("pythia", run_name="__main__", alter_sys=True)
