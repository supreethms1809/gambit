"""Baseline adapters. Each method emits a per-hypothesis map.

The shared hypothesis selector, the single budget conversion, and the failure
rule live here so a later method cannot pick its own classes or its own mask
conversion. Third-party algorithms are not modified.
"""
