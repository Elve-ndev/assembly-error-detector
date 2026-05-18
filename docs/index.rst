assembly-error-detector Documentation
======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   architecture
   installation
   usage
   api
   references

Welcome to the Cobot Assembly Error Detection Pipeline documentation!

This project provides an end-to-end real-time system for assembly error detection in industrial environments, designed to guide collaborative robots through procedure monitoring and anomaly detection.

Key Features
------------

- **Real-time Detection**: SlowFast R50 backbone for egocentric video understanding
- **Semantic Grouping**: 72 fine-grained actions grouped into 10 semantic categories
- **BiGRU Dual-Head**: Simultaneous action classification and anomaly scoring
- **Viterbi Decoding**: Probabilistic temporal consistency enforcement
- **Mahalanobis Anomaly Detection**: Prototype-based unsupervised anomaly scoring
- **Cobot Decision Engine**: 4-level intervention framework (NORMAL, MONITOR, PAUSE, STOP)

Quick Links
-----------

- `GitHub Repository <https://github.com/Elve-ndev/assembly-error-detector>`_
- `IndustReal Dataset <https://github.com/TimSchoonbeek/IndustReal>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
