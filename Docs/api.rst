API Reference
=============

This page documents the main public interface of the pipeline.

CobotDecisionEngine
-------------------

The main inference class. Loads BiGRU weights, scaler, and stride map,
then produces per-frame cobot decisions from feature vectors.

**Import**

.. code-block:: python

   from pipeline import CobotDecisionEngine

**Constructor**

.. code-block:: python

   engine = CobotDecisionEngine(
       bigru_checkpoint = "checkpoints/bigru_2classes_best.pth",
       scaler_path      = "checkpoints/scaler.pkl",
       stride_map_path  = "data/stride_map_train.pkl"
   )

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``bigru_checkpoint``
     - str
     - Path to the BiGRU dual-head ``.pth`` file
   * - ``scaler_path``
     - str
     - Path to the fitted ``StandardScaler`` ``.pkl``
   * - ``stride_map_path``
     - str
     - Path to the stride alignment map ``.pkl``

**predict()**

.. code-block:: python

   decision, message = engine.predict(feature_vector, timestamp)

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``feature_vector``
     - np.ndarray
     - 2304-dim SlowFast feature vector for the current frame
   * - ``timestamp``
     - int
     - Frame index (used for temporal smoothing window)

Returns a tuple ``(decision: str, message: str)`` where decision is one of
``'NORMAL'``, ``'WATCH'``, ``'MONITOR'``, ``'PAUSE'``, ``'STOP'``.

**Example**

.. code-block:: python

   import numpy as np
   from pipeline import CobotDecisionEngine

   engine = CobotDecisionEngine(
       bigru_checkpoint="checkpoints/bigru_2classes_best.pth",
       scaler_path="checkpoints/scaler.pkl",
       stride_map_path="data/stride_map_train.pkl"
   )

   # Simulate a single feature vector
   feat = np.random.randn(2304).astype(np.float32)
   decision, message = engine.predict(feat, timestamp=42)
   print(f"Decision: {decision}")
   print(f"Message:  {message}")

----

Decision Thresholds
-------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Decision
     - Anomaly score
     - Cobot action
   * - ``NORMAL``
     - < 0.189
     - Continue procedure
   * - ``WATCH``
     - ≥ 0.189
     - Log event, increase observation
   * - ``MONITOR``
     - ≥ 0.221
     - Reduce speed, heightened attention
   * - ``PAUSE``
     - ≥ 0.261
     - Stop movement, request confirmation
   * - ``STOP``
     - ≥ 0.313
     - Full stop, supervisor alert

----

Feature Extraction
------------------

SlowFast features are extracted via a forward hook on ``blocks.4``:

.. code-block:: python

   SF_WEIGHTS = "checkpoints/meccano_slowfast_mapped_clean.pth"
   FEAT_DIR   = "data/features/"
   BATCH      = "train_p1"   # recording batch identifier

See ``notebooks/01_slowfast_extraction.ipynb`` for the full extraction pipeline.

----

Training
--------

.. code-block:: python

   TRAIN_FEAT_DIR = "data/features/train/"
   VAL_FEAT_DIR   = "data/features/val/"
   PREP_DIR       = "data/preprocessing/"

See ``notebooks/03_bigru_training.ipynb`` for training configuration, loss weights,
and PSR error masking details.
