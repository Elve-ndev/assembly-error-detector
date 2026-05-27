API Reference
=============

The pipeline exposes one main class: ``CobotDecisionEngine``.
All other components (feature extraction, anomaly scoring) are called internally.

----

CobotDecisionEngine
-------------------

**Import**

.. code-block:: python

   from pipeline import CobotDecisionEngine

**Constructor**

.. code-block:: python

   engine = CobotDecisionEngine(
       bigru_checkpoint="checkpoints/bigru_2classes_best.pth",
       scaler_path="checkpoints/scaler.pkl",
       stride_map_path="data/stride_map_train.pkl"
   )

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``bigru_checkpoint``
     - str
     - Path to BiGRU dual-head ``.pth`` weights
   * - ``scaler_path``
     - str
     - Path to fitted ``StandardScaler`` ``.pkl``
   * - ``stride_map_path``
     - str
     - Path to stride alignment map ``.pkl``

----

predict()
---------

.. code-block:: python

   decision, message = engine.predict(feature_vector, timestamp)

**Parameters**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - ``feature_vector``
     - np.ndarray
     - 2304-dim SlowFast feature vector for the current frame
   * - ``timestamp``
     - int
     - Frame index used for temporal smoothing (window W=20)

**Returns** — tuple ``(decision: str, message: str)``

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Decision
     - Anomaly score
     - Meaning
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

**Example**

.. code-block:: python

   import numpy as np
   from pipeline import CobotDecisionEngine

   engine = CobotDecisionEngine(
       bigru_checkpoint="checkpoints/bigru_2classes_best.pth",
       scaler_path="checkpoints/scaler.pkl",
       stride_map_path="data/stride_map_train.pkl"
   )

   feat = np.random.randn(2304).astype(np.float32)
   decision, message = engine.predict(feat, timestamp=42)
   print(decision)   # e.g. 'STOP'
   print(message)    # e.g. 'Anomaly score 0.341 exceeds STOP threshold'

----

Feature extraction (standalone)
--------------------------------

To extract SlowFast features outside of the main pipeline:

.. code-block:: python

   SF_WEIGHTS = "checkpoints/meccano_slowfast_mapped_clean.pth"
   FEAT_DIR   = "data/features/"
   BATCH      = "train_p1"

See ``notebooks/01_slowfast_extraction.ipynb`` for the full extraction loop with stride map generation.
