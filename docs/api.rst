API Reference
=============

Core Classes and Functions
--------------------------

Feature Extraction
~~~~~~~~~~~~~~~~~~

.. class:: SlowFastFeatureExtractor

   SlowFast R50 backbone for video feature extraction.
   
   **Attributes:**
   
   - ``backbone``: SlowFast R50 pre-trained on MECCANO
   - ``device``: torch.device for computation
   - ``feature_dim``: Output feature dimension (2304)
   
   **Methods:**
   
   .. method:: __init__(weights_path: str, device: str = "cuda")
   
      Initialize the feature extractor.
      
      **Args:**
         - ``weights_path`` (str): Path to pre-trained SlowFast weights
         - ``device`` (str): Computation device ("cuda" or "cpu")
      
      **Example:**
         >>> extractor = SlowFastFeatureExtractor(
         ...     weights_path="models/slowfast_r50.pth",
         ...     device="cuda"
         ... )
   
   .. method:: forward(frames: torch.Tensor) -> torch.Tensor
   
      Extract features from input frames.
      
      **Args:**
         - ``frames`` (torch.Tensor): Input frames, shape (batch, 3, H, W) or (batch, T, 3, H, W)
      
      **Returns:**
         torch.Tensor: Features of shape (batch, 2304) or (batch, T, 2304)
      
      **Example:**
         >>> frames = torch.randn(1, 3, 224, 224)  # Single frame
         >>> features = extractor(frames)
         >>> print(features.shape)
         torch.Size([1, 2304])


Model Architecture
~~~~~~~~~~~~~~~~~~

.. class:: BiGRUDualHead

   Bidirectional GRU with dual heads for action classification and anomaly detection.
   
   **Attributes:**
   
   - ``input_proj``: Input projection layer (feature_dim → hidden_dim)
   - ``bigru``: Bidirectional GRU (2 layers)
   - ``attention``: Temporal attention module
   - ``action_head``: Action classification head (hidden_dim*2 → num_classes)
   - ``anomaly_head``: Anomaly scoring head (hidden_dim*2 → 1)
   
   **Methods:**
   
   .. method:: __init__(feature_dim: int = 2304, hidden_dim: int = 256, num_classes: int = 10, dropout: float = 0.3)
   
      Initialize the BiGRU model.
      
      **Args:**
         - ``feature_dim`` (int): Input feature dimension (default: 2304)
         - ``hidden_dim`` (int): GRU hidden dimension (default: 256)
         - ``num_classes`` (int): Number of action classes (default: 10)
         - ``dropout`` (float): Dropout rate (default: 0.3)
      
      **Example:**
         >>> model = BiGRUDualHead(
         ...     feature_dim=2304,
         ...     hidden_dim=256,
         ...     num_classes=10
         ... )
   
   .. method:: forward(features: torch.Tensor, seq_length: int = None) -> Tuple[torch.Tensor, torch.Tensor]
   
      Forward pass through the model.
      
      **Args:**
         - ``features`` (torch.Tensor): Input features, shape (batch, seq_len, 2304)
         - ``seq_length`` (int, optional): Sequence length for masking
      
      **Returns:**
         Tuple[torch.Tensor, torch.Tensor]:
            - action_logits: Shape (batch, seq_len, num_classes)
            - anomaly_scores: Shape (batch, seq_len, 1)
      
      **Example:**
         >>> features = torch.randn(8, 16, 2304)  # Batch of 8 sequences
         >>> actions, anomalies = model(features)
         >>> print(actions.shape, anomalies.shape)
         torch.Size([8, 16, 10]) torch.Size([8, 16, 1])


Temporal Decoding
~~~~~~~~~~~~~~~~~

.. class:: ViterbiDecoder

   Viterbi decoder for temporal consistency in action sequences.
   
   **Methods:**
   
   .. method:: __init__(num_classes: int = 10, smoothing: float = 1.0, min_duration: int = 3)
   
      Initialize Viterbi decoder.
      
      **Args:**
         - ``num_classes`` (int): Number of action classes
         - ``smoothing`` (float): Laplace smoothing for transitions
         - ``min_duration`` (int): Minimum frames per action segment
      
      **Example:**
         >>> decoder = ViterbiDecoder(num_classes=10)
   
   .. method:: decode(action_logits: torch.Tensor) -> torch.Tensor
   
      Decode action sequence with temporal consistency.
      
      **Args:**
         - ``action_logits`` (torch.Tensor): Model predictions, shape (seq_len, num_classes)
      
      **Returns:**
         torch.Tensor: Decoded action sequence, shape (seq_len,)
      
      **Example:**
         >>> logits = torch.randn(100, 10)
         >>> decoded_actions = decoder.decode(logits)
         >>> print(decoded_actions.shape)
         torch.Size([100])


Anomaly Detection
~~~~~~~~~~~~~~~~~

.. class:: MahalanobisAnomalyDetector

   Mahalanobis distance-based anomaly detection using prototype bank.
   
   **Methods:**
   
   .. method:: __init__(num_classes: int = 10)
   
      Initialize anomaly detector.
      
      **Args:**
         - ``num_classes`` (int): Number of action classes
      
      **Example:**
         >>> detector = MahalanobisAnomalyDetector(num_classes=10)
   
   .. method:: fit(features: torch.Tensor, labels: torch.Tensor)
   
      Build prototype bank from training features.
      
      **Args:**
         - ``features`` (torch.Tensor): Training features, shape (N, feature_dim)
         - ``labels`` (torch.Tensor): Action labels, shape (N,)
      
      **Example:**
         >>> features = torch.randn(10000, 256)
         >>> labels = torch.randint(0, 10, (10000,))
         >>> detector.fit(features, labels)
   
   .. method:: score(features: torch.Tensor) -> torch.Tensor
   
      Compute anomaly scores for features.
      
      **Args:**
         - ``features`` (torch.Tensor): Features, shape (batch, feature_dim)
      
      **Returns:**
         torch.Tensor: Anomaly scores in [0, 1], shape (batch, 1)
      
      **Example:**
         >>> test_features = torch.randn(32, 256)
         >>> scores = detector.score(test_features)
         >>> print(scores.shape)
         torch.Size([32, 1])


Decision Engine
~~~~~~~~~~~~~~~

.. class:: CobotDecisionEngine

   Decision engine for cobot guidance based on predictions.
   
   **Methods:**
   
   .. method:: __init__(critical_threshold: float = 0.85, pause_threshold: float = 0.65, monitor_threshold: float = 0.45)
   
      Initialize decision engine.
      
      **Args:**
         - ``critical_threshold`` (float): Threshold for STOP decision
         - ``pause_threshold`` (float): Threshold for PAUSE decision
         - ``monitor_threshold`` (float): Threshold for MONITOR decision
      
      **Example:**
         >>> engine = CobotDecisionEngine(
         ...     critical_threshold=0.85,
         ...     pause_threshold=0.65,
         ...     monitor_threshold=0.45
         ... )
   
   .. method:: decide(action_logit: torch.Tensor, anomaly_score: float, consistency: bool = True) -> Tuple[str, str]
   
      Make cobot decision.
      
      **Args:**
         - ``action_logit`` (torch.Tensor): Action logits, shape (num_classes,)
         - ``anomaly_score`` (float): Anomaly score in [0, 1]
         - ``consistency`` (bool): Whether Viterbi and BiGRU agree
      
      **Returns:**
         Tuple[str, str]: (decision, reason)
         - decision: "NORMAL" | "MONITOR" | "PAUSE" | "STOP"
         - reason: Explanation string
      
      **Example:**
         >>> logits = torch.randn(10)
         >>> decision, reason = engine.decide(
         ...     action_logit=logits,
         ...     anomaly_score=0.5,
         ...     consistency=True
         ... )
         >>> print(decision, reason)
         NORMAL Action: TAKE with confidence 0.82


Data Loading
~~~~~~~~~~~~

.. class:: IndustRealDataset

   PyTorch Dataset for IndustReal industrial assembly videos.
   
   **Methods:**
   
   .. method:: __init__(root_path: str, split: str = "train", transform: callable = None)
   
      Initialize dataset.
      
      **Args:**
         - ``root_path`` (str): Path to IndustReal root directory
         - ``split`` (str): Data split ("train", "val", "test")
         - ``transform`` (callable): Optional transform to apply
      
      **Example:**
         >>> dataset = IndustRealDataset(
         ...     root_path="data/industreal",
         ...     split="train"
         ... )
   
   .. method:: __len__() -> int
   
      Return dataset size.
      
      **Example:**
         >>> len(dataset)
         5000
   
   .. method:: __getitem__(idx: int) -> Tuple[torch.Tensor, int, int]
   
      Get sample.
      
      **Args:**
         - ``idx`` (int): Sample index
      
      **Returns:**
         Tuple[torch.Tensor, int, int]:
            - features: Extracted features, shape (seq_len, 2304)
            - action_label: Action class ID (0-9)
            - anomaly_label: Anomaly flag (0 or 1)


Utility Functions
~~~~~~~~~~~~~~~~~

.. function:: preprocess_frame(frame: np.ndarray, config: dict) -> torch.Tensor

   Preprocess a single frame for model input.
   
   **Args:**
      - ``frame`` (np.ndarray): Input frame (H, W, 3) in BGR format
      - ``config`` (dict): Configuration dictionary
   
   **Returns:**
      torch.Tensor: Preprocessed frame, shape (3, H, W)
   
   **Example:**
      >>> import cv2
      >>> frame = cv2.imread("sample.jpg")
      >>> frame_tensor = preprocess_frame(frame, config)


.. function:: load_config(config_path: str) -> dict

   Load YAML configuration file.
   
   **Args:**
      - ``config_path`` (str): Path to config.yaml
   
   **Returns:**
      dict: Configuration dictionary
   
   **Example:**
      >>> config = load_config("config.yaml")
      >>> print(config["model"]["hidden_dim"])
      256


.. function:: compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict

   Compute classification metrics.
   
   **Args:**
      - ``predictions`` (np.ndarray): Predicted labels
      - ``targets`` (np.ndarray): Ground truth labels
   
   **Returns:**
      dict: Dictionary with keys: accuracy, precision, recall, f1
   
   **Example:**
      >>> metrics = compute_metrics(pred, target)
      >>> print(metrics["f1"])
      0.85


Exceptions
~~~~~~~~~~

.. exception:: ModelNotFoundError

   Raised when model weights cannot be found.
   
   **Example:**
      >>> try:
      ...     extractor = SlowFastFeatureExtractor("nonexistent.pth")
      ... except ModelNotFoundError as e:
      ...     print(f"Error: {e}")


.. exception:: ConfigurationError

   Raised when configuration is invalid.
   
   **Example:**
      >>> try:
      ...     config = load_config("invalid_config.yaml")
      ... except ConfigurationError as e:
      ...     print(f"Config error: {e}")


Constants
~~~~~~~~~

.. data:: ACTION_CLASSES

   Mapping of action class IDs to names.
   
   **Type:** dict
   
   **Example:**
      >>> from assembly_error_detector import ACTION_CLASSES
      >>> print(ACTION_CLASSES[0])
      TAKE


.. data:: DECISION_LEVELS

   Cobot intervention levels.
   
   **Type:** dict
   
   **Example:**
      >>> from assembly_error_detector import DECISION_LEVELS
      >>> print(DECISION_LEVELS["CRITICAL"])
      Stop and alert supervisor
