import { useEffect, useMemo, useRef, useState } from 'react';
import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

const CONFIRMED_MIN_CONF = 0.30;
const CANDIDATE_MIN_CONF = 0.08;
const DEFAULT_MODEL = 'Xenova/detr-resnet-50';
const FALLBACK_MODEL = 'Xenova/yolos-tiny';
const detectorCache = new Map();
const DEMO_IMAGE_PATH = '/data/human_review_preview.jpg';

function getDetector(modelName, onProgress) {
  if (!detectorCache.has(modelName)) {
    detectorCache.set(
      modelName,
      pipeline('object-detection', modelName, {
        progress_callback: onProgress,
      })
    );
  }
  return detectorCache.get(modelName);
}

function round(value, digits = 4) {
  const factor = 10 ** digits;
  return Math.round(Number(value || 0) * factor) / factor;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function normalizeBox(box, frameWidth, frameHeight) {
  const rawX1 = box?.xmin ?? box?.x1 ?? box?.left ?? 0;
  const rawY1 = box?.ymin ?? box?.y1 ?? box?.top ?? 0;
  const rawX2 = box?.xmax ?? box?.x2 ?? ((box?.x ?? rawX1) + (box?.width ?? 0));
  const rawY2 = box?.ymax ?? box?.y2 ?? ((box?.y ?? rawY1) + (box?.height ?? 0));
  const x1 = clamp(Number(rawX1), 0, frameWidth);
  const y1 = clamp(Number(rawY1), 0, frameHeight);
  const x2 = clamp(Number(rawX2), 0, frameWidth);
  const y2 = clamp(Number(rawY2), 0, frameHeight);
  const left = Math.min(x1, x2);
  const top = Math.min(y1, y2);
  const right = Math.max(x1, x2);
  const bottom = Math.max(y1, y2);
  return {
    x1: round(left, 2),
    y1: round(top, 2),
    x2: round(right, 2),
    y2: round(bottom, 2),
    width: round(right - left, 2),
    height: round(bottom - top, 2),
  };
}

function classifyDetection(confidence, width, height, frameWidth, frameHeight) {
  const aspect = height > 0 ? width / height : 0;
  const relativeArea = (width * height) / Math.max(frameWidth * frameHeight, 1);

  if (confidence < CANDIDATE_MIN_CONF) {
    return {
      status: 'rejected_low_confidence',
      reason: 'below_candidate_confidence',
      review_required: false,
      candidate_strength: null,
      aspect_width_over_height: aspect,
      relative_area: relativeArea,
    };
  }

  if (aspect < 0.4 || aspect > 3.0 || width <= 0 || height <= 0) {
    return {
      status: 'rejected_shadow_candidate',
      reason: 'outside_plausible_top_down_geometry',
      review_required: false,
      candidate_strength: null,
      aspect_width_over_height: aspect,
      relative_area: relativeArea,
    };
  }

  if (confidence >= CONFIRMED_MIN_CONF) {
    return {
      status: 'confirmed',
      reason: 'confidence_meets_confirmed_threshold',
      review_required: false,
      candidate_strength: null,
      aspect_width_over_height: aspect,
      relative_area: relativeArea,
    };
  }

  return {
    status: 'candidate',
    reason: 'low_conf_valid_top_down_geometry',
    review_required: true,
    candidate_strength: aspect >= 0.6 && aspect <= 2.0 ? 'strong' : 'weak',
    aspect_width_over_height: aspect,
    relative_area: relativeArea,
  };
}

function countsFromPayload(payload) {
  const detections = payload?.detections ?? [];
  return {
    confirmed: detections.filter((det) => det.status === 'confirmed').length,
    candidate: detections.filter((det) => det.status === 'candidate').length,
    rejected: detections.filter((det) => String(det.status).startsWith('rejected')).length,
  };
}

function drawPreview(canvas, image, detections) {
  const ctx = canvas.getContext('2d');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
  ctx.lineWidth = Math.max(2, Math.round(canvas.width / 500));
  ctx.font = `${Math.max(13, Math.round(canvas.width / 85))}px system-ui, sans-serif`;
  ctx.textBaseline = 'top';

  detections
    .filter((det) => det.status === 'confirmed' || det.status === 'candidate')
    .forEach((det) => {
      const [x, y, width, height] = det.bbox_xywh;
      const color = det.status === 'confirmed' ? '#16a34a' : '#facc15';
      const label = `${det.status} ${(det.confidence * 100).toFixed(1)}%`;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.strokeRect(x, y, width, height);
      const textWidth = ctx.measureText(label).width + 10;
      ctx.fillRect(x, Math.max(0, y - 22), textWidth, 20);
      ctx.fillStyle = det.status === 'confirmed' ? '#ffffff' : '#1f2937';
      ctx.fillText(label, x + 5, Math.max(0, y - 20));
    });
}

async function loadImage(file) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.decoding = 'async';
    image.src = url;
    await image.decode();
    return { image, url };
  } catch (error) {
    URL.revokeObjectURL(url);
    throw error;
  }
}

export default function BrowserCvDetector({ backendOnline, cudaDataLoaded, onSaved, onOpenTacticalMap }) {
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const imageUrlRef = useRef(null);
  const processedFileRef = useRef(null);
  const inFlightRunRef = useRef(null);
  const savedPayloadRef = useRef(null);
  const savingPayloadRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedFileId, setSelectedFileId] = useState(null);
  const [inputSource, setInputSource] = useState('none');
  const [cameraStream, setCameraStream] = useState(null);
  const [cameraError, setCameraError] = useState(null);
  const [engineMode, setEngineMode] = useState('browser');
  const [modelName, setModelName] = useState(DEFAULT_MODEL);
  const [modelStatus, setModelStatus] = useState('Not loaded');
  const [progress, setProgress] = useState('');
  const [payload, setPayload] = useState(null);
  const [saveResult, setSaveResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [syncMessage, setSyncMessage] = useState('');

  const counts = useMemo(() => countsFromPayload(payload), [payload]);
  const reviewRequired = Boolean(payload?.metadata?.review_required);
  const browserCvStatus = error
    ? 'Error'
    : modelStatus === 'Model loaded'
      ? 'Available'
      : 'Model not loaded';
  const saveFuseStatus = backendOnline && payload ? 'Enabled' : 'Disabled';

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  useEffect(() => {
    return () => {
      cameraStream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraStream]);

  const makeFileId = (file, source) => {
    if (!file) return null;
    return `${source}:${file.name}:${file.size}:${file.type}:${file.lastModified ?? 0}`;
  };

  const setActiveFile = (file, source) => {
    const nextFileId = makeFileId(file, source);
    setSelectedFile(file);
    setSelectedFileId(nextFileId);
    setInputSource(file ? source : 'none');
    setPayload(null);
    setSaveResult(null);
    setError(null);
    setSyncMessage('');
    setProgress(file ? 'Image loaded. Ready for browser detection.' : '');
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;
    setActiveFile(file, file ? 'upload' : 'none');
  };

  const startCamera = async () => {
    setCameraError(null);
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError('Camera is not available in this browser. Upload an image instead.');
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' },
        audio: false,
      });
      setCameraStream(stream);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      setCameraError(err.message || 'Camera permission failed. Upload an image instead.');
    }
  };

  const stopCamera = () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
    setCameraStream(null);
  };

  const captureCameraFrame = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) {
      setCameraError('Camera frame is not ready yet.');
      return;
    }

    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const context = captureCanvas.getContext('2d');
    context.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    captureCanvas.toBlob((blob) => {
      if (!blob) {
        setCameraError('Could not capture camera frame. Upload an image instead.');
        return;
      }
      const file = new File([blob], `camera-uav-frame-${Date.now()}.jpg`, { type: 'image/jpeg' });
      setActiveFile(file, 'camera');
      setCameraError(null);
    }, 'image/jpeg', 0.92);
  };

  const useDemoImage = async () => {
    setError(null);
    setCameraError(null);
    try {
      const response = await fetch(DEMO_IMAGE_PATH);
      if (!response.ok) throw new Error('Demo image is not available.');
      const blob = await response.blob();
      const file = new File([blob], 'human-reviewed-demo.jpg', { type: blob.type || 'image/jpeg' });
      setActiveFile(file, 'demo');
    } catch (err) {
      setError(err.message || 'Could not load demo image.');
    }
  };

  const saveDetections = async (payloadOverride = payload) => {
    if (!payloadOverride) {
      setError('Run browser detection before saving.');
      return null;
    }

    const payloadKey = payloadOverride.run_id ?? payloadOverride.timestamp;
    if (savedPayloadRef.current === payloadKey || savingPayloadRef.current === payloadKey) {
      return saveResult;
    }

    savingPayloadRef.current = payloadKey;
    setIsSaving(true);
    setError(null);
    try {
      const previewImageBase64 = canvasRef.current.toDataURL('image/jpeg', 0.9);
      const response = await fetch('http://127.0.0.1:8000/api/browser-cv/save_detections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payloadOverride,
          preview_base64: previewImageBase64,
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.message || 'Could not save browser detections.');
      }
      savedPayloadRef.current = payloadKey;
      setSaveResult(result);
      setSyncMessage('Detection saved and tactical fusion updated.');
      await onSaved?.();
      return result;
    } catch (err) {
      setError(err.message || 'Could not save browser detections.');
      return null;
    } finally {
      savingPayloadRef.current = null;
      setIsSaving(false);
    }
  };

  const runDetection = async ({ fileOverride = selectedFile, runId = selectedFileId, autoSave = false } = {}) => {
    if (!fileOverride) {
      setError('Upload a UAV image before running browser detection.');
      return null;
    }

    if (inFlightRunRef.current === runId) {
      return null;
    }

    inFlightRunRef.current = runId;
    setIsRunning(true);
    setSaveResult(null);
    setError(null);
    setSyncMessage('');
    setModelStatus('Loading model');
    setProgress('');

    try {
      const detector = await getDetector(modelName, (event) => {
        const label = event.file ? `${event.status}: ${event.file}` : event.status;
        const pct = Number.isFinite(event.progress) ? ` ${Math.round(event.progress)}%` : '';
        setProgress(`${label ?? 'loading'}${pct}`);
      });
      setModelStatus('Model loaded');
      setProgress('Running local browser inference');

      const { image, url } = await loadImage(fileOverride);
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
      imageUrlRef.current = url;

      console.log('[BROWSER_CV] selected file:', fileOverride.name, fileOverride.type, fileOverride.size);
      console.log('[BROWSER_CV] inference input type:', typeof url, url);

      const results = await detector(url, {
        threshold: Math.min(CANDIDATE_MIN_CONF, 0.05),
      });
      const personResults = results.filter((result) => {
        const label = String(result.label ?? result.class ?? '').toLowerCase();
        return label === 'person';
      });
      const frameWidth = image.naturalWidth;
      const frameHeight = image.naturalHeight;
      const detections = personResults.map((result, index) => {
        const box = normalizeBox(result.box, frameWidth, frameHeight);
        const confidence = round(result.score ?? result.confidence ?? 0, 6);
        const classification = classifyDetection(
          confidence,
          box.width,
          box.height,
          frameWidth,
          frameHeight
        );
        return {
          id: `det_${String(index + 1).padStart(3, '0')}`,
          class_name: String(result.label ?? result.class ?? 'object').toLowerCase(),
          confidence,
          bbox_xyxy: [box.x1, box.y1, box.x2, box.y2],
          bbox_xywh: [box.x1, box.y1, box.width, box.height],
          status: classification.status,
          reason: classification.reason,
          candidate_strength: classification.candidate_strength,
          review_required: classification.review_required,
          source: 'browser_transformers',
          model: modelName,
          stage: 'browser_inference',
          aspect_width_over_height: round(classification.aspect_width_over_height, 4),
          relative_area: round(classification.relative_area, 6),
        };
      });

      drawPreview(canvasRef.current, image, detections);

      const confirmedCount = detections.filter((det) => det.status === 'confirmed').length;
      const candidateCount = detections.filter((det) => det.status === 'candidate').length;
      const rejectedCount = detections.filter((det) => String(det.status).startsWith('rejected')).length;
      const nextPayload = {
        source: 'browser_transformers',
        run_id: runId,
        timestamp: new Date().toISOString(),
        frame_width: frameWidth,
        frame_height: frameHeight,
        metadata: {
          model: modelName,
          confirmed_count: confirmedCount,
          candidate_count: candidateCount,
          rejected_count: rejectedCount,
          review_required: candidateCount > 0,
          confirmed_min_conf: CONFIRMED_MIN_CONF,
          candidate_min_conf: CANDIDATE_MIN_CONF,
          confidence_policy: 'confirmed_only_for_tactical_fusion',
        },
        detections,
      };
      setPayload(nextPayload);
      setProgress(
        personResults.length === 0
          ? 'No person detections found.'
          : `Detection complete: ${confirmedCount} confirmed, ${candidateCount} candidate, ${rejectedCount} rejected`
      );
      if (autoSave) {
        if (backendOnline) {
          await saveDetections(nextPayload);
        } else {
          setSyncMessage('Detection completed locally. Start backend to save detections and update tactical fusion.');
        }
      }
      return nextPayload;
    } catch (err) {
      setError(err.message || 'Browser detection failed.');
      setModelStatus('Detection failed');
      return null;
    } finally {
      setIsRunning(false);
      inFlightRunRef.current = null;
    }
  };

  useEffect(() => {
    if (!selectedFile || !selectedFileId || engineMode !== 'browser') return;
    if (processedFileRef.current === selectedFileId) return;
    processedFileRef.current = selectedFileId;
    runDetection({ fileOverride: selectedFile, runId: selectedFileId, autoSave: true });
  }, [selectedFile, selectedFileId, engineMode, backendOnline]);

  const resetInput = () => {
    processedFileRef.current = null;
    savedPayloadRef.current = null;
    setSelectedFile(null);
    setSelectedFileId(null);
    setInputSource('none');
    setPayload(null);
    setSaveResult(null);
    setError(null);
    setSyncMessage('');
    setProgress('');
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const statusMessage = counts.confirmed > 0
    ? 'Confirmed browser detections found. Tactical Fusion can use confirmed targets.'
    : counts.candidate > 0
      ? 'Browser detection found low-confidence candidates only. Human review required. Tactical Command uses CUDA risk ranking fallback.'
      : payload
        ? 'No confirmed browser detections found. Tactical Command uses CUDA risk ranking fallback.'
        : 'Upload an image and run browser-side detection.';
  const inputLabel = inputSource === 'upload'
    ? 'Uploaded UAV image'
    : inputSource === 'camera'
      ? 'Captured camera frame'
      : inputSource === 'demo'
        ? 'Human-reviewed demo image'
        : 'No image selected';
  const browserStatusLabel = error ? 'Error' : isRunning ? 'Loading' : 'Ready';
  const detectedPeople = counts.confirmed + counts.candidate;

  return (
    <section className="panel browser-cv-card">
      <div className="browser-cv-simple-header">
        <div>
          <span>Primary MVP path</span>
          <h2>Browser Transformers.js Detection</h2>
          <p>Select an input. Detection runs automatically in the browser; backend sync runs automatically when available.</p>
        </div>
      </div>

      <div className="browser-cv-simple-status">
        <span>Browser CV: <strong>{browserStatusLabel}</strong></span>
        <span>Backend Sync: <strong>{backendOnline ? 'Online' : 'Offline'}</strong></span>
        {syncMessage ? <em>{syncMessage}</em> : null}
      </div>

      <h3 className="browser-cv-section-label">Input Source</h3>
      <div className="cv-input-source-grid">
        <article className={inputSource === 'upload' ? 'is-active' : ''}>
          <h3>Upload UAV Image</h3>
          <label className="drone-file-control compact">
            <span>Select Image</span>
            <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileChange} />
          </label>
        </article>
        <article className={inputSource === 'camera' ? 'is-active' : ''}>
          <h3>Open Camera</h3>
          <div className="camera-controls">
            <button type="button" onClick={cameraStream ? stopCamera : startCamera}>
              {cameraStream ? 'Stop Camera' : 'Open Camera'}
            </button>
            <button type="button" onClick={captureCameraFrame} disabled={!cameraStream}>
              Capture Frame
            </button>
          </div>
          {cameraError ? <small>{cameraError}</small> : null}
        </article>
        <article className={inputSource === 'demo' ? 'is-active' : ''}>
          <h3>Use Demo Image</h3>
          <button type="button" className="cv-mode-button" onClick={useDemoImage}>
            Use Demo Image
          </button>
        </article>
      </div>

      {cameraStream ? (
        <video
          ref={videoRef}
          className="cv-camera-preview"
          autoPlay
          muted
          playsInline
        />
      ) : null}

      <canvas ref={canvasRef} className="browser-cv-canvas" aria-label="Browser detection preview" />

      <div className="browser-cv-summary">
        <div><span>Detected</span><strong>{detectedPeople}</strong></div>
        <div><span>Confirmed</span><strong>{counts.confirmed}</strong></div>
        <div><span>Candidates</span><strong>{counts.candidate}</strong></div>
        <div><span>Rejected</span><strong>{counts.rejected}</strong></div>
        <div><span>Review Required</span><strong>{reviewRequired ? 'Yes' : 'No'}</strong></div>
      </div>

      <div className="browser-cv-simple-actions">
        <button type="button" className="cv-run-detection" onClick={resetInput}>
          Run Again
        </button>
        <button type="button" className="cv-run-detection" onClick={onOpenTacticalMap}>
          Open Tactical Map
        </button>
      </div>

      <div className={`browser-cv-message ${counts.confirmed > 0 ? 'is-success' : counts.candidate > 0 ? 'is-warning' : ''}`}>
        {statusMessage}
      </div>

      {saveResult ? (
        <div className="browser-cv-save-result">
          <strong>Tactical fusion updated</strong>
          <span>{saveResult.fusion_path}</span>
        </div>
      ) : null}

      {error ? <div className="cv-backend-error"><strong>Error</strong><small>{error}</small></div> : null}

      <details className="advanced-toggle browser-cv-advanced">
        <summary>Advanced Details</summary>
        <div className="advanced-content">
          <div className="browser-cv-model">
            <label>
              Model
              <select value={modelName} onChange={(event) => setModelName(event.target.value)} disabled={isRunning}>
                <option value={DEFAULT_MODEL}>Xenova/detr-resnet-50</option>
                <option value={FALLBACK_MODEL}>Xenova/yolos-tiny</option>
              </select>
            </label>
          </div>

          <div className="cv-capability-status">
            <article className={backendOnline ? 'is-online' : 'is-offline'}>
              <span>Backend</span>
              <strong>{backendOnline ? 'Online' : 'Offline'}</strong>
            </article>
            <article className={browserCvStatus === 'Error' ? 'is-offline' : 'is-online'}>
              <span>Browser CV</span>
              <strong>{browserCvStatus}</strong>
            </article>
            <article className={cudaDataLoaded ? 'is-online' : 'is-offline'}>
              <span>CUDA Data</span>
              <strong>{cudaDataLoaded ? 'Loaded' : 'Missing'}</strong>
            </article>
            <article className={saveFuseStatus === 'Enabled' ? 'is-online' : 'is-offline'}>
              <span>Save & Fuse</span>
              <strong>{saveFuseStatus}</strong>
            </article>
          </div>

          <div className="cv-engine-selector">
            <label className={engineMode === 'browser' ? 'is-selected' : ''}>
              <input type="radio" name="cv-engine" value="browser" checked={engineMode === 'browser'} onChange={() => setEngineMode('browser')} />
              <strong>Browser Transformers.js</strong>
              <span>Recommended</span>
            </label>
            <label className={engineMode === 'yolo' ? 'is-selected' : ''}>
              <input type="radio" name="cv-engine" value="yolo" checked={engineMode === 'yolo'} onChange={() => setEngineMode('yolo')} />
              <strong>Python YOLO Backend</strong>
              <span>Optional fallback</span>
            </label>
            <label className={engineMode === 'demo' ? 'is-selected' : ''}>
              <input type="radio" name="cv-engine" value="demo" checked={engineMode === 'demo'} onChange={() => setEngineMode('demo')} />
              <strong>Human-reviewed Demo</strong>
              <span>Safe fallback</span>
            </label>
          </div>

          <div className="browser-cv-controls">
            <div className="cv-selected-input">
              <span>Input source</span>
              <strong>{inputLabel}</strong>
            </div>
            <button type="button" className="cv-run-detection" onClick={() => runDetection({ autoSave: false })} disabled={!selectedFile || isRunning || engineMode !== 'browser'}>
              {isRunning ? 'Running Browser Detection...' : 'Run Detection'}
            </button>
            <button type="button" className="cv-run-detection" onClick={() => saveDetections()} disabled={!payload || isSaving || !backendOnline}>
              {isSaving ? 'Saving...' : 'Save and Fuse'}
            </button>
            {!backendOnline ? <small>Start backend to save detections and update tactical fusion.</small> : null}
            {engineMode !== 'browser' ? <small>Switch to Browser Transformers.js to run the primary MVP detector.</small> : null}
          </div>

          <div className="browser-cv-status-grid">
            <div><span>Model status</span><strong>{modelStatus}</strong></div>
            <div><span>Progress</span><strong>{progress || 'Idle'}</strong></div>
            <div><span>Confirmed</span><strong>{counts.confirmed}</strong></div>
            <div><span>Candidates</span><strong>{counts.candidate}</strong></div>
            <div><span>Rejected</span><strong>{counts.rejected}</strong></div>
            <div><span>Review required</span><strong>{reviewRequired ? 'Yes' : 'No'}</strong></div>
          </div>

          <pre className="browser-cv-json-preview">
            {payload ? JSON.stringify(payload, null, 2) : 'No detection payload yet.'}
          </pre>
        </div>
      </details>
    </section>
  );
}
