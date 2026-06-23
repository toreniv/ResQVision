import { useMemo, useRef, useState } from 'react';
import { pipeline, env } from '@huggingface/transformers';

env.allowLocalModels = false;

const CONFIRMED_MIN_CONF = 0.30;
const CANDIDATE_MIN_CONF = 0.08;
const DEFAULT_MODEL = 'Xenova/detr-resnet-50';
const FALLBACK_MODEL = 'Xenova/yolos-tiny';
const detectorCache = new Map();

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

export default function BrowserCvDetector({ backendOnline, onSaved }) {
  const canvasRef = useRef(null);
  const imageUrlRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [modelName, setModelName] = useState(DEFAULT_MODEL);
  const [modelStatus, setModelStatus] = useState('Not loaded');
  const [progress, setProgress] = useState('');
  const [payload, setPayload] = useState(null);
  const [saveResult, setSaveResult] = useState(null);
  const [error, setError] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const counts = useMemo(() => countsFromPayload(payload), [payload]);
  const reviewRequired = Boolean(payload?.metadata?.review_required);

  const handleFileChange = (event) => {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setPayload(null);
    setSaveResult(null);
    setError(null);
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const runDetection = async () => {
    if (!selectedFile) {
      setError('Upload a UAV image before running browser detection.');
      return;
    }

    setIsRunning(true);
    setSaveResult(null);
    setError(null);
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

      const { image, url } = await loadImage(selectedFile);
      if (imageUrlRef.current) URL.revokeObjectURL(imageUrlRef.current);
      imageUrlRef.current = url;

      console.log('[BROWSER_CV] selected file:', selectedFile.name, selectedFile.type, selectedFile.size);
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
    } catch (err) {
      setError(err.message || 'Browser detection failed.');
      setModelStatus('Detection failed');
    } finally {
      setIsRunning(false);
    }
  };

  const saveDetections = async () => {
    if (!payload) {
      setError('Run browser detection before saving.');
      return;
    }

    setIsSaving(true);
    setError(null);
    try {
      const previewImageBase64 = canvasRef.current.toDataURL('image/jpeg', 0.9);
      const response = await fetch('http://127.0.0.1:8000/api/browser-cv/save_detections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...payload,
          preview_base64: previewImageBase64,
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) {
        throw new Error(result.message || 'Could not save browser detections.');
      }
      setSaveResult(result);
      await onSaved?.();
    } catch (err) {
      setError(err.message || 'Could not save browser detections.');
    } finally {
      setIsSaving(false);
    }
  };

  const statusMessage = counts.confirmed > 0
    ? 'Confirmed browser detections found. Tactical Fusion can use confirmed targets.'
    : counts.candidate > 0
      ? 'Browser detection found low-confidence candidates only. Human review required. Tactical Command uses CUDA risk ranking fallback.'
      : payload
        ? 'No confirmed browser detections found. Tactical Command uses CUDA risk ranking fallback.'
        : 'Upload an image and run browser-side detection.';

  return (
    <section className="panel browser-cv-card">
      <div className="browser-cv-header">
        <div>
          <span>Primary MVP path</span>
          <h2>Browser Transformers.js Detection</h2>
          <p>Runs object detection locally in the browser, then saves structured detections through the backend for tactical fusion.</p>
        </div>
        <div className="browser-cv-model">
          <label>
            Model
            <select value={modelName} onChange={(event) => setModelName(event.target.value)} disabled={isRunning}>
              <option value={DEFAULT_MODEL}>Xenova/detr-resnet-50</option>
              <option value={FALLBACK_MODEL}>Xenova/yolos-tiny</option>
            </select>
          </label>
        </div>
      </div>

      <div className="browser-cv-controls">
        <label className="drone-file-control compact">
          <span>Upload UAV Image</span>
          <input type="file" accept="image/*" onChange={handleFileChange} />
        </label>
        <button type="button" className="cv-run-detection" onClick={runDetection} disabled={!selectedFile || isRunning}>
          {isRunning ? 'Running Browser Detection...' : 'Run Browser Detection'}
        </button>
        <button type="button" className="cv-run-detection" onClick={saveDetections} disabled={!payload || isSaving || !backendOnline}>
          {isSaving ? 'Saving...' : 'Save and Fuse'}
        </button>
        {!backendOnline ? <small>Start the local backend to save detections and update tactical fusion.</small> : null}
      </div>

      <div className="browser-cv-status-grid">
        <div><span>Model status</span><strong>{modelStatus}</strong></div>
        <div><span>Progress</span><strong>{progress || 'Idle'}</strong></div>
        <div><span>Confirmed</span><strong>{counts.confirmed}</strong></div>
        <div><span>Candidates</span><strong>{counts.candidate}</strong></div>
        <div><span>Rejected</span><strong>{counts.rejected}</strong></div>
        <div><span>Review required</span><strong>{reviewRequired ? 'Yes' : 'No'}</strong></div>
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

      <canvas ref={canvasRef} className="browser-cv-canvas" aria-label="Browser detection preview" />
    </section>
  );
}
