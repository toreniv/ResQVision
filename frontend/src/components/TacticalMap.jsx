import { useMemo, useRef, useState } from 'react';
import { mapObjects } from '../data.js';

const colors = {
  critical: '#dc2626',
  urgent: '#f97316',
  stable: '#16a34a'
};

function toPercent(value) {
  return `${value / 10}%`;
}

const MAP_SIZE = 1000;
const MIN_VIEW_SIZE = 300;
const MAX_VIEW_SIZE = 1000;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function clampViewBox(nextViewBox) {
  const w = clamp(nextViewBox.w, MIN_VIEW_SIZE, MAX_VIEW_SIZE);
  const h = clamp(nextViewBox.h, MIN_VIEW_SIZE, MAX_VIEW_SIZE);
  return {
    x: clamp(nextViewBox.x, 0, MAP_SIZE - w),
    y: clamp(nextViewBox.y, 0, MAP_SIZE - h),
    w,
    h
  };
}

export default function TacticalMap({ planning = false, showArrows = false, soldiers = [], attentionData = [], fusionMode = null }) {
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: MAP_SIZE, h: MAP_SIZE });
  const [isPanning, setIsPanning] = useState(false);
  const panStartRef = useRef(null);
  const mapRef = useRef(null);
  const svgRef = useRef(null);
  const visibleSoldiers = soldiers.slice(0, 10);
  const topThree = visibleSoldiers.slice(0, 3);

  const safeAttentionData = Array.isArray(attentionData) ? attentionData : [];

  const attentionById = useMemo(() => {
    const map = {};
    safeAttentionData.forEach(row => {
      map[String(row.soldier_id)] = {
        maxAttention: Number(row.max_attention),
        entropy: Number(row.entropy),
      };
    });
    return map;
  }, [safeAttentionData]);

  const attentionThresholds = useMemo(() => {
    if (!safeAttentionData.length) return { high: 0.8, medium: 0.5 };
    const scores = safeAttentionData
      .map(r => Number(r.max_attention))
      .filter(Number.isFinite)
      .sort((a, b) => b - a);

    if (!scores.length) return { high: 0.8, medium: 0.5 };

    return {
      high:   scores[Math.min(2, scores.length - 1)], // top 3 → high
      medium: scores[Math.min(5, scores.length - 1)], // top 4-6 → medium
    };
  }, [safeAttentionData]);

  function getAttentionLevel(soldierId, attentionById, thresholds) {
    const entry = attentionById[String(soldierId)];
    if (!entry) return 'none';
    const score = entry.maxAttention;
    if (score >= thresholds.high)   return 'high';
    if (score >= thresholds.medium) return 'medium';
    return 'low';
  }

  const HALO = {
    high:   { r: 22, fill: '#ef4444', opacity: 0.20 }, // red   – won't clash with markers
    medium: { r: 16, fill: '#f97316', opacity: 0.15 }, // orange
    low:    { r: 10, fill: '#3b82f6', opacity: 0.12 }, // blue  – distinct from green stable markers
  };

  const zoomMap = (factor) => {
    setViewBox((current) => {
      const nextW = clamp(current.w * factor, MIN_VIEW_SIZE, MAX_VIEW_SIZE);
      const nextH = clamp(current.h * factor, MIN_VIEW_SIZE, MAX_VIEW_SIZE);
      const centerX = current.x + current.w / 2;
      const centerY = current.y + current.h / 2;

      return clampViewBox({
        x: centerX - nextW / 2,
        y: centerY - nextH / 2,
        w: nextW,
        h: nextH
      });
    });
  };

  const zoomAtPoint = (clientX, clientY, zoomFactor) => {
    if (!svgRef.current) {
      return;
    }

    setViewBox((current) => {
      const rect = svgRef.current.getBoundingClientRect();
      const relativeX = clamp((clientX - rect.left) / rect.width, 0, 1);
      const relativeY = clamp((clientY - rect.top) / rect.height, 0, 1);
      const pointX = current.x + relativeX * current.w;
      const pointY = current.y + relativeY * current.h;
      const nextW = clamp(current.w * zoomFactor, MIN_VIEW_SIZE, MAX_VIEW_SIZE);
      const nextH = clamp(current.h * zoomFactor, MIN_VIEW_SIZE, MAX_VIEW_SIZE);

      return clampViewBox({
        x: pointX - relativeX * nextW,
        y: pointY - relativeY * nextH,
        w: nextW,
        h: nextH
      });
    });
  };

  const resetView = () => {
    setViewBox({ x: 0, y: 0, w: MAP_SIZE, h: MAP_SIZE });
  };

  const handleKeyDown = (event) => {
    if (event.key === '+' || event.key === '=') {
      event.preventDefault();
      zoomMap(0.88);
    } else if (event.key === '-') {
      event.preventDefault();
      zoomMap(1.12);
    } else if (event.key === '0') {
      event.preventDefault();
      resetView();
    }
  };

  const handleWheel = (event) => {
    event.preventDefault();
    zoomAtPoint(event.clientX, event.clientY, event.deltaY < 0 ? 0.88 : 1.12);
  };

  const handlePointerDown = (event) => {
    if (event.target.closest('button')) {
      return;
    }

    panStartRef.current = {
      clientX: event.clientX,
      clientY: event.clientY,
      viewBox
    };
    setIsPanning(true);
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event) => {
    if (!panStartRef.current || !svgRef.current) {
      return;
    }

    const rect = svgRef.current.getBoundingClientRect();
    const start = panStartRef.current;
    const deltaX = event.clientX - start.clientX;
    const deltaY = event.clientY - start.clientY;
    const unitsPerPixelX = start.viewBox.w / rect.width;
    const unitsPerPixelY = start.viewBox.h / rect.height;

    setViewBox(
      clampViewBox({
        ...start.viewBox,
        x: start.viewBox.x - deltaX * unitsPerPixelX,
        y: start.viewBox.y - deltaY * unitsPerPixelY
      })
    );
  };

  const endPan = (event) => {
    panStartRef.current = null;
    setIsPanning(false);

    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  return (
    <div
      ref={mapRef}
      className={`tactical-map ${planning ? 'planning-map' : ''} ${isPanning ? 'is-panning' : ''}`}
      aria-label="ResQVision tactical grid"
      tabIndex={0}
      onKeyDown={handleKeyDown}
      onWheel={handleWheel}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={endPan}
      onPointerCancel={endPan}
      onPointerLeave={endPan}
    >
      <p className="map-help-text">Ctrl + wheel / Ctrl +/- to zoom · drag to pan</p>
      {fusionMode === 'YOLO_LIVE' ? <div className="yolo-fusion-badge">YOLO LIVE FUSION</div> : null}
      <div className="map-zoom-controls" aria-label="Map zoom controls">
        <button type="button" onClick={() => zoomMap(0.88)}>+</button>
        <button type="button" onClick={() => zoomMap(1.12)}>-</button>
        <button type="button" onClick={resetView}>Reset</button>
      </div>
      <svg ref={svgRef} viewBox={`${viewBox.x} ${viewBox.y} ${viewBox.w} ${viewBox.h}`} preserveAspectRatio="xMidYMid meet" role="img">
        <defs>
          <pattern id={planning ? 'grid-plan' : 'grid-live'} width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#dce9f6" strokeWidth="0.7" />
          </pattern>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">
            <path d="M0,0 L0,6 L8,3 z" fill="#3b82f6" />
          </marker>
          <filter id="terrain-blur">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>
        <rect width="1000" height="1000" fill="#f8fbff" />
        <g className="terrain-layer" opacity="0.2">
          <path d="M40 88 C170 40 240 110 338 76 S520 42 650 88 820 94 965 52 L1000 0 L0 0 Z" fill="#f5f1e8" />
          <path d="M65 785 C190 736 300 820 420 760 S650 710 790 780 922 842 1000 792 L1000 1000 L0 1000 Z" fill="#f3efe4" />
          <path d="M42 612 C148 520 245 590 336 548 S512 442 642 510 782 562 940 490" fill="none" stroke="#c8e4ef" strokeWidth="15" opacity="0.42" filter="url(#terrain-blur)" />
          <path d="M128 204 C238 156 320 244 456 196 S642 120 804 192 930 168 992 134" fill="none" stroke="#d8ccb5" strokeWidth="9" opacity="0.34" />
          <path d="M104 718 C248 660 350 704 486 664 S704 602 886 642" fill="none" stroke="#d8ccb5" strokeWidth="8" opacity="0.34" />
          {[120, 188, 305, 462, 534, 640, 746, 858, 930].map((x, index) => (
            <g key={`tree-${x}`} fill="#9ed59c" opacity="0.48">
              <circle cx={x} cy={index % 2 ? 192 : 150} r="18" />
              <circle cx={x + 24} cy={index % 2 ? 218 : 172} r="15" />
              <circle cx={x - 18} cy={index % 2 ? 230 : 188} r="12" />
            </g>
          ))}
          {[92, 168, 352, 448, 608, 704, 824, 918].map((x, index) => (
            <g key={`brush-${x}`} fill="#8bcf82" opacity="0.52">
              <circle cx={x} cy={index % 2 ? 682 : 806} r="20" />
              <circle cx={x + 28} cy={index % 2 ? 704 : 830} r="14" />
              <circle cx={x - 20} cy={index % 2 ? 720 : 846} r="12" />
            </g>
          ))}
          {[260, 585, 835].map((x, index) => (
            <g key={`structure-${x}`} fill="#c7cdd3" stroke="#aab3bd" opacity="0.46">
              <rect x={x} y={index === 1 ? 418 : 520} width="34" height="28" rx="4" />
              <rect x={x + 40} y={index === 1 ? 400 : 540} width="38" height="32" rx="4" />
              <rect x={x - 28} y={index === 1 ? 444 : 548} width="28" height="26" rx="4" />
            </g>
          ))}
        </g>
        <rect width="1000" height="1000" fill={`url(#${planning ? 'grid-plan' : 'grid-live'})`} />
        <rect x="1" y="1" width="998" height="998" fill="none" stroke="#b9cfe6" strokeWidth="2" />

        {mapObjects.riskZones.map((zone) => (
          <g key={`${zone.x}-${zone.y}`}>
            <circle
              cx={zone.x}
              cy={zone.y}
              r={zone.size / 5.4}
              fill={zone.level === 'critical' ? '#fee2e2' : '#ffedd5'}
              stroke={zone.level === 'critical' ? '#ef4444' : '#fb923c'}
              strokeWidth="1.5"
              strokeDasharray="8 10"
              opacity="0.34"
            />
            <text x={zone.x} y={zone.y + zone.size / 5.4 + 16} textAnchor="middle" className="map-label risk-label">
              {zone.level.toUpperCase()} RISK
            </text>
          </g>
        ))}

        {mapObjects.evacZones.map((zone) => (
          <g key={zone.label} className="tactical-marker evac-marker">
            <rect x={zone.x - 22} y={zone.y - 15} width="44" height="30" rx="5" fill="#ecfdf5" stroke="#22c55e" strokeWidth="2" />
            <rect x={zone.x - 3.5} y={zone.y - 10} width="7" height="20" fill="#16a34a" rx="1.5" />
            <rect x={zone.x - 10} y={zone.y - 3.5} width="20" height="7" fill="#16a34a" rx="1.5" />
            <text x={zone.x} y={zone.y + 28} textAnchor="middle" className="map-label evac-label">{zone.label}</text>
          </g>
        ))}

        {mapObjects.relays.map((relay) => (
          <g key={relay.label} className="tactical-marker relay-marker">
            <polygon
              points={`${relay.x},${relay.y - 10} ${relay.x + 10},${relay.y - 5} ${relay.x + 10},${relay.y + 5} ${relay.x},${relay.y + 10} ${relay.x - 10},${relay.y + 5} ${relay.x - 10},${relay.y - 5}`}
              fill="#eff6ff"
              stroke="#3b82f6"
              strokeWidth="1.6"
            />
            <path d={`M ${relay.x - 6} ${relay.y + 3} Q ${relay.x} ${relay.y - 7} ${relay.x + 6} ${relay.y + 3}`} fill="none" stroke="#2563eb" strokeWidth="1.5" />
            <circle cx={relay.x} cy={relay.y + 3.5} r="2" fill="#2563eb" />
            <text x={relay.x} y={relay.y + 22} textAnchor="middle" className="map-label relay-label">{relay.label}</text>
          </g>
        ))}

        {mapObjects.squads.map((squad) => (
          <g key={squad.label} className="tactical-marker squad-marker">
            <rect x={squad.x - 20} y={squad.y - 14} width="40" height="28" rx="5" fill="#f5f9ff" stroke="#93c5fd" strokeWidth="1.8" />
            <path d={`M ${squad.x - 10} ${squad.y - 2} L ${squad.x} ${squad.y + 7} L ${squad.x + 10} ${squad.y - 2}`} fill="none" stroke="#2563eb" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
            <text x={squad.x} y={squad.y + 25} textAnchor="middle" className="map-label squad-label">{squad.label}</text>
          </g>
        ))}

        {showArrows
          ? topThree.map((target) => (
              <line
                key={`route-${target.id}`}
                x1={mapObjects.uav.x}
                y1={mapObjects.uav.y}
                x2={target.x}
                y2={target.source === 'YOLO' ? target.y : MAP_SIZE - target.y}
                stroke="#3b82f6"
              strokeWidth="1.4"
              strokeDasharray="8 12"
              markerEnd="url(#arrow)"
              opacity="0.28"
            />
          ))
          : null}

        <g className="tactical-marker uav-marker">
          <path d={`M ${mapObjects.uav.x} ${mapObjects.uav.y - 19} L ${mapObjects.uav.x + 23} ${mapObjects.uav.y + 18} L ${mapObjects.uav.x} ${mapObjects.uav.y + 10} L ${mapObjects.uav.x - 23} ${mapObjects.uav.y + 18} Z`} fill="#2563eb" stroke="#ffffff" strokeWidth="2.2" />
          <path d={`M ${mapObjects.uav.x} ${mapObjects.uav.y - 9} L ${mapObjects.uav.x} ${mapObjects.uav.y + 14}`} stroke="#ffffff" strokeWidth="2.2" strokeLinecap="round" />
          <text x={mapObjects.uav.x} y={mapObjects.uav.y + 33} textAnchor="middle" className="map-label uav-label">{mapObjects.uav.label}</text>
        </g>

        {/* ATTENTION HALOS – rendered before markers so they sit behind */}
        {visibleSoldiers.map((soldier) => {
          const sx = soldier.x;
          const sy = soldier.source === 'YOLO' ? soldier.y : MAP_SIZE - soldier.y;
          const level = getAttentionLevel(soldier.id, attentionById, attentionThresholds);
          
          if (level === 'none') return null;
          
          const halo = HALO[level];
          return (
            <circle
              key={`halo-${soldier.id}`}
              cx={sx}
              cy={sy}
              r={halo.r}
              fill={halo.fill}
              opacity={halo.opacity}
            />
          );
        })}

        {visibleSoldiers.map((soldier) => {
          const topRank = topThree.findIndex((target) => target.id === soldier.id) + 1;
          const isTopTarget = topRank > 0;
          const sx = soldier.x;
          const sy = soldier.source === 'YOLO' ? soldier.y : MAP_SIZE - soldier.y;

          return (
          <g key={soldier.id} className={`casualty-marker ${isTopTarget ? 'top-casualty' : ''}`}>
            {isTopTarget ? (
              <>
                <circle cx={sx} cy={sy} r="17" fill={colors[soldier.category]} opacity="0.05" />
                <circle cx={sx} cy={sy} r="13.5" fill="none" stroke={colors[soldier.category]} strokeWidth="1.8" opacity="0.5" />
              </>
            ) : null}
            <circle cx={sx} cy={sy} r={isTopTarget ? 7.5 : 4} fill={colors[soldier.category]} stroke="#ffffff" strokeWidth={isTopTarget ? 2.2 : 1.2} />
            {isTopTarget ? (
              <g>
                <circle cx={sx + 15} cy={sy - 15} r="8" fill="#ffffff" stroke="#0f2747" strokeWidth="1.2" />
                <text x={sx + 15} y={sy - 12} textAnchor="middle" className="map-label rank-label">
                  {topRank}
                </text>
                <text x={sx} y={sy - 22} textAnchor="middle" className="map-label casualty-label top-label">{soldier.id}</text>
              </g>
            ) : null}
          </g>
          );
        })}

        <g transform="translate(20, 920)">
          <text fontSize="11" fill="#64748b" fontWeight="600" letterSpacing="1">ATTENTION</text>
          <circle cx="8"  cy="18" r="6" fill="#ef4444" opacity="0.6" />
          <text x="18" y="22" fontSize="10" fill="#64748b">High</text>
          <circle cx="8"  cy="34" r="6" fill="#f97316" opacity="0.6" />
          <text x="18" y="38" fontSize="10" fill="#64748b">Medium</text>
          <circle cx="8"  cy="50" r="6" fill="#3b82f6" opacity="0.6" />
          <text x="18" y="54" fontSize="10" fill="#64748b">Low</text>
        </g>
      </svg>
      <div className="map-coordinates">
        <span>Grid 1000 x 1000</span>
        <span>UAV {toPercent(mapObjects.uav.x)} / {toPercent(mapObjects.uav.y)}</span>
      </div>
    </div>
  );
}
