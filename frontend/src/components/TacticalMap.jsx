import { mapObjects, soldiers } from '../data.js';

const colors = {
  critical: '#dc2626',
  urgent: '#f97316',
  stable: '#16a34a'
};

function toPercent(value) {
  return `${value / 10}%`;
}

export default function TacticalMap({ planning = false, showArrows = false }) {
  const visibleSoldiers = soldiers;
  const topThree = soldiers.slice(0, 3);

  return (
    <div className={`tactical-map ${planning ? 'planning-map' : ''}`} aria-label="ResQVision tactical grid">
      <svg viewBox="0 0 1000 1000" preserveAspectRatio="xMidYMid slice" role="img">
        <defs>
          <pattern id={planning ? 'grid-plan' : 'grid-live'} width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#c7d9ee" strokeWidth="1" />
          </pattern>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="#2563eb" />
          </marker>
          <filter id="terrain-blur">
            <feGaussianBlur stdDeviation="3" />
          </filter>
        </defs>
        <rect width="1000" height="1000" fill="#f8fbff" />
        <g className="terrain-layer" opacity="0.52">
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
              r={zone.size / 2}
              fill={zone.level === 'critical' ? '#fee2e2' : '#ffedd5'}
              stroke={zone.level === 'critical' ? '#ef4444' : '#fb923c'}
              strokeWidth="4"
              strokeDasharray="12 10"
              opacity="0.8"
            />
            <text x={zone.x} y={zone.y + zone.size / 2 + 30} textAnchor="middle" className="map-label risk-label">
              {zone.level.toUpperCase()} RISK
            </text>
          </g>
        ))}

        {mapObjects.evacZones.map((zone) => (
          <g key={zone.label} className="tactical-marker evac-marker">
            <rect x={zone.x - 52} y={zone.y - 36} width="104" height="72" rx="10" fill="#dcfce7" stroke="#16a34a" strokeWidth="5" />
            <rect x={zone.x - 8} y={zone.y - 24} width="16" height="48" fill="#16a34a" rx="2" />
            <rect x={zone.x - 24} y={zone.y - 8} width="48" height="16" fill="#16a34a" rx="2" />
            <text x={zone.x} y={zone.y + 62} textAnchor="middle" className="map-label">{zone.label}</text>
          </g>
        ))}

        {mapObjects.relays.map((relay) => (
          <g key={relay.label} className="tactical-marker relay-marker">
            <polygon
              points={`${relay.x},${relay.y - 28} ${relay.x + 25},${relay.y - 14} ${relay.x + 25},${relay.y + 14} ${relay.x},${relay.y + 28} ${relay.x - 25},${relay.y + 14} ${relay.x - 25},${relay.y - 14}`}
              fill="#dbeafe"
              stroke="#2563eb"
              strokeWidth="5"
            />
            <path d={`M ${relay.x - 16} ${relay.y + 8} Q ${relay.x} ${relay.y - 18} ${relay.x + 16} ${relay.y + 8}`} fill="none" stroke="#2563eb" strokeWidth="4" />
            <circle cx={relay.x} cy={relay.y + 9} r="4" fill="#2563eb" />
            <text x={relay.x} y={relay.y + 52} textAnchor="middle" className="map-label">{relay.label}</text>
          </g>
        ))}

        {mapObjects.squads.map((squad) => (
          <g key={squad.label} className="tactical-marker squad-marker">
            <rect x={squad.x - 48} y={squad.y - 34} width="96" height="68" rx="8" fill="#eef6ff" stroke="#60a5fa" strokeWidth="5" />
            <path d={`M ${squad.x - 24} ${squad.y - 6} L ${squad.x} ${squad.y + 16} L ${squad.x + 24} ${squad.y - 6}`} fill="none" stroke="#2563eb" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
            <text x={squad.x} y={squad.y + 58} textAnchor="middle" className="map-label">{squad.label}</text>
          </g>
        ))}

        {showArrows
          ? topThree.map((target) => (
              <line
                key={`route-${target.id}`}
                x1={mapObjects.uav.x}
                y1={mapObjects.uav.y}
                x2={target.x}
                y2={target.y}
                stroke="#2563eb"
              strokeWidth="3"
              strokeDasharray="14 12"
              markerEnd="url(#arrow)"
              opacity="0.62"
            />
          ))
          : null}

        <g className="tactical-marker uav-marker">
          <path d={`M ${mapObjects.uav.x} ${mapObjects.uav.y - 46} L ${mapObjects.uav.x + 54} ${mapObjects.uav.y + 42} L ${mapObjects.uav.x} ${mapObjects.uav.y + 22} L ${mapObjects.uav.x - 54} ${mapObjects.uav.y + 42} Z`} fill="#2563eb" stroke="#ffffff" strokeWidth="6" />
          <path d={`M ${mapObjects.uav.x} ${mapObjects.uav.y - 22} L ${mapObjects.uav.x} ${mapObjects.uav.y + 32}`} stroke="#ffffff" strokeWidth="6" strokeLinecap="round" />
          <text x={mapObjects.uav.x} y={mapObjects.uav.y + 76} textAnchor="middle" className="map-label">{mapObjects.uav.label}</text>
        </g>

        {visibleSoldiers.map((soldier) => {
          const topRank = topThree.findIndex((target) => target.id === soldier.id) + 1;
          const isTopTarget = topRank > 0;

          return (
          <g key={soldier.id} className={`casualty-marker ${isTopTarget ? 'top-casualty' : ''}`}>
            {isTopTarget ? (
              <>
                <circle cx={soldier.x} cy={soldier.y} r="42" fill={colors[soldier.category]} opacity="0.08" />
                <circle cx={soldier.x} cy={soldier.y} r="31" fill="none" stroke={colors[soldier.category]} strokeWidth="4" opacity="0.72" />
              </>
            ) : null}
            <circle cx={soldier.x} cy={soldier.y} r={isTopTarget ? 16 : 10} fill={colors[soldier.category]} stroke="#ffffff" strokeWidth={isTopTarget ? 5 : 3} />
            {isTopTarget ? (
              <g>
                <circle cx={soldier.x + 32} cy={soldier.y - 30} r="17" fill="#ffffff" stroke="#0f2747" strokeWidth="3" />
                <text x={soldier.x + 32} y={soldier.y - 23} textAnchor="middle" className="map-label rank-label">
                  {topRank}
                </text>
                <text x={soldier.x} y={soldier.y - 46} textAnchor="middle" className="map-label casualty-label top-label">{soldier.id}</text>
              </g>
            ) : null}
          </g>
          );
        })}
      </svg>
      <div className="map-coordinates">
        <span>Grid 1000 x 1000</span>
        <span>UAV {toPercent(mapObjects.uav.x)} / {toPercent(mapObjects.uav.y)}</span>
      </div>
    </div>
  );
}
