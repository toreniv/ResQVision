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
  const visibleSoldiers = planning ? [] : soldiers;
  const topThree = soldiers.slice(0, 3);

  return (
    <div className={`tactical-map ${planning ? 'planning-map' : ''}`} aria-label="ResQVision tactical grid">
      <svg viewBox="0 0 1000 1000" role="img">
        <defs>
          <pattern id={planning ? 'grid-plan' : 'grid-live'} width="50" height="50" patternUnits="userSpaceOnUse">
            <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#c7d9ee" strokeWidth="1" />
          </pattern>
          <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
            <path d="M0,0 L0,6 L9,3 z" fill="#2563eb" />
          </marker>
        </defs>
        <rect width="1000" height="1000" fill="#f8fbff" />
        <rect width="1000" height="1000" fill={`url(#${planning ? 'grid-plan' : 'grid-live'})`} />

        {mapObjects.riskZones.map((zone) => (
          <circle
            key={`${zone.x}-${zone.y}`}
            cx={zone.x}
            cy={zone.y}
            r={zone.size / 2}
            fill={zone.level === 'critical' ? '#fee2e2' : '#ffedd5'}
            stroke={zone.level === 'critical' ? '#ef4444' : '#fb923c'}
            strokeWidth="3"
            opacity="0.78"
          />
        ))}

        {mapObjects.evacZones.map((zone) => (
          <g key={zone.label}>
            <rect x={zone.x - 42} y={zone.y - 28} width="84" height="56" rx="12" fill="#dcfce7" stroke="#16a34a" strokeWidth="4" />
            <text x={zone.x} y={zone.y + 5} textAnchor="middle" className="map-label">{zone.label}</text>
          </g>
        ))}

        {mapObjects.relays.map((relay) => (
          <g key={relay.label}>
            <circle cx={relay.x} cy={relay.y} r="18" fill="#dbeafe" stroke="#2563eb" strokeWidth="4" />
            <text x={relay.x} y={relay.y + 5} textAnchor="middle" className="map-label">{relay.label}</text>
          </g>
        ))}

        {mapObjects.squads.map((squad) => (
          <g key={squad.label}>
            <circle cx={squad.x} cy={squad.y} r="30" fill="#eef6ff" stroke="#60a5fa" strokeWidth="4" />
            <text x={squad.x} y={squad.y + 6} textAnchor="middle" className="map-label">{squad.label}</text>
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
                strokeWidth="5"
                strokeDasharray="14 12"
                markerEnd="url(#arrow)"
                opacity="0.72"
              />
            ))
          : null}

        <g>
          <path d={`M ${mapObjects.uav.x} ${mapObjects.uav.y - 34} l 42 68 h -84 z`} fill="#2563eb" />
          <text x={mapObjects.uav.x} y={mapObjects.uav.y + 60} textAnchor="middle" className="map-label">{mapObjects.uav.label}</text>
        </g>

        {visibleSoldiers.map((soldier) => (
          <g key={soldier.id}>
            <circle cx={soldier.x} cy={soldier.y} r="20" fill={colors[soldier.category]} stroke="#ffffff" strokeWidth="6" />
            <text x={soldier.x} y={soldier.y - 30} textAnchor="middle" className="map-label">{soldier.id}</text>
          </g>
        ))}
      </svg>
      <div className="map-coordinates">
        <span>Grid 1000 x 1000</span>
        <span>UAV {toPercent(mapObjects.uav.x)} / {toPercent(mapObjects.uav.y)}</span>
      </div>
    </div>
  );
}
