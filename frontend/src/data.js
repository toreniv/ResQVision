export const missionBase = {
  name: 'ResQVision Guardian Shield',
  date: '2026-06-02',
  area: 'Sector Delta-7 / Forward Medical Corridor',
  soldierCount: 1000,
  resqBandCount: 936,
  medicalTeams: 6,
  uavAvailable: true,
  loraRelays: 12,
  evacuationZones: 4,
  readinessScore: 94
};

export const missionVariant = {
  name: 'ResQVision Night Relief',
  date: '2026-06-02',
  area: 'Sector Echo-3 / Urban Edge',
  soldierCount: 1000,
  resqBandCount: 948,
  medicalTeams: 7,
  uavAvailable: true,
  loraRelays: 14,
  evacuationZones: 5,
  readinessScore: 96
};

export const mapObjects = {
  uav: { x: 510, y: 130, label: 'UAV-1' },
  evacZones: [
    { x: 155, y: 185, label: 'EVAC A' },
    { x: 828, y: 255, label: 'EVAC B' },
    { x: 276, y: 805, label: 'EVAC C' },
    { x: 724, y: 788, label: 'EVAC D' }
  ],
  squads: [
    { x: 320, y: 320, label: 'Alpha' },
    { x: 540, y: 430, label: 'Bravo' },
    { x: 670, y: 620, label: 'Charlie' },
    { x: 215, y: 590, label: 'Delta' },
    { x: 820, y: 510, label: 'Echo' }
  ],
  relays: [
    { x: 230, y: 280, label: 'R1' },
    { x: 460, y: 250, label: 'R2' },
    { x: 720, y: 350, label: 'R3' },
    { x: 390, y: 680, label: 'R4' },
    { x: 615, y: 745, label: 'R5' }
  ],
  riskZones: [
    { x: 675, y: 450, size: 170, level: 'critical' },
    { x: 250, y: 645, size: 130, level: 'urgent' },
    { x: 455, y: 365, size: 110, level: 'urgent' }
  ]
};

export const soldiers = [
  { id: 'S-0412', name: 'M. Hale', x: 690, y: 466, category: 'critical', hr: 139, spo2: 82, risk: 0.982 },
  { id: 'S-0188', name: 'R. Novak', x: 650, y: 492, category: 'critical', hr: 132, spo2: 84, risk: 0.964 },
  { id: 'S-0774', name: 'A. Kim', x: 720, y: 430, category: 'critical', hr: 128, spo2: 86, risk: 0.951 },
  { id: 'S-0601', name: 'J. Ortiz', x: 255, y: 656, category: 'urgent', hr: 121, spo2: 89, risk: 0.894 },
  { id: 'S-0309', name: 'L. Singh', x: 462, y: 370, category: 'urgent', hr: 117, spo2: 91, risk: 0.861 },
  { id: 'S-0915', name: 'T. Brooks', x: 235, y: 608, category: 'urgent', hr: 113, spo2: 92, risk: 0.824 },
  { id: 'S-0440', name: 'E. Cohen', x: 810, y: 520, category: 'urgent', hr: 110, spo2: 93, risk: 0.771 },
  { id: 'S-0231', name: 'P. Morgan', x: 332, y: 326, category: 'stable', hr: 92, spo2: 96, risk: 0.524 },
  { id: 'S-0670', name: 'D. Park', x: 548, y: 422, category: 'stable', hr: 88, spo2: 97, risk: 0.481 },
  { id: 'S-0997', name: 'N. Reed', x: 676, y: 614, category: 'stable', hr: 84, spo2: 98, risk: 0.436 }
];

export const topTargets = soldiers
  .slice()
  .sort((a, b) => b.risk - a.risk)
  .map((soldier, index) => ({ ...soldier, rank: index + 1 }));

export const readinessCards = [
  { title: 'Communication Readiness', value: '97%', detail: 'LoRa mesh relay integrity nominal' },
  { title: 'Medical Readiness', value: '92%', detail: '6 trauma teams staged near EVAC corridors' },
  { title: 'Sensor Readiness', value: '94%', detail: '936 ResQBands streaming HR and SpO2' },
  { title: 'CUDA Processing Readiness', value: 'PASS', detail: 'Attention kernel validated against CPU baseline' }
];

export const benchmarkRows = [
  { soldiers: 128, cpu: 3.84, gpu: 0.42, speedup: 9.1 },
  { soldiers: 256, cpu: 14.72, gpu: 0.88, speedup: 16.7 },
  { soldiers: 512, cpu: 58.96, gpu: 1.94, speedup: 30.4 },
  { soldiers: 1000, cpu: 224.51, gpu: 5.86, speedup: 38.3 }
];

export const correctnessMetrics = {
  status: 'PASS',
  top10Overlap: '10/10',
  maxAbsError: '0.000031',
  meanAbsError: '0.000004',
  attentionEntropy: 'Low entropy rows identify concentrated casualty-risk attention around critical telemetry clusters.'
};

export const architectureSteps = [
  'Drone Observation',
  'YOLO Soldier Detection',
  'ResQBand Telemetry',
  'Feature Matrix',
  'CUDA Attention Engine',
  'Risk Assessment',
  'Evacuation Ranking',
  'Tactical Command Dashboard'
];
