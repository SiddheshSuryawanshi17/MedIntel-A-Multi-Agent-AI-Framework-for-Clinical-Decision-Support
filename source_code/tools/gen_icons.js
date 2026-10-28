const fs = require('fs');
const path = require('path');
const React = require('react');
const ReactDOMServer = require('react-dom/server');
const fa = require('react-icons/fa6');

const NAMES = {
  home: 'FaHouse', upload: 'FaFileArrowUp', run: 'FaCircleNodes', triage: 'FaListCheck', report: 'FaFileMedical',
  whatif: 'FaSliders', governance: 'FaShieldHalved', compare: 'FaCodeCompare', about: 'FaCircleInfo', play: 'FaPlay',
  moon: 'FaMoon', sun: 'FaSun', menu: 'FaBars', close: 'FaXmark', check: 'FaCheck', warn: 'FaTriangleExclamation',
  question: 'FaCircleQuestion', ban: 'FaBan', up: 'FaArrowUp', down: 'FaArrowDown', lock: 'FaLock', clock: 'FaClock',
  download: 'FaDownload', copy: 'FaCopy', trace: 'FaRoute', reset: 'FaRotateLeft', right: 'FaChevronRight',
  left: 'FaChevronLeft', chevdown: 'FaChevronDown', heart: 'FaHeartPulse', clinician: 'FaUserDoctor',
  curator: 'FaUserShield', sysadmin: 'FaUserGear', dataadmin: 'FaUserPen', database: 'FaDatabase',
  okcircle: 'FaCircleCheck', xcircle: 'FaCircleXmark', annotate: 'FaPenToSquare', flag: 'FaFlag', arrow: 'FaArrowRight',
  keyboard: 'FaKeyboard', ml: 'FaBrain', rules: 'FaScaleBalanced', retrieval: 'FaBookMedical', context: 'FaComments',
  fusion: 'FaGaugeHigh', audit: 'FaFileWaveform', csv: 'FaFileCsv', layers: 'FaLayerGroup', search: 'FaMagnifyingGlass',
  eye: 'FaEye', users: 'FaUsers', grad: 'FaGraduationCap', filter: 'FaFilter', notes: 'FaNotesMedical',
  branch: 'FaCodeBranch', gear: 'FaGear', target: 'FaBullseye', quote: 'FaQuoteLeft', forward: 'FaForward',
  pause: 'FaPause', bolt: 'FaBolt', inbox: 'FaInbox', box: 'FaBoxArchive', stethoscope: 'FaStethoscope',
  ingest: 'FaFileImport', robot: 'FaRobot', userblank: 'FaUser', list: 'FaListUl', plus: 'FaPlus', expand: 'FaUpRightAndDownLeftFromCenter',
};
const out = {};
for (const [key, comp] of Object.entries(NAMES)) {
  const C = fa[comp];
  if (!C) throw new Error(`missing icon ${comp}`);
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(C, {}));
  const vb = svg.match(/viewBox="([^"]+)"/)[1];
  const inner = svg.replace(/^<svg[^>]*>/, '').replace(/<\/svg>$/, '');
  out[key] = [vb, inner];
}
const js = `(function (root) {\n  'use strict';\n  const ICONS = ${JSON.stringify(out)};\n  root.MedIntelIcons = ICONS;\n})(typeof window !== 'undefined' ? window : globalThis);\n`;
fs.writeFileSync(path.join(__dirname, '..', 'src', 'icons.js'), js);
console.log(Object.keys(out).length, 'icons,', js.length, 'bytes');
