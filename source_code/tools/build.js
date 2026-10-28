const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'src');
const FONT_DIR = path.join(ROOT, 'node_modules', '@fontsource');
const LOGO = path.join(ROOT, 'assets', 'logo.png');

const fonts = [
  ['IBM Plex Sans', 400, 'normal', 'ibm-plex-sans/files/ibm-plex-sans-latin-400-normal.woff2'],
  ['IBM Plex Sans', 400, 'italic', 'ibm-plex-sans/files/ibm-plex-sans-latin-400-italic.woff2'],
  ['IBM Plex Sans', 500, 'normal', 'ibm-plex-sans/files/ibm-plex-sans-latin-500-normal.woff2'],
  ['IBM Plex Sans', 600, 'normal', 'ibm-plex-sans/files/ibm-plex-sans-latin-600-normal.woff2'],
  ['IBM Plex Sans', 700, 'normal', 'ibm-plex-sans/files/ibm-plex-sans-latin-700-normal.woff2'],
  ['IBM Plex Mono', 400, 'normal', 'ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2'],
  ['IBM Plex Mono', 500, 'normal', 'ibm-plex-mono/files/ibm-plex-mono-latin-500-normal.woff2'],
];
const fontCss = fonts.map(([family, weight, style, file]) => {
  const b64 = fs.readFileSync(path.join(FONT_DIR, file)).toString('base64');
  return `@font-face{font-family:"${family}";font-style:${style};font-weight:${weight};font-display:swap;src:url(data:font/woff2;base64,${b64}) format("woff2");}`;
}).join('\n');

const css = fs.readFileSync(path.join(SRC, 'styles.css'), 'utf8')
  .replace('/*FONTS*/', `${fontCss}\n[hidden]{display:none!important}`);

const jsFiles = ['data.js', 'engine.js', 'icons.js', 'app/core.js', 'app/screens_a.js', 'app/screens_b.js', 'app/screens_c.js', 'app/tour.js', 'app/main.js'];
const logo = `data:image/png;base64,${fs.readFileSync(LOGO).toString('base64')}`;
const js = ['/* MedIntel concept prototype. Icons: Font Awesome Free 6 via react-icons (CC BY 4.0). Fonts: IBM Plex Sans and IBM Plex Mono (SIL Open Font License 1.1). */', `window.MI_ASSETS = { logo: ${JSON.stringify(logo)} };`]
  .concat(jsFiles.map((f) => `/* ${f} */\n${fs.readFileSync(path.join(SRC, f), 'utf8')}`))
  .join('\n')
  .replace(/<\/script/gi, '<\\/script');

const page = fs.readFileSync(path.join(SRC, 'index.html'), 'utf8')
  .replace('/*CSS*/', () => css)
  .replace('/*JS*/', () => js);

const RESET = ':root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px);color-scheme:light}body{margin:0;font:14px system-ui,-apple-system,Segoe UI,sans-serif}img{max-width:100%}[hidden]{display:none!important}';
const standalone = `<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n<style>${RESET}</style>\n</head>\n<body>\n${page}\n</body>\n</html>\n`;

fs.mkdirSync(path.join(ROOT, 'dist'), { recursive: true });
fs.writeFileSync(path.join(ROOT, 'dist', 'artifact.html'), page);
fs.writeFileSync(path.join(ROOT, 'dist', 'MedIntel_Prototype.html'), standalone);
const kb = (s) => `${(Buffer.byteLength(s) / 1024).toFixed(0)} KB`;
console.log(JSON.stringify({ artifact: kb(page), standalone: kb(standalone) }));
