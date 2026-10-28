MEDINTEL PROTOTYPE: SOURCE CODE (OPTIONAL)

You only need this folder to change the prototype. It builds the single
file MedIntel_Prototype.html.

Requirements: Node.js 18 or later (https://nodejs.org).

1. Open a terminal in this folder and run:  npm install
2. Edit the files in src:
     src/data.js      patients, rules, knowledge passages, model weights
     src/engine.js    validation, agents, fusion (Equation 4.2), reports
     src/app/         the screens and the guided demo (tour.js)
     src/styles.css   colours and layout
3. Build:  npm run build
   The new file is written to dist/MedIntel_Prototype.html
4. Check the engine:  npm test

Fonts: IBM Plex Sans and IBM Plex Mono (SIL Open Font License 1.1).
Icons: Font Awesome Free 6 via react-icons (CC BY 4.0).
Both are bundled into the built file.
