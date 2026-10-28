MEDINTEL CONCEPT PROTOTYPE
BE Computer Engineering, Project Stage I, JESITMR Nashik, AY 2026-27

QUICK START
1. If you received a .zip file, extract it first.
2. Double-click MedIntel_Prototype.html. It opens in your web browser
   (Google Chrome, Microsoft Edge or Safari). No internet connection
   or installation is needed.
3. Full screen: F11 on Windows, Ctrl+Cmd+F on a Mac.
4. Click "Guided demo" at the top right.
   Next step: right arrow key, Page Down or a presentation clicker.
   Back: left arrow key. Exit: Esc.

Reload the page to start again from a clean state.

The full guide, with the step-by-step demo script, things to try and
answers to likely questions, is in "How to Use the MedIntel Prototype.pdf".

FOLDER CONTENTS
MedIntel_Prototype.html                 The prototype (open this)
How to Use the MedIntel Prototype.pdf   The guide
MedIntel_Presentation.pptx              The project presentation
sample_data/morning_batch.csv           The sample batch used in the demo
sample_data/boundary_tests.csv          Extra patients for testing the rules
source_code/                            Optional: only needed to change the prototype

All patients are synthetic. This is a concept prototype, not a clinical tool.

FILES NOT UPLOADED TO GITHUB
Everything in the project folder was uploaded except these 4 files:

  .DS_Store
  MedIntel_Stage_II_handover/.DS_Store
  MedIntel_Stage_II_handover/medintel/.DS_Store
  MedIntel_Stage_II_handover/medintel/tests/.DS_Store

.DS_Store files are hidden files that macOS Finder creates on its own to
remember folder view settings (icon positions, sort order). They are not
part of the project and nothing uses them. The .gitignore file in this
repository keeps them out of future uploads.

Also note: MedIntel_Stage_II_handover/medintel/.gitignore lists files
that the Stage II code generates when you run it (for example .venv/,
trained model files in models/*.json, local databases *.db and the .env
file for API keys). None of these existed when the project was uploaded,
so nothing was left out because of them. If you create them later, they
will stay on your computer and will not be uploaded. Never upload .env,
because it holds secret API keys.
