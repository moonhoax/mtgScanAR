// js/app.js

const FRONTEND_BASE = window.location.origin; // Frontend base URL (Vercel)
const BACKEND_BASE = 'https://mtg-ar-ac5ba1441f06.herokuapp.com/';

const STATIC_MINDAR_DIR = 'assets/mindar/';
const STATIC_MODELS_DIR = 'assets/models/';
const DYNAMIC_MINDAR_DIR = 'assets/mind/';
const DYNAMIC_MODELS_DIR = 'assets/models/'; // Adjust if you also have dynamic models on backend

const dataFile = 'cards.json'; // contains list of scanned cards

async function loadCards() {
  const response = await fetch(dataFile);
  if (!response.ok) throw new Error('Failed to fetch cards.json');
  return await response.json(); // returns an array of card objects
}

function getMindarFileURL(filename) {
  // Example heuristic:
  // Static files start with "static_" or just no prefix (your choice)
  // Dynamic generated files start with "generated_" or "dynamic_"
  
  if (filename.startsWith('static_') || filename.startsWith('mindar_') || !filename.startsWith('generated_')) {
    // Load from frontend static folder
    return `${FRONTEND_BASE}/${STATIC_MINDAR_DIR}${filename}`;
  } else {
    // Load from backend dynamic folder
    return `${BACKEND_BASE}/${DYNAMIC_MINDAR_DIR}${filename}`;
  }
}

function getModelFileURL(filename) {
  // For now, assume all models are static on frontend. Adjust if needed.
  return `${FRONTEND_BASE}/${STATIC_MODELS_DIR}${filename}`;
}

function createTargetEntity(card) {
  const entity = document.createElement('a-entity');
  
  // Use the proper mind file URL (static or dynamic)
  const mindFileUrl = getMindarFileURL(card.mind_file);

  entity.setAttribute('mindar-image-target', `targetIndex: ${card.index}`);
  // MindAR requires imageTargetSrc on <a-scene>, so mindFileUrl is used there
  // For each target entity, just set the index

  const model = document.createElement('a-gltf-model');
  const modelUrl = getModelFileURL(card.model_file);
  
  model.setAttribute('src', modelUrl);
  model.setAttribute('scale', '0.1 0.1 0.1'); // adjust as needed
  model.setAttribute('rotation', '0 180 0');
  entity.appendChild(model);

  return entity;
}

async function initAR() {
  try {
    const cards = await loadCards();
    const sceneEl = document.querySelector('a-scene[mindar-image]');

    if (cards.length === 0) {
      console.warn('No cards in cards.json');
      return;
    }

    // Use the first card's mind file URL as the imageTargetSrc on scene
    const mainMindFileUrl = getMindarFileURL(cards[0].mind_file);
    sceneEl.setAttribute('mindar-image', `imageTargetSrc: ${mainMindFileUrl}`);

    // Wait for scene to load before adding targets
    sceneEl.addEventListener('renderstart', () => {
      for (const card of cards) {
        const entity = createTargetEntity(card);
        sceneEl.appendChild(entity);
      }
    });

  } catch (error) {
    console.error('AR scene initialization failed:', error);
  }
}

window.addEventListener('load', () => {
  initAR();
});
