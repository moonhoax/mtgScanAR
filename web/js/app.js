// js/app.js

const mindarDir = 'assets/mindar/';
const modelsDir = 'assets/models/';
const dataFile = 'cards.json'; // contains list of scanned cards

async function loadCards() {
  const response = await fetch(dataFile);
  if (!response.ok) throw new Error('Failed to fetch cards.json');
  return await response.json(); // returns an array of card objects
}

function createTargetEntity(card) {
  const entity = document.createElement('a-entity');
  entity.setAttribute('mindar-image-target', `targetIndex: ${card.index}`);

  const model = document.createElement('a-gltf-model');
  model.setAttribute('src', `${modelsDir}${card.model_file}`);
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

    // Use the first card's .mind file as the imageTargetSrc
    const mainMindFile = cards[0].mind_file;
    sceneEl.setAttribute('mindar-image', `imageTargetSrc: ${mindarDir}${mainMindFile}`);

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
