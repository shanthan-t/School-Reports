document.addEventListener('DOMContentLoaded', () => {
    const svg = document.getElementById('map');
    const resetButton = document.getElementById('reset');
    let zoomed = false;
  
    svg.addEventListener('click', (e) => {
      if (e.target.classList.contains('state') && !zoomed) {
        const state = e.target;
        const bbox = state.getBBox(); // Get bounding box of the state
        const centerX = bbox.x + bbox.width / 2;
        const centerY = bbox.y + bbox.height / 2;
  
        // Apply zoom with scaling and translation
        svg.style.transition = 'transform 0.5s ease-in-out';
        svg.style.transformOrigin = `${centerX}px ${centerY}px`;
        svg.style.transform = `scale(3) translate(${-centerX}px, ${-centerY}px)`;
  
        zoomed = true;
      }
    });
  
    resetButton.addEventListener('click', () => {
      // Reset zoom to default
      svg.style.transition = 'transform 0.5s ease-in-out';
      svg.style.transform = 'scale(1) translate(0, 0)';
      zoomed = false;
    });
  });  