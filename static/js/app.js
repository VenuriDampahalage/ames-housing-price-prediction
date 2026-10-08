/**
 * Ames Housing Valuation AI - Client Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('valuation-form');
  const priceDisplay = document.getElementById('valuation-price');
  const priceRange = document.getElementById('valuation-range');
  const metricSqft = document.getElementById('metric-sqft');
  const metricMortgage = document.getElementById('metric-mortgage');
  const metricAge = document.getElementById('metric-age');
  const metricBaths = document.getElementById('metric-baths');
  const qualitySlider = document.getElementById('input-OverallQual');
  const qualityNum = document.getElementById('quality-num');
  const qualityLabel = document.getElementById('quality-label');
  const presetSelect = document.getElementById('preset-select');
  const toastContainer = document.getElementById('toast-container');

  const inputBsmt = document.getElementById('input-TotalBsmtSF');
  const input1st = document.getElementById('input-FirstFlrSF');
  const input2nd = document.getElementById('input-SecondFlrSF');
  const inputFullBath = document.getElementById('input-FullBath');
  const inputHalfBath = document.getElementById('input-HalfBath');
  const inputBsmtFullBath = document.getElementById('input-BsmtFullBath');
  const inputYearBuilt = document.getElementById('input-YearBuilt');
  const inputYrSold = document.getElementById('input-YrSold');

  let currentPrice = 208625;
  let debounceTimer = null;
  let presetsData = {};

  const QUALITY_NAMES = {
    1: 'Very Poor',
    2: 'Poor',
    3: 'Fair',
    4: 'Below Average',
    5: 'Average',
    6: 'Above Average',
    7: 'Good',
    8: 'Very Good',
    9: 'Excellent',
    10: 'Luxury'
  };

  function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    toastContainer.appendChild(toast);
    setTimeout(() => {
      if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 2800);
  }

  function updateSliderFill(val) {
    const min = parseFloat(qualitySlider.min) || 1;
    const max = parseFloat(qualitySlider.max) || 10;
    const pct = ((val - min) / (max - min)) * 100;
    qualitySlider.style.background = `linear-gradient(to right, #0f766e 0%, #0f766e ${pct}%, #e5e7eb ${pct}%, #e5e7eb 100%)`;
    qualityNum.textContent = val;
    qualityLabel.textContent = QUALITY_NAMES[val] || 'Good';
  }

  function updateLocalMetrics() {
    const full = parseFloat(inputFullBath.value) || 0;
    const half = parseFloat(inputHalfBath.value) || 0;
    const bsmtBaths = parseFloat(inputBsmtFullBath.value) || 0;
    const totalBaths = full + (0.5 * half) + bsmtBaths;
    metricBaths.textContent = totalBaths.toFixed(1);

    const yrBuilt = parseInt(inputYearBuilt.value) || 2000;
    const yrSold = parseInt(inputYrSold.value) || 2008;
    const age = Math.max(0, yrSold - yrBuilt);
    metricAge.textContent = `${age} yrs`;
  }

  function animatePrice(fromVal, toVal, duration = 300) {
    const start = performance.now();
    function step(timestamp) {
      const progress = Math.min((timestamp - start) / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const val = Math.round(fromVal + (toVal - fromVal) * ease);
      priceDisplay.textContent = `$${val.toLocaleString()}`;
      if (progress < 1) {
        requestAnimationFrame(step);
      } else {
        priceDisplay.textContent = `$${Math.round(toVal).toLocaleString()}`;
        currentPrice = toVal;
      }
    }
    requestAnimationFrame(step);
  }

  function getFormData() {
    const data = {};
    const formData = new FormData(form);
    for (let [key, val] of formData.entries()) {
      if (['OverallQual', 'GarageCars', 'FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath', 'TotRmsAbvGrd', 'YearBuilt', 'YrSold'].includes(key)) {
        data[key] = parseInt(val, 10);
      } else if (['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF'].includes(key)) {
        data[key] = parseFloat(val);
      } else {
        data[key] = val;
      }
    }
    return data;
  }

  async function calculateValuation() {
    updateLocalMetrics();
    const payload = getFormData();

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) return;

      const res = await response.json();
      const data = res.data;

      animatePrice(currentPrice, data.predicted_price);
      priceRange.textContent = `Range $${Math.round(data.price_low).toLocaleString()} – $${Math.round(data.price_high).toLocaleString()}`;
      metricSqft.textContent = `$${data.price_per_sqft.toFixed(2)}`;
      metricMortgage.textContent = `$${Math.round(data.monthly_mortgage_estimate).toLocaleString()}/mo`;
      metricAge.textContent = `${data.house_age} yrs`;
      metricBaths.textContent = data.total_bathrooms.toFixed(1);
    } catch (e) {
      console.warn('Valuation error:', e);
    }
  }

  function triggerLiveCalculation(isUserEdit = true) {
    if (isUserEdit && presetSelect.value !== 'custom') {
      presetSelect.value = 'custom';
    }
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      calculateValuation();
    }, 140);
  }

  function applyPreset(specs) {
    for (const [key, value] of Object.entries(specs)) {
      const input = form.elements[key];
      if (input) {
        input.value = value;
      }
    }
    updateSliderFill(specs.OverallQual || 7);
    updateLocalMetrics();
    calculateValuation();
  }

  // Load Presets from server
  async function loadPresets() {
    try {
      const res = await fetch('/api/presets');
      const data = await res.json();
      if (data.presets) {
        data.presets.forEach(p => {
          presetsData[p.id] = p;
        });
      }
    } catch (e) {
      console.warn('Could not load presets:', e);
    }
  }

  presetSelect.addEventListener('change', (e) => {
    const val = e.target.value;
    if (val === 'custom') return;
    const preset = presetsData[val];
    if (preset) {
      applyPreset(preset.specs);
      showToast(`Loaded ${preset.name}`);
    }
  });

  qualitySlider.addEventListener('input', (e) => {
    updateSliderFill(parseInt(e.target.value));
    triggerLiveCalculation(true);
  });

  form.querySelectorAll('input, select').forEach(elem => {
    if (elem.id === 'input-OverallQual') return;
    elem.addEventListener('input', () => triggerLiveCalculation(true));
    elem.addEventListener('change', () => triggerLiveCalculation(true));
  });

  // Action links
  document.getElementById('btn-copy-summary').addEventListener('click', () => {
    const payload = getFormData();
    const summary = `Ames Housing Valuation\nEstimated Value: ${priceDisplay.textContent}\n${priceRange.textContent}\nLiving Area: ${payload.GrLivArea} sq ft\nOverall Quality: ${payload.OverallQual}/10 (${qualityLabel.textContent})\nNeighborhood: ${payload.Neighborhood}\nGarage: ${payload.GarageCars} cars | Bathrooms: ${metricBaths.textContent}`;
    navigator.clipboard.writeText(summary).then(() => {
      showToast('Summary copied to clipboard');
    });
  });

  document.getElementById('btn-save-pdf').addEventListener('click', () => {
    window.print();
  });

  // Init
  loadPresets();
  updateSliderFill(parseInt(qualitySlider.value));
  updateLocalMetrics();
  calculateValuation();
});
