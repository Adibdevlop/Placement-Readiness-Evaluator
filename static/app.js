const form = document.querySelector('#predict-form');
const button = document.querySelector('#submit-button');
const errorBox = document.querySelector('#form-error');
const resultBox = document.querySelector('#results');

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  errorBox.textContent = '';
  button.disabled = true;
  button.innerHTML = 'Analysing your profile…';

  const payload = Object.fromEntries(new FormData(form).entries());
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(Object.values(data.errors || {}).join(' '));

    document.querySelector('#score').textContent = data.score;
    document.querySelector('#level').textContent = data.level;
    document.querySelector('.score-ring').dataset.tone = data.tone;
    document.querySelector('#recommendations').innerHTML = data.recommendations.map((item, index) => `
      <div class="action"><b>${String(index + 1).padStart(2, '0')}</b><span><strong>${item.area}</strong>${item.action}</span></div>
    `).join('');
    document.querySelector('#factors').innerHTML = data.top_factors.map(item => `
      <div class="factor"><span>${item.name}<b>${item.importance}%</b></span><i><em style="width:${item.importance}%"></em></i></div>
    `).join('');
    resultBox.classList.remove('hidden');
    resultBox.scrollIntoView({behavior: 'smooth', block: 'start'});
  } catch (error) {
    errorBox.textContent = error.message || 'Prediction failed. Please check the values and try again.';
  } finally {
    button.disabled = false;
    button.innerHTML = 'Generate my CareerLens report <span>→</span>';
  }
});

