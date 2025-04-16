document.getElementById('prediction-form').onsubmit = async (e) => {
  e.preventDefault();

  const formData = new FormData(e.target);
  const data = {};

  // Convert formData to JSON
  formData.forEach((value, key) => {
    data[key] = value;
  });

  // Fix encoder key mismatch for country
  if (data['country_of_res']) {
    data['contry_of_res'] = data['country_of_res'];
    delete data['country_of_res'];
  }

  try {
    const response = await fetch('http://127.0.0.1:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }

    const result = await response.json();

    // Display prediction result
    if (result.error) {
      document.getElementById('result').innerHTML = `<p class="text-danger">Error: ${result.error}</p>`;
    } else {
      document.getElementById('result').innerHTML = `<p class="text-success">Autism Prediction: ${result.prediction}</p>`;
    }
  } catch (error) {
    console.error('Error:', error);
    document.getElementById('result').innerHTML = `<p class="text-danger">An error occurred. Please try again.</p>`;
  }
};
