const API_URL = "https://YOUR-RENDER-APP-NAME.onrender.com"; // TODO: Replace with actual Render URL

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const previewImg = document.getElementById('preview-img');
    const removeBtn = document.getElementById('remove-btn');
    const predictBtn = document.getElementById('predict-btn');
    const resultCard = document.getElementById('result-card');
    const loader = document.querySelector('.loader');
    const btnText = document.querySelector('.btn-text');

    let currentFile = null;

    // Drag and drop handlers
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        uploadArea.classList.add('dragover');
    }

    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }

    uploadArea.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    // Click handler
    uploadArea.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function () {
        handleFiles(this.files);
    });

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files[0];
            if (file.type.startsWith('image/')) {
                currentFile = file;
                showPreview(file);
                predictBtn.disabled = false;
                resultCard.classList.add('hidden');
            } else {
                alert('Please upload an image file.');
            }
        }
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = function () {
            previewImg.src = reader.result;
            imagePreview.classList.add('active');
            uploadArea.style.display = 'none';
        }
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent triggering upload click
        clearFile();
    });

    function clearFile() {
        currentFile = null;
        fileInput.value = '';
        imagePreview.classList.remove('active');
        uploadArea.style.display = 'block';
        predictBtn.disabled = true;
        resultCard.classList.add('hidden');
    }

    // Predict
    predictBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // Loading state
        predictBtn.disabled = true;
        btnText.style.display = 'none';
        loader.classList.remove('hidden');
        resultCard.classList.add('hidden');

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            // Use the full API URL for the fetch request
            const response = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                displayResult(data);
            } else {
                alert('Prediction failed: ' + (data.error || 'Unknown error'));
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction. Please ensure the backend is running and the API URL is correct.');
        } finally {
            // Reset button state
            predictBtn.disabled = false;
            btnText.style.display = 'block';
            loader.classList.add('hidden');
        }
    });

    function displayResult(data) {
        resultCard.classList.remove('hidden');

        // Update main result
        document.getElementById('prediction-text').textContent = data.prediction;
        document.getElementById('confidence-badge').textContent = `${data.confidence} Confidence`;

        // Update list
        const probList = document.getElementById('prob-list');
        probList.innerHTML = '';

        // Sort probabilities
        const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);

        entries.forEach(([label, score]) => {
            const item = document.createElement('div');
            item.className = 'prob-item';

            // Color based on score
            // Determine bar color intensity? no just use primary for now.

            item.innerHTML = `
                <div class="prob-label">${label}</div>
                <div class="prob-bar-container">
                    <div class="prob-bar" style="width: ${score}%"></div>
                </div>
                <div class="prob-value">${score.toFixed(1)}%</div>
            `;
            probList.appendChild(item);
        });

        // Scroll to result on mobile
        if (window.innerWidth < 768) {
            resultCard.scrollIntoView({ behavior: 'smooth' });
        }
    }
});
